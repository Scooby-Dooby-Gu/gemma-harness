"""Planner: turn a goal into small steps, and re-plan the remainder when a step fails."""

import json

from .checks import allowed_types, clean_checks
from .memory import memory_block
from .client import Ollama
from .context import _clip, workspace_listing
from .store import Store

MAX_STEPS = 8

SYSTEM = """You break a task into small steps for an assistant that works on one step at a time.
The assistant remembers nothing between steps except short notes, so every step must make sense on its own.

Rules:
- Use between 1 and {max_steps} steps. Prefer fewer; do not pad.
- Each step is one concrete action that names the files involved, and is small enough to finish in about 5 tool actions.
- Each step needs "done_when": something that can be checked afterwards, such as a file existing with certain content.
- Each step needs "checks": 1 to 3 conditions the system will test automatically right after the step. Types:
{check_types}
  Paths are relative to the workspace. Only check what this step itself makes true, not what later steps will do.
  A check must be true as soon as THIS step is done. A step that only looks (reads files, runs failing tests to see the errors) fixes nothing, so never give it a check that needs the fix, such as the tests passing; give it a file_exists check instead.
  For file_contains, use short text that will literally be in THAT file after the step. A file that only calls a function does not contain its `def` line.
- Put the steps in the order they must happen. Do not add steps for things that are already true.
- The assistant can only use these tools: {tools}. Do not plan steps that need anything else.
{extra_rules}
Reply by calling submit_plan."""

CHECK_TYPES = {
    "file_exists": '  - {"type": "file_exists", "path": "app.py"}  the file exists and is not empty',
    "file_contains": '  - {"type": "file_contains", "path": "app.py", "text": "def main"}  '
    "a short exact string that must appear in the file, such as a function name",
    "file_lacks": '  - {"type": "file_lacks", "path": "app.py", "text": "old_name"}  '
    "the text must be gone from the file; use it on every step that renames or removes something",
    "command_succeeds": '  - {"type": "command_succeeds", "command": "python3 -m unittest test_app.py"}  '
    "the command exits with code 0; the strongest check, use it for tests",
    "sources_cited": '  - {"type": "sources_cited", "path": "answer.md", "min": 2}  '
    "the file gives at least that many source URLs, and every URL in it was really read; use it for research answers",
}

WEB_RULES = """- For research: one step searches and reads pages and records what it found as facts, then a later step writes the answer to a file, with the URL of each page used.
- An answer may only cite pages whose content was shown: pages loaded with fetch_url, or search results that came with a page extract. A bare title and link is not a source.
"""


def submit_plan_schema(allowed: set[str]) -> dict:
    types = [t for t in CHECK_TYPES if t in allowed]
    check_props = {
        "type": {"type": "string", "enum": types},
        "path": {"type": "string", "description": "File path, for the file checks."},
        "text": {"type": "string", "description": "Exact text, for file_contains and file_lacks."},
    }
    if "command_succeeds" in allowed:
        check_props["command"] = {"type": "string", "description": "Shell command, for command_succeeds."}
    if "sources_cited" in allowed:
        check_props["min"] = {"type": "integer", "description": "Fewest source URLs required, for sources_cited."}
    return {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit the ordered list of steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string", "description": "What to do, naming the files."},
                                "done_when": {"type": "string", "description": "How to check the step is complete."},
                                "checks": {
                                    "type": "array",
                                    "description": "Conditions the system tests automatically after the step.",
                                    "items": {"type": "object", "properties": check_props, "required": ["type"]},
                                },
                            },
                            "required": ["description", "done_when", "checks"],
                        },
                    }
                },
                "required": ["steps"],
            },
        },
    }


class PlanError(RuntimeError):
    pass


def defer_repeated_commands(steps: list[dict]) -> list[dict]:
    """Drop a command check from a step when a later step carries the same one.

    The same command on an early and a late step is the task's end condition (usually
    "the tests pass"). On the early step it cannot be true yet, and it would get good
    work rejected. Returns the checks that were removed.
    """
    removed = []
    for i, step in enumerate(steps):
        later = {
            c["command"] for s in steps[i + 1:] for c in s["checks"] if c["type"] == "command_succeeds"
        }
        keep = [c for c in step["checks"] if not (c["type"] == "command_succeeds" and c["command"] in later)]
        removed += [{"step": i + 1, **c} for c in step["checks"] if c not in keep]
        step["checks"] = keep
    return removed


def _validate(args, allowed: set[str]) -> tuple[list[dict], list]:
    if isinstance(args, str):
        args = json.loads(args)
    steps = args.get("steps") if isinstance(args, dict) else None
    if isinstance(steps, str):
        steps = json.loads(steps)
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty list.")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"too many steps ({len(steps)}); use at most {MAX_STEPS}.")
    clean, dropped = [], []
    for i, s in enumerate(steps, 1):
        if not isinstance(s, dict) or not str(s.get("description", "")).strip():
            raise ValueError(f"step {i} needs a description.")
        # Malformed checks are dropped rather than failing the plan; the step then
        # falls back to the model inspector.
        checks, bad = clean_checks(s.get("checks"), allowed)
        dropped.extend(bad)
        clean.append(
            {
                "description": str(s["description"]).strip(),
                "done_when": str(s.get("done_when", "")).strip(),
                "checks": checks,
            }
        )
    return clean, dropped


def _ask(client: Ollama, store: Store, run_id: int, kind: str, tool_names: list[str], user: str) -> list[dict]:
    allowed = allowed_types(tool_names)
    system, schema = _system(tool_names), submit_plan_schema(allowed)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    problem = "no reply"
    for _ in range(4):
        resp = client.chat(messages, tools=[schema])
        msg = resp.get("message", {})
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
        call = next((c for c in calls if c.get("function", {}).get("name") == "submit_plan"), None)
        if call is None:
            problem = "you did not call submit_plan."
            # Keep the unusable reply, so a failed plan can be diagnosed afterwards.
            store.event(run_id, None, "model", {
                "phase": kind, "content": msg.get("content", ""), "thinking": msg.get("thinking", ""),
                "tool_calls": calls, "done_reason": resp.get("done_reason", ""),
                "tokens_out": resp.get("eval_count", 0),
            })
            # It often tries to start the work instead. Answer each such call, then say what is wanted.
            for c in calls:
                messages.append({
                    "role": "tool", "tool_name": c.get("function", {}).get("name", ""),
                    "content": "Error: no tools can be used while planning. Do not do the work now.",
                })
            messages.append({
                "role": "user",
                "content": "You are only planning. Do not read files or run anything yet: put that in the steps. "
                           "Call submit_plan now with the list of steps.",
            })
            continue
        try:
            steps, dropped = _validate(call["function"].get("arguments"), allowed)
            deferred = defer_repeated_commands(steps)
        except (ValueError, json.JSONDecodeError) as e:
            problem = str(e)
            messages.append(
                {"role": "tool", "tool_name": "submit_plan", "content": f"Error: {problem} Call submit_plan again."}
            )
            continue
        store.event(
            run_id,
            None,
            kind,
            {
                "user": user,
                "steps": steps,
                "dropped_checks": dropped,
                "deferred_checks": deferred,
                "thinking": msg.get("thinking", ""),
                "tokens_in": resp.get("prompt_eval_count", 0),
                "tokens_out": resp.get("eval_count", 0),
            },
        )
        return steps
    raise PlanError(f"The model did not produce a usable plan: {problem}")


def _system(tool_names: list[str]) -> str:
    allowed = allowed_types(tool_names)
    types = [line for t, line in CHECK_TYPES.items() if t in allowed]
    return SYSTEM.format(
        max_steps=MAX_STEPS,
        tools=", ".join(tool_names),
        check_types="\n".join(types),
        extra_rules=WEB_RULES if "fetch_url" in tool_names else "",
    )


def make_plan(client: Ollama, store: Store, run_id: int, tool_names: list[str], memory: bool = True) -> list[dict]:
    run = store.get_run(run_id)
    user = f"Task: {run['goal']}\n\nFiles in the workspace now:\n{workspace_listing(run['workspace'])}"
    remembered = memory_block(store, run_id, run["goal"], for_planner=True) if memory else ""
    if remembered:
        user += "\n\n" + remembered
    return _ask(client, store, run_id, "plan", tool_names, user)


def replan(
    client: Ollama, store: Store, run_id: int, failed_step_id: int, tool_names: list[str], memory: bool = True
) -> list[dict]:
    """Plan the remaining work after a step failed."""
    run = store.get_run(run_id)
    steps = store.list_steps(run_id)
    failed = next(s for s in steps if s["id"] == failed_step_id)
    done = [s for s in steps if s["status"] == "done"]
    facts = list(dict.fromkeys(store.list_facts(run_id)))[-20:]

    parts = [f"Task: {run['goal']}", f"Files in the workspace now:\n{workspace_listing(run['workspace'])}"]
    if done:
        parts.append(
            "Steps already completed (do not repeat them):\n"
            + "\n".join(f"- {_clip(s['description'], 110)} -> {_clip(s['summary'], 160)}" for s in done)
        )
    parts.append(
        f"This step FAILED:\n- {failed['description']}\n  What happened: {_clip(failed['summary'], 400)}"
    )
    if facts:
        parts.append("Known facts:\n" + "\n".join(f"- {_clip(f, 200)}" for f in facts))
    remembered = memory_block(store, run_id, failed["description"], for_planner=False) if memory else ""
    if remembered:
        parts.append(remembered)
    parts.append(
        "Plan the remaining work to finish the task. If the step failed only because of an automatic check, ask whether that check could ever be true for that step alone; if not, do not attach it again. Approach the failed step differently, "
        "or split it into smaller steps."
    )
    return _ask(client, store, run_id, "replan", tool_names, "\n\n".join(parts))
