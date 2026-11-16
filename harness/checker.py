"""Late checks: write a step's checks just before it runs, not at planning time.

The planner writes its checks before any work has happened and without having seen a
single file, so it guesses. Here the model can open the files first and sees what the
earlier steps found out. The planner's checks are kept as a draft to correct.

If this call fails for any reason, the step keeps the draft checks.
"""

import json

from .checks import allowed_types, clean_checks, describe
from .client import Ollama
from .context import _clip, workspace_listing
from .loop import tool_loop
from .planner import CHECK_TYPES
from .store import Store
from .tools import Toolbox

SYSTEM = """You write the automatic checks for ONE step of a larger task, just before another assistant carries the step out.
Afterwards the system runs your checks itself to decide whether the step was really done.
You may look at the files first with the tools. Do not do the step yourself.

Rules for checks:
- A check must be FALSE now and TRUE as soon as this step alone is done correctly.
- Never use a condition that only a later step makes true.
- Base text on what is really in the files. Do not guess names or lines you have not seen.
- If the step renames or removes something, use file_lacks for the old text.
- If the step only looks at things (reads files, runs failing tests to see the errors), it changes nothing: give one file_exists check on a file it looks at and nothing else.
- 1 to 3 checks. Types:
{check_types}
  Paths are relative to the workspace.

When you are ready, call set_checks. Never answer in plain text."""


def _schema(allowed: set[str]) -> dict:
    props = {
        "type": {"type": "string", "enum": [t for t in CHECK_TYPES if t in allowed]},
        "path": {"type": "string", "description": "File path, for the file checks."},
        "text": {"type": "string", "description": "Exact text, for file_contains and file_lacks."},
    }
    if "command_succeeds" in allowed:
        props["command"] = {"type": "string", "description": "Shell command, for command_succeeds."}
    if "sources_cited" in allowed:
        props["min"] = {"type": "integer", "description": "Fewest source URLs required, for sources_cited."}
    return {
        "type": "function",
        "function": {
            "name": "set_checks",
            "description": "Give the checks for this step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checks": {"type": "array", "items": {"type": "object", "properties": props, "required": ["type"]}},
                },
                "required": ["checks"],
            },
        },
    }


def write_checks(
    store: Store, client: Ollama, toolbox: Toolbox, run_id: int, step_id: int, tool_names: list[str],
    max_tool_calls: int = 3, on_tool=None,
) -> list[dict] | None:
    """Rewrite the step's checks with the files in view. Returns the new checks, or None if the draft was kept."""
    def log(kind: str, payload: dict) -> None:
        store.event(run_id, step_id, kind, {"phase": "late_checks", **payload})

    run, step = store.get_run(run_id), store.get_step(step_id)
    allowed = allowed_types(tool_names)
    draft = json.loads(step["checks"] or "[]")
    done = [s for s in store.list_steps(run_id) if s["status"] == "done"][-4:]
    facts = list(dict.fromkeys(store.list_facts(run_id)))[-12:]

    parts = [
        f"Overall task (for context only): {run['goal']}",
        f"Files in the workspace now:\n{workspace_listing(run['workspace'])}",
    ]
    if done:
        parts.append("Steps already done:\n" + "\n".join(
            f"- {_clip(s['description'], 110)} -> {_clip(s['summary'], 160)}" for s in done))
    if facts:
        parts.append("Known facts:\n" + "\n".join(f"- {_clip(f, 200)}" for f in facts))
    parts.append(f"The step that is about to run: {step['description']}")
    if step["done_when"]:
        parts.append(f"It is complete when: {step['done_when']}")
    if draft:
        parts.append(
            "Draft checks from the plan, written earlier WITHOUT seeing any file. Keep, fix or replace them:\n"
            + "\n".join(f"- {describe(c)}" for c in draft)
        )
    prompt = "\n\n".join(parts)
    system = SYSTEM.format(check_types="\n".join(line for t, line in CHECK_TYPES.items() if t in allowed))
    log("prompt", {"system": system, "user": prompt})

    def acceptable(args: dict) -> str | None:
        kept, _ = clean_checks(args.get("checks"), allowed)
        return None if kept else "give at least one well-formed check."

    # Look, don't touch: read-only tools and no commands, so writing checks cannot change anything.
    viewer = Toolbox(toolbox.workspace, commands="off", read_only=True, web=False)
    loop = tool_loop(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        viewer.schemas() + [_schema(allowed)],
        viewer.call,
        "set_checks",
        acceptable,
        max_tool_calls,
        log,
        on_tool,
    )
    if loop.args is None:
        store.event(run_id, step_id, "late_checks", {"kept_draft": True, "reason": loop.reason, "draft": draft})
        return None
    checks, dropped = clean_checks(loop.args.get("checks"), allowed)
    store.set_step_checks(step_id, checks)
    store.event(run_id, step_id, "late_checks", {"draft": draft, "checks": checks, "dropped": dropped})
    return checks
