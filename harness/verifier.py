"""Verifier: check a step the executor claims is done, without trusting the claim.

Three layers, cheapest first:
1. Evidence the harness recorded itself (did any tool call actually succeed?).
2. Code checks the planner attached to the step (file exists, file contains text,
   command exits 0), run by the harness with no model involved. If any ran, they decide.
3. A separate model call with a clean context and read-only tools, which inspects the
   workspace. It never sees the executor's summary. It runs when no code check could run,
   when the step changed files but every passing check was already true before it started,
   and when the only failures are "file contains text" checks: that text was guessed by
   the planner before the work existed, so a mismatch can be the plan's mistake.
"""

import json
from dataclasses import dataclass

from .checks import run_checks
from .client import Ollama
from .executor import StepResult
from .loop import tool_loop
from .store import Store
from .tools import Toolbox

SYSTEM = """You check another assistant's work. It says a step is complete. Find out whether that is true by inspecting the workspace with the tools.
Do not assume anything. Open the files and look. If a file that should exist is missing, or its content does not do what the step requires, the step fails.
Judge only this one step, not the whole task. Small style issues are not failures.
When you have checked, call the verdict tool. Never answer in plain text."""

VERDICT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "verdict",
        "description": "Give the result of your check.",
        "parameters": {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean", "description": "true only if you confirmed the step is complete."},
                "reason": {"type": "string", "description": "One or two sentences on what you checked and found."},
            },
            "required": ["passed", "reason"],
        },
    },
}


@dataclass
class Verdict:
    passed: bool
    reason: str


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _check_verdict(args: dict) -> str | None:
    if _as_bool(args.get("passed")) is None:
        return "passed must be true or false."
    return None


def describe_changes(changes: dict) -> str:
    lines = [f"{label}: {', '.join(paths[:20])}" for label, paths in changes.items() if paths]
    return "\n".join(lines) if lines else "No files were created, changed or deleted."


def verify_step(
    store: Store,
    client: Ollama,
    toolbox: Toolbox,
    run_id: int,
    step_id: int,
    result: StepResult,
    changes: dict,
    max_tool_calls: int = 5,
    on_tool=None,
    always_inspect: bool = False,
    already_true: list[dict] | None = None,
) -> Verdict:
    def log(kind: str, payload: dict) -> None:
        store.event(run_id, step_id, kind, {"phase": "verify", **payload})

    def done(passed: bool, reason: str) -> Verdict:
        store.event(run_id, step_id, "verify", {"passed": passed, "reason": reason, "changes": changes})
        return Verdict(passed, reason)

    # Layer 1: a step cannot be done if nothing it tried actually worked.
    if result.ok_calls == 0:
        return done(False, "The step was reported done, but no tool call succeeded, so nothing was actually done.")

    # Layer 2: code checks from the plan.
    run, step = store.get_run(run_id), store.get_step(step_id)
    results = run_checks(toolbox, json.loads(step["checks"] or "[]"), store.fetched_urls(run_id))
    if results:
        store.event(
            run_id, step_id, "checks",
            {
                "results": [{"check": r.check, "status": r.status, "detail": r.detail} for r in results],
                "already_true_before_step": already_true or [],
            },
        )
    failed = [r for r in results if r.status == "failed"]
    passed = [r for r in results if r.status == "passed"]
    failure = "Automatic check failed: " + " Also: ".join(r.detail for r in failed)
    disputed = bool(failed) and all(r.check["type"] == "file_contains" for r in failed)
    if failed and not disputed:
        return done(False, failure)
    # Only a check that was false before the step and true after it shows the step did something.
    proved = [r for r in passed if r.check not in (already_true or [])]
    if proved and not failed and not always_inspect:
        return done(True, f"Passed {len(proved)} automatic check(s): " + " ".join(r.detail for r in proved))
    if passed and not failed and not always_inspect and not any(changes.values()):
        # A step that only looked at things. Its checks prove nothing, and there is no change to inspect;
        # whether the task is finished is decided by the final check of the goal.
        return done(True, "The step changed no files and its checks were already true before it ran, "
                          "so it is accepted as a look-only step without proof.")

    # Layer 3: an independent look at the workspace by the model.
    inspector = Toolbox(toolbox.workspace, commands=toolbox.commands, read_only=True, web=toolbox.web)
    parts = [
        f"Overall task (for context only): {run['goal']}",
        f"Step to check: {step['description']}",
    ]
    if step["done_when"]:
        parts.append(f"The step is complete when: {step['done_when']}")
    parts.append(f"What the system recorded during the step:\n{describe_changes(changes)}")
    if inspector.commands != "off":
        parts.append("You may run commands, such as tests, to check the work.")
    if passed and not proved:
        parts.append(
            "The step's automatic checks were already true before it started, so they prove nothing. "
            "Look at what actually changed and decide whether it does what the step requires."
        )
    if disputed:
        parts.append(
            "An automatic check did not pass: " + " Also: ".join(r.detail for r in failed) + "\n"
            "That expected text was written before the work was done and may be a wrong guess "
            "(for example, text that belongs in a different file). Read the file and decide from the "
            "step and its completion condition, not from the expected text, whether the step is complete."
        )
    prompt = "\n\n".join(parts)
    log("prompt", {"system": SYSTEM, "user": prompt})

    loop = tool_loop(
        client,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        inspector.schemas() + [VERDICT_SCHEMA],
        inspector.call,
        "verdict",
        _check_verdict,
        max_tool_calls,
        log,
        on_tool,
    )
    reason = str((loop.args or {}).get("reason", "")).strip()
    if disputed:
        if loop.args is not None and _as_bool(loop.args["passed"]):
            store.event(run_id, step_id, "warning", {"warning": f"check overruled by inspection. {failure}"})
            return done(True, f"A text check did not match, but inspection found the step complete: {reason}")
        return done(False, failure + (f" Inspection agreed: {reason}" if reason else ""))
    if loop.args is None:
        return done(False, f"The check could not be completed: {loop.reason}")
    return done(_as_bool(loop.args["passed"]), reason)


GOAL_SYSTEM = """You check whether a whole task has really been completed. Another assistant worked on it and says it is finished.
Do not assume anything. Inspect the workspace with the tools: open the files, and run the tests or the program if you can.
The task is complete only if everything it asks for is true right now. Work that was only looked at, or only partly done, is not complete.
Small style issues are not failures.
When you have checked, call the verdict tool. If it is not complete, say exactly what is still missing or wrong. Never answer in plain text."""


def verify_goal(store: Store, client: Ollama, toolbox: Toolbox, run_id: int, max_tool_calls: int = 6, on_tool=None) -> Verdict:
    """The last gate before a run may end as done: is the goal itself met, whatever the steps said?

    Step checks only prove each step did what the plan asked. If the plan left out the
    real work, every step can pass while the task is unfinished.
    """
    def log(kind: str, payload: dict) -> None:
        store.event(run_id, None, kind, {"phase": "final", **payload})

    run = store.get_run(run_id)
    inspector = Toolbox(toolbox.workspace, commands=toolbox.commands, read_only=True, web=False)
    prompt = f"Task to check: {run['goal']}"
    if inspector.commands != "off":
        prompt += "\n\nYou may run commands, such as tests, to check the work."
    log("prompt", {"system": GOAL_SYSTEM, "user": prompt})
    loop = tool_loop(
        client,
        [{"role": "system", "content": GOAL_SYSTEM}, {"role": "user", "content": prompt}],
        inspector.schemas() + [VERDICT_SCHEMA],
        inspector.call,
        "verdict",
        _check_verdict,
        max_tool_calls,
        log,
        on_tool,
    )
    if loop.args is None:
        verdict = Verdict(False, f"The final check could not be completed: {loop.reason}")
    else:
        verdict = Verdict(_as_bool(loop.args["passed"]), str(loop.args.get("reason", "")).strip())
    store.event(run_id, None, "verify", {"final": True, "passed": verdict.passed, "reason": verdict.reason})
    return verdict
