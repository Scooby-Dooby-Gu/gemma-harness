"""Runner: plan a goal, then execute and verify the steps one at a time."""

import json
import time

from .checks import describe
from .client import ModelError, Ollama
from .executor import StepResult, run_step
from .memory import consolidate
from .planner import PlanError, make_plan, replan
from .store import Store
from .tools import Toolbox, diff_snapshots
from .verifier import verify_step


def attempt_step(
    store: Store,
    client: Ollama,
    toolbox: Toolbox,
    run_id: int,
    step_id: int,
    max_tool_calls: int = 8,
    max_attempts: int = 2,
    verify: bool = True,
    report=print,
    on_tool=None,
    always_inspect: bool = False,
    memory: bool = True,
) -> StepResult:
    """Execute one step, verify a 'done' claim, and record the outcome.

    A step only counts as done, and its facts are only saved, once the check passes.
    A rejected step goes back in the queue until it has used up its attempts.
    """
    before = toolbox.snapshot()
    result = run_step(store, client, toolbox, run_id, step_id, max_tool_calls, on_tool, memory)
    report(f"  {result.status.upper()} ({result.tool_calls} tool calls): {result.summary}")

    if result.status == "done" and verify:
        changes = diff_snapshots(before, toolbox.snapshot())
        verdict = verify_step(
            store, client, toolbox, run_id, step_id, result, changes, on_tool=on_tool, always_inspect=always_inspect
        )
        report(f"  CHECK {'PASSED' if verdict.passed else 'FAILED'}: {verdict.reason}")
        if not verdict.passed:
            result = StepResult("rejected", verdict.reason, [], result.tool_calls, result.ok_calls)

    if result.status == "rejected":
        note = f"It was reported done, but a check found: {result.summary}"
        if store.get_step(step_id)["attempts"] < max_attempts:
            store.requeue_step(step_id, note)
        else:
            store.finish_step(step_id, "failed", note)
            result.status = "failed"
    else:
        store.finish_step(step_id, result.status, result.summary)
        if result.status == "done":
            for fact in result.facts:
                store.add_fact(run_id, step_id, fact)
    return result


def run(
    store: Store,
    client: Ollama,
    toolbox: Toolbox,
    run_id: int,
    max_tool_calls: int = 8,
    max_replans: int = 2,
    max_steps_run: int = 20,
    max_attempts: int = 2,
    max_minutes: float = 30,
    verify: bool = True,
    report=print,
    on_tool=None,
    always_inspect: bool = False,
    memory: bool = True,
) -> str:
    """Drive a run to completion. Safe to call again on an interrupted run. Returns the final status.

    The limits (re-plans, step attempts, minutes) apply to this session: resuming a run starts them afresh.
    """
    tool_names = [s["function"]["name"] for s in toolbox.schemas()]
    deadline = time.monotonic() + max_minutes * 60
    executed = 0
    replans = 0
    store.set_run_status(run_id, "running")
    store.reset_running_steps(run_id)  # a step left 'running' means the last session was interrupted

    def stop(status: str, why: str) -> str:
        store.set_run_status(run_id, status)
        store.event(run_id, None, "run_end", {"status": status, "reason": why})
        report(f"\nRun {run_id} {status.upper()}: {why}")
        # Distil lasting notes once the work is over. A problem here never changes the run's outcome.
        if memory and executed and status in ("done", "failed"):
            try:
                for note in consolidate(client, store, run_id):
                    report(f"  remembered {note}")
            except Exception as e:
                store.event(run_id, None, "error", {"error": f"memory consolidation failed: {e}"})
        return status

    def add_steps(steps: list[dict]) -> None:
        for s in steps:
            store.add_step(run_id, s["description"], s["done_when"], s.get("checks"))
        for s in store.list_steps(run_id):
            if s["status"] == "pending":
                report(f"  {s['position']}. {s['description']}\n       done when: {s['done_when']}")
                for c in json.loads(s["checks"]):
                    report(f"       check: {describe(c)}")

    try:
        if not store.list_steps(run_id):
            report("Planning…")
            add_steps(make_plan(client, store, run_id, tool_names, memory))

        while True:
            step = store.next_pending_step(run_id)
            if step is None:
                return stop("done", "all steps completed" + (" and checked." if verify else "."))
            if executed >= max_steps_run:
                return stop("failed", f"stopped after {max_steps_run} step attempts.")
            if time.monotonic() > deadline:
                return stop("failed", f"stopped at the {max_minutes:g}-minute time limit; resume to continue.")
            executed += 1

            again = " (retry)" if step["attempts"] else ""
            report(f"\nStep {step['position']}{again}: {step['description']}")
            result = attempt_step(
                store, client, toolbox, run_id, step["id"], max_tool_calls, max_attempts, verify, report, on_tool,
                always_inspect, memory,
            )

            if result.status == "blocked":
                return stop("blocked", f"step {step['position']} needs your input: {result.summary}")
            if result.status == "rejected":
                report("  Trying the step again with the check's finding.")
            if result.status == "failed":
                if replans >= max_replans:
                    return stop("failed", f"step {step['position']} failed and the re-plan limit is reached.")
                replans += 1
                report("  Re-planning the remaining work…")
                new_steps = replan(client, store, run_id, step["id"], tool_names, memory)
                store.skip_pending_steps(run_id)
                add_steps(new_steps)
    except (PlanError, ModelError) as e:
        store.event(run_id, None, "error", {"error": str(e)})
        return stop("failed", str(e))
    except KeyboardInterrupt:
        store.reset_running_steps(run_id)
        executed = 0  # no consolidation on an interrupted session
        return stop("failed", "interrupted; resume to continue.")
