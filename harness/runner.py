"""Runner: plan a goal, then execute and verify the steps one at a time."""

import json
import time

from .checker import write_checks
from .checks import describe, run_checks
from .client import ModelError, Ollama
from .executor import StepResult, run_step
from .memory import consolidate
from .planner import PlanError, make_plan, replan
from .store import Store
from .tools import Toolbox, diff_snapshots
from .verifier import verify_goal, verify_step


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
    late_checks: bool = True,
) -> StepResult:
    """Execute one step, verify a 'done' claim, and record the outcome.

    A step only counts as done, and its facts are only saved, once the check passes.
    A rejected step goes back in the queue until it has used up its attempts.
    """
    # On a step's first attempt, rewrite the plan's checks now that the files can be looked at.
    if verify and late_checks and store.get_step(step_id)["attempts"] == 0:
        tool_names = [s["function"]["name"] for s in toolbox.schemas()]
        try:
            written = write_checks(store, client, toolbox, run_id, step_id, tool_names, on_tool=on_tool)
        except ModelError as e:
            written = None
            store.event(run_id, step_id, "error", {"error": f"late checks failed: {e}"})
        for c in written or []:
            report(f"  check: {describe(c)}")
    before = toolbox.snapshot()
    # Checks that are already true before the work starts cannot show that the work was done.
    already_true = []
    if verify:
        checks = json.loads(store.get_step(step_id)["checks"] or "[]")
        already_true = [r.check for r in run_checks(toolbox, checks, store.fetched_urls(run_id)) if r.status == "passed"]
    result = run_step(store, client, toolbox, run_id, step_id, max_tool_calls, on_tool, memory)
    report(f"  {result.status.upper()} ({result.tool_calls} tool calls): {result.summary}")

    if result.status == "done" and verify:
        changes = diff_snapshots(before, toolbox.snapshot())
        verdict = verify_step(
            store, client, toolbox, run_id, step_id, result, changes, on_tool=on_tool, always_inspect=always_inspect,
            already_true=already_true,
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
    final_check: bool = True,
    late_checks: bool = True,
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
                if not (verify and final_check):
                    return stop("done", "all steps completed" + (" and checked." if verify else "."))
                # Every step passing does not prove the task is finished: the plan may have left work out.
                report("\nFinal check of the whole task…")
                verdict = verify_goal(store, client, toolbox, run_id, on_tool=on_tool)
                outcome = "PASSED" if verdict.passed else "UNDECIDED" if verdict.undecided else "FAILED"
                report(f"  FINAL CHECK {outcome}: {verdict.reason}")
                if verdict.undecided:
                    # Every step passed its own check, and nothing was found wrong. Not grounds for more work.
                    store.event(run_id, None, "warning", {"warning": f"final check undecided. {verdict.reason}"})
                    return stop("done", "all steps completed and checked, but the final check of the whole task "
                                        "could not reach a verdict.")
                if verdict.passed:
                    return stop("done", "all steps completed, and the finished task was checked.")
                # Recorded as a failed step, so it shows in the log and the re-planner sees the finding.
                gate = store.add_step(run_id, "Final check: is the whole task complete?", "The task's goal is fully met.")
                store.start_step(gate)
                store.finish_step(gate, "failed", f"All planned steps passed, but the task is NOT complete: {verdict.reason}")
                if replans >= max_replans or time.monotonic() > deadline:
                    return stop("failed", f"the final check failed: {verdict.reason}")
                replans += 1
                report("  Planning the missing work…")
                add_steps(replan(client, store, run_id, gate, tool_names, memory))
                continue
            if executed >= max_steps_run:
                return stop("failed", f"stopped after {max_steps_run} step attempts.")
            if time.monotonic() > deadline:
                return stop("failed", f"stopped at the {max_minutes:g}-minute time limit; resume to continue.")
            executed += 1

            again = " (retry)" if step["attempts"] else ""
            report(f"\nStep {step['position']}{again}: {step['description']}")
            result = attempt_step(
                store, client, toolbox, run_id, step["id"], max_tool_calls, max_attempts, verify, report, on_tool,
                always_inspect, memory, late_checks,
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
