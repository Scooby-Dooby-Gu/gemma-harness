"""Context builder: assemble each step's prompt from the store, within a token budget.

Every step starts from a fresh prompt built here, so nothing depends on the model
remembering earlier steps.
"""

import json
import os
from pathlib import Path

from .checks import describe
from .store import Store
from .tools import SKIP_DIRS

CHARS_PER_TOKEN = 3.5  # rough estimate; good enough for budgeting


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def workspace_listing(workspace: str | Path, max_entries: int = 60, max_depth: int = 2) -> str:
    """A short file tree so the planner is grounded in what actually exists."""
    root = Path(workspace)
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth >= max_depth:
            dirnames[:] = []
        for name in sorted(filenames):
            lines.append(str(rel / name) if depth else name)
            if len(lines) >= max_entries:
                return "\n".join(lines) + "\n[more files not shown]"
    return "\n".join(lines) if lines else "(the workspace is empty)"


def build_step_prompt(
    store: Store, run_id: int, step_id: int, budget_tokens: int = 2000, memory: bool = True
) -> str:
    run = store.get_run(run_id)
    steps = store.list_steps(run_id)
    current = next(s for s in steps if s["id"] == step_id)
    visible = [s for s in steps if s["status"] != "skipped"]

    # Ordered from what stays the same between steps to what changes, so that consecutive
    # prompts begin identically and the model's cache can skip the shared part.
    head = [f"Overall goal: {run['goal']}"]
    progress = []
    if len(visible) > 1:
        head.append("Plan:\n" + "\n".join(f"{n}. {_clip(s['description'], 110)}" for n, s in enumerate(visible, 1)))
        number = {s["id"]: n for n, s in enumerate(visible, 1)}
        done = [str(number[s["id"]]) for s in visible if s["status"] == "done"]
        failed = [str(number[s["id"]]) for s in visible if s["status"] == "failed"]
        state = f"You are on step {number[step_id]} of {len(visible)}."
        if done:
            state += f" Finished: {', '.join(done)}."
        if failed:
            state += f" Failed: {', '.join(failed)}."
        progress.append(state)

    task = [f"Current step: {current['description']}"]
    if current["done_when"]:
        task.append(f"The step is done when: {current['done_when']}")
    checks = json.loads(current["checks"] or "[]")
    if checks:
        task.append(
            "After you finish, the system will automatically check that:\n"
            + "\n".join(f"- {describe(c)}" for c in checks)
        )
    if current["attempts"] > 1 and current["summary"]:
        task.append(f"A previous attempt at this step ended with: {_clip(current['summary'], 300)}")

    finished = [s for s in visible if s["status"] in ("done", "failed") and s["id"] != step_id and s["summary"]]
    recent = [f"- {_clip(s['description'], 80)} -> {s['status']}: {_clip(s['summary'], 240)}" for s in finished[-2:]]

    # De-duplicate facts, keeping the latest wording.
    facts = list(dict.fromkeys(reversed(store.list_facts(run_id))))  # newest first

    remembered = ""
    if memory:
        from .memory import memory_block  # imported here: memory.py is independent of this module

        remembered = memory_block(store, run_id, current["description"], for_planner=False, step_id=step_id)

    def assemble(fact_list: list[str], recent_list: list[str], remembered: str = "") -> str:
        parts = list(head)
        # Facts only ever get added to, oldest first, so this block also starts the same each step.
        if fact_list:
            parts.append("Known facts from earlier steps:\n" + "\n".join(f"- {f}" for f in reversed(fact_list)))
        parts.extend(progress)
        if recent_list:
            parts.append("Results of the most recent steps:\n" + "\n".join(recent_list))
        if remembered:
            parts.append(remembered)
        parts.extend(task)
        parts.append("Do only the current step, then call finish.")
        return "\n\n".join(parts)

    # Over budget: drop remembered notes first (this run's own facts are more reliable),
    # then the oldest facts, then the recent results.
    facts = [_clip(f, 300) for f in facts]
    if estimate_tokens(assemble(facts, recent, remembered)) <= budget_tokens:
        return assemble(facts, recent, remembered)
    while facts and estimate_tokens(assemble(facts, recent)) > budget_tokens:
        facts.pop()
    while recent and estimate_tokens(assemble(facts, recent)) > budget_tokens:
        recent.pop(0)
    return assemble(facts, recent)
