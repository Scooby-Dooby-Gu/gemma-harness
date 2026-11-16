"""Memory across runs, kept in the same SQLite database.

Episodic memory: what happened in earlier runs. The runs, steps and events tables
already record it; this module finds earlier runs that resemble a new goal and
turns each into a short digest for the planner.

Semantic memory: short notes distilled at the end of a run.
  - project notes: lasting facts about one project folder ("tests run with ...")
  - lessons: general lessons that apply anywhere ("a tool error means nothing happened")

Notes carry the run they came from, so a bad one can be traced and forgotten.
Ranking is plain keyword overlap weighted by rarity; at hundreds of rows that is
enough, and it needs no embedding model or search index.
"""

import json
import math
import re

from .client import ModelError, Ollama
from .store import Store

MAX_NOTE_CHARS = 240
MAX_PROJECT_NOTES = 40

STOP = set(
    "the and for with that this from into then than when where what which will must should have has are was "
    "were not all any each one two its it's use using used make makes made file files create add write run "
    "also can does they them their there here over under about your you".split()
)


def realm(project: str) -> str:
    """Test-task runs keep their memory apart from real work."""
    return "eval" if project.startswith("eval:") else "user"


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 2 and t not in STOP}


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def rank(query: str, docs: list[str]) -> list[float]:
    """Score each doc 0..1 by how much of the query's (rarity-weighted) vocabulary it covers."""
    q = tokens(query)
    doc_tokens = [tokens(d) for d in docs]
    n = len(docs)
    df = {t: sum(1 for d in doc_tokens if t in d) for t in q}
    weight = {t: math.log(1 + n / df[t]) if df[t] else math.log(1 + n) for t in q}
    total = sum(weight.values()) or 1.0
    return [sum(weight[t] for t in q & d) / total for d in doc_tokens]


def similar(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


# --- recall ------------------------------------------------------------------------


def recall_notes(store: Store, project: str, query: str, k_project: int = 5, k_lessons: int = 3) -> list:
    """The most relevant active notes: this project's first, then general lessons."""
    rows = store.list_memory(realm(project))
    picked = []
    for scope, k in (("project", k_project), ("global", k_lessons)):
        group = [r for r in rows if r["scope"] == scope and (scope == "global" or r["project"] == project)]
        scores = rank(query, [r["text"] for r in group])
        # Relevance first; among equals, the most recently saved.
        order = sorted(range(len(group)), key=lambda i: (scores[i], group[i]["id"]), reverse=True)
        picked.extend(group[i] for i in order[:k])
    return picked


def recall_episodes(store: Store, run_id: int, k: int = 2) -> list[tuple[int, str]]:
    """Digests of earlier finished runs whose goal resembles this one. Returns (run id, digest)."""
    run = store.get_run(run_id)
    mine = realm(run["project"])
    past = [r for r in store.finished_runs(exclude=run_id) if realm(r["project"]) == mine]
    if not past:
        return []
    scores = rank(run["goal"], [r["goal"] for r in past])
    scored = []
    for r, score in zip(past, scores):
        same_project = r["project"] == run["project"]
        if score >= 0.35 or (same_project and score >= 0.15):
            scored.append((score + (0.1 if same_project else 0), r))
    scored.sort(key=lambda x: (x[0], x[1]["id"]), reverse=True)

    out = []
    for _, r in scored[:k]:
        steps = [s for s in store.list_steps(r["id"]) if s["status"] != "skipped"]
        worked = [_clip(s["description"], 90) for s in steps if s["status"] == "done"]
        problems = [_clip(s["summary"], 140) for s in steps if s["status"] == "failed" and s["summary"]]
        lines = [f'- Earlier task: "{_clip(r["goal"], 160)}" ended {r["status"].upper()}.']
        if worked:
            lines.append("  Steps that worked: " + "; ".join(f"{i}) {w}" for i, w in enumerate(worked[:6], 1)))
        if problems:
            lines.append("  What went wrong: " + " | ".join(problems[:2]))
        out.append((r["id"], "\n".join(lines)))
    return out


def memory_block(store: Store, run_id: int, query: str, for_planner: bool, step_id: int | None = None) -> str:
    """Text to add to a prompt, or '' if nothing relevant is remembered. Logs what was used."""
    run = store.get_run(run_id)
    notes = recall_notes(store, run["project"], query, *((5, 3) if for_planner else (3, 2)))
    episodes = recall_episodes(store, run_id) if for_planner else []
    if not notes and not episodes:
        return ""

    parts = []
    project_notes = [n for n in notes if n["scope"] == "project"]
    lessons = [n for n in notes if n["scope"] == "global"]
    if project_notes:
        parts.append(
            "Notes about this project from earlier sessions (they may be out of date; the files are the truth):\n"
            + "\n".join(f"- {n['text']}" for n in project_notes)
        )
    if lessons:
        parts.append("Lessons from earlier tasks:\n" + "\n".join(f"- {n['text']}" for n in lessons))
    if episodes:
        parts.append("Similar earlier tasks:\n" + "\n".join(d for _, d in episodes))

    store.touch_memory([n["id"] for n in notes])
    store.event(
        run_id, step_id, "memory",
        {"notes": [n["id"] for n in notes], "episodes": [rid for rid, _ in episodes], "for": "plan" if for_planner else "step"},
    )
    return "\n\n".join(parts)


# --- consolidation -----------------------------------------------------------------

SYSTEM = """You are writing notes for your future self after working on a task. You will not remember this task, only the notes.
Save only what will still be true and useful on a DIFFERENT task in the future.

- project_notes: lasting facts about this project folder, such as how its tests are run, where things live, or its conventions. Not a record of what was done today.
- lessons: general lessons about how to work that apply to any project, especially from things that went wrong.

Each note is one short sentence. At most 3 project notes and 2 lessons. Do not repeat notes that already exist.
If nothing is worth keeping, submit empty lists. Reply by calling save_notes."""

SAVE_NOTES = {
    "type": "function",
    "function": {
        "name": "save_notes",
        "description": "Save notes for future tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "project_notes": {"type": "array", "items": {"type": "string"}},
                "lessons": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["project_notes", "lessons"],
        },
    },
}


def _problems(store: Store, run_id: int) -> list[str]:
    found = []
    for e in store.list_events(run_id):
        payload = json.loads(e["payload"])
        if e["kind"] == "verify" and not payload.get("passed"):
            found.append("A step was reported done but failed its check: " + _clip(payload.get("reason", ""), 200))
        elif e["kind"] == "tool" and str(payload.get("result", "")).startswith("Error:"):
            found.append(f"{payload.get('name')} -> {_clip(payload['result'], 160)}")
    return list(dict.fromkeys(found))[:6]


def consolidate(client: Ollama, store: Store, run_id: int) -> list[str]:
    """One model call at the end of a run to distil lasting notes. Returns the notes saved."""
    run = store.get_run(run_id)
    steps = [s for s in store.list_steps(run_id) if s["status"] != "skipped"]
    any_done = any(s["status"] == "done" for s in steps)
    existing = store.list_memory(realm(run["project"]))
    existing = [n for n in existing if n["scope"] == "global" or n["project"] == run["project"]]

    parts = [f"Task: {run['goal']}", f"Outcome: {run['status']}"]
    parts.append("Steps:\n" + "\n".join(f"- [{s['status']}] {_clip(s['description'], 100)} -> {_clip(s['summary'], 160)}" for s in steps))
    facts = list(dict.fromkeys(store.list_facts(run_id)))[-12:]
    if facts:
        parts.append("Facts confirmed during the task:\n" + "\n".join(f"- {_clip(f, 200)}" for f in facts))
    problems = _problems(store, run_id)
    if problems:
        parts.append("Problems along the way:\n" + "\n".join(f"- {p}" for p in problems))
    if existing:
        parts.append("Notes that already exist (do not repeat them):\n" + "\n".join(f"- {n['text']}" for n in existing[-15:]))

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
    try:
        resp = client.chat(messages, tools=[SAVE_NOTES])
    except ModelError as e:
        store.event(run_id, None, "error", {"error": f"memory consolidation failed: {e}"})
        return []
    calls = resp.get("message", {}).get("tool_calls") or []
    args = next((c["function"].get("arguments") for c in calls if c.get("function", {}).get("name") == "save_notes"), None)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None
    if not isinstance(args, dict):
        store.event(run_id, None, "consolidate", {"saved": [], "note": "the model did not call save_notes"})
        return []

    saved = []
    known = [n["text"] for n in existing]

    def keep(scope: str, items, limit: int) -> None:
        if not isinstance(items, list):
            return
        for text in items[:limit]:
            text = _clip(text, MAX_NOTE_CHARS)
            if len(tokens(text)) < 3 or any(similar(text, k) >= 0.7 for k in known):
                continue  # too thin to be useful, or a repeat
            store.add_memory(scope, run["project"] if scope == "project" else "", realm(run["project"]), text, run_id)
            known.append(text)
            saved.append(f"[{scope}] {text}")

    # Project facts are only trusted from a run in which at least one step passed its check.
    if any_done:
        keep("project", args.get("project_notes"), 3)
    keep("global", args.get("lessons"), 2)
    store.trim_memory(run["project"], MAX_PROJECT_NOTES)
    store.event(run_id, None, "consolidate", {"saved": saved})
    return saved
