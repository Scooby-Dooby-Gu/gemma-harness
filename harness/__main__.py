"""Command line: python3 -m harness <command>

  run "<goal>"           plan a goal into steps and work through them
  resume <run_id>        continue an interrupted, failed or blocked run
  step "<instruction>"   run a single step as its own run, with no planning
  eval                   run the fixed test tasks and grade them independently
  evals [batch]          compare test-task batches, or show one batch in detail
  memory [add|forget]    list, add or retire remembered notes
  runs                   list past runs
  show <run_id>          show a run's steps, facts and event log
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .checks import describe
from .client import Ollama
from .evals import TASKS, run_evals
from .runner import attempt_step, run as run_goal
from .store import Store
from .tools import Toolbox

ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines into the environment, without overriding what is already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip().removeprefix("export ").strip(), value.strip().strip("'\""))


def short(value, width: int = 100) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def print_tool(name, call_args, result):
    print(f"  → {name}({short(call_args, 70)})\n    {short(result)}")


def limits(args) -> dict:
    return {
        "max_tool_calls": args.max_tool_calls,
        "max_minutes": args.max_minutes,
        "verify": not args.no_verify,
        "always_inspect": args.always_inspect,
        "memory": not args.no_memory,
    }


def cmd_run(args, store: Store) -> int:
    workspace = Path(args.workspace).resolve()
    client = Ollama(model=args.model, num_ctx=args.num_ctx, think=not args.no_think)
    toolbox = Toolbox(workspace, commands=args.commands, web=args.web)
    run_id = store.create_run(args.goal, str(workspace), args.model)
    print(f"run {run_id} · {args.model} · workspace {workspace}")
    status = run_goal(store, client, toolbox, run_id, **limits(args), on_tool=print_tool)
    return 0 if status == "done" else 1


def cmd_resume(args, store: Store) -> int:
    run = store.get_run(args.run_id)
    if run is None:
        print(f"No run {args.run_id}.", file=sys.stderr)
        return 1
    if args.retry_failed:
        store.retry_failed_steps(run["id"])
    client = Ollama(model=run["model"], num_ctx=args.num_ctx, think=not args.no_think)
    toolbox = Toolbox(Path(run["workspace"]), commands=args.commands, web=args.web)
    print(f"resuming run {run['id']} · {run['model']} · workspace {run['workspace']}")
    status = run_goal(store, client, toolbox, run["id"], **limits(args), on_tool=print_tool)
    return 0 if status == "done" else 1


def cmd_step(args, store: Store) -> int:
    workspace = Path(args.workspace).resolve()
    client = Ollama(model=args.model, num_ctx=args.num_ctx, think=not args.no_think)
    toolbox = Toolbox(workspace, commands=args.commands, web=args.web)

    run_id = store.create_run(args.instruction, str(workspace), args.model)
    step_id = store.add_step(run_id, args.instruction, args.done_when)
    print(f"run {run_id} · {args.model} · workspace {workspace}")

    result = attempt_step(
        store, client, toolbox, run_id, step_id, args.max_tool_calls,
        max_attempts=1, verify=not args.no_verify, on_tool=print_tool, always_inspect=args.always_inspect,
        memory=not args.no_memory,
    )
    store.set_run_status(run_id, result.status)
    for fact in result.facts:
        print(f"  fact: {fact}")
    return 0 if result.status == "done" else 1


def print_batch(store: Store, batch: str) -> None:
    rows = store.eval_rows(batch)
    print(f"\n{'task':<14} {'result':<6} {'harness':<8} {'secs':>5} {'calls':>5} {'steps':>5} {'replans':>7} {'rejected':>8}  run")
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{r['task']:<14} {mark:<6} {r['harness_status']:<8} {r['seconds']:>5.0f} {r['model_calls']:>5} "
              f"{r['steps']:>5} {r['replans']:>7} {r['rejections']:>8}  {r['run_id']}")
        if r["detail"]:
            print(f"    {short(r['detail'], 150)}")
    passed = sum(r["passed"] for r in rows)
    false_done = sum(1 for r in rows if r["harness_status"] == "done" and not r["passed"])
    print(f"\n{passed}/{len(rows)} passed · {false_done} false done · "
          f"{sum(r['seconds'] for r in rows):.0f}s · {sum(r['model_calls'] for r in rows)} model calls")


def cmd_eval(args, store: Store) -> int:
    unknown = [n for n in args.task if n not in [t.name for t in TASKS]]
    if unknown:
        print(f"Unknown task(s): {', '.join(unknown)}. Available: {', '.join(t.name for t in TASKS)}", file=sys.stderr)
        return 1
    options = limits(args)
    options["memory"] = args.memory  # off by default here, so batches are comparable
    batch = run_evals(
        store, ROOT / "eval-workspaces", args.task, args.repeat, args.label, args.model, args.num_ctx,
        not args.no_think, options, args.verbose, web=args.web,
    )
    print_batch(store, batch)
    return 0


def cmd_evals(args, store: Store) -> int:
    if args.batch:
        print_batch(store, args.batch)
        return 0
    print(f"{'batch':<18} {'label':<22} {'passed':>7} {'false done':>10} {'secs':>6} {'calls':>6}")
    for b in store.eval_batches():
        print(f"{b['batch']:<18} {short(b['label'], 22):<22} {str(b['passed']) + '/' + str(b['n']):>7} "
              f"{b['false_done']:>10} {b['seconds']:>6.0f} {b['model_calls']:>6}")
    return 0


def cmd_memory(args, store: Store) -> int:
    if args.action == "add":
        if not args.value:
            print('Give the note text: harness memory add "…"', file=sys.stderr)
            return 1
        project = "" if args.everywhere else str(Path(args.workspace).resolve())
        note_id = store.add_memory("global" if args.everywhere else "project", project, "user", args.value, None)
        print(f"saved note {note_id}")
        return 0
    if args.action == "forget":
        if not (args.value or "").isdigit() or not store.forget_memory(int(args.value)):
            print("Give the id of an existing note: harness memory forget 12", file=sys.stderr)
            return 1
        print(f"note {args.value} retired; it will no longer be used")
        return 0
    rows = store.list_memory(None if args.all else "user", include_inactive=args.all)
    if not rows:
        print("Nothing remembered yet.")
    for r in rows:
        where = "everywhere" if r["scope"] == "global" else r["project"]
        flags = ("" if r["active"] else " (retired)") + (" (test tasks)" if r["realm"] == "eval" else "")
        print(f"{r['id']:>4}  used {r['uses']:>2}×  from run {r['source_run_id'] or '-':<4} {where}{flags}\n      {r['text']}")
    return 0


def cmd_runs(args, store: Store) -> int:
    for r in store.list_runs(args.limit):
        print(f"{r['id']:>4}  {r['created_at'][:16]}  {r['status']:<8} {r['n_done']}/{r['n_steps']}  {short(r['goal'], 70)}")
    return 0


def cmd_show(args, store: Store) -> int:
    run = store.get_run(args.run_id)
    if run is None:
        print(f"No run {args.run_id}.", file=sys.stderr)
        return 1
    print(f"run {run['id']} · {run['status']} · {run['model']} · {run['created_at']}")
    print(f"goal: {run['goal']}\nworkspace: {run['workspace']}\n")
    for s in store.list_steps(run["id"]):
        print(f"step {s['position']} [{s['status']}] {s['description']}")
        for c in json.loads(s["checks"] or "[]"):
            print(f"    check: {describe(c)}")
        if s["summary"]:
            print(f"    {s['summary']}")
    facts = store.list_facts(run["id"])
    if facts:
        print("\nfacts:")
        for f in facts:
            print(f"  - {f}")
    print("\nevents:")
    for e in store.list_events(run["id"]):
        payload = json.loads(e["payload"])
        if e["kind"] == "model" and not args.full:
            payload.pop("thinking", None)
        text = json.dumps(payload, ensure_ascii=False, indent=2) if args.full else short(payload, 140)
        print(f"  {e['ts'][11:19]} {e['kind']:<8} {text}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="harness")
    parser.add_argument("--db", default=str(ROOT / "harness.db"), help="SQLite database path")
    sub = parser.add_subparsers(dest="command", required=True)

    def model_options(p, choose_target: bool = True):
        if choose_target:
            p.add_argument("--workspace", default=str(ROOT / "workspace"))
            p.add_argument("--model", default="gemma4:12b-mlx")
        p.add_argument("--num-ctx", type=int, default=16384)
        p.add_argument("--max-tool-calls", type=int, default=8, help="tool calls allowed per step")
        p.add_argument("--no-think", action="store_true", help="turn off the model's thinking")
        p.add_argument("--commands", choices=["off", "ask", "auto"], default="off",
                       help="shell commands: off, ask before each, or auto (ask only for risky ones)")
        p.add_argument("--web", action="store_true", help="give the model web_search and fetch_url")
        p.add_argument("--no-memory", action="store_true", help="neither use nor save notes from other runs")
        p.add_argument("--no-verify", action="store_true", help="skip the independent check of each step")
        p.add_argument("--always-inspect", action="store_true",
                       help="also have the model inspect steps whose automatic checks passed")
        p.add_argument("--max-minutes", type=float, default=30, help="time limit for this session of the run")

    p = sub.add_parser("run", help="plan a goal and work through the steps")
    p.add_argument("goal")
    model_options(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="continue an existing run")
    p.add_argument("run_id", type=int)
    p.add_argument("--retry-failed", action="store_true", help="put failed and blocked steps back in the queue")
    model_options(p, choose_target=False)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("step", help="run a single step with no planning")
    p.add_argument("instruction")
    p.add_argument("--done-when", default="", help="how to tell the step is complete")
    model_options(p)
    p.set_defaults(func=cmd_step)

    p = sub.add_parser("eval", help="run the fixed test tasks")
    p.add_argument("--task", action="append", default=[], help="run only this task (repeatable)")
    p.add_argument("--repeat", type=int, default=1, help="times to run each task")
    p.add_argument("--label", default="", help="a name for this configuration, to compare batches")
    p.add_argument("--verbose", action="store_true", help="print each run's full progress")
    p.add_argument("--memory", action="store_true",
                   help="let test tasks use and save memory (kept apart from your real work's memory)")
    model_options(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("evals", help="compare test-task batches")
    p.add_argument("batch", nargs="?", help="show this batch in detail")
    p.set_defaults(func=cmd_evals)

    p = sub.add_parser("memory", help="list, add or retire remembered notes")
    p.add_argument("action", nargs="?", choices=["list", "add", "forget"], default="list")
    p.add_argument("value", nargs="?", help="note text for add, note id for forget")
    p.add_argument("--everywhere", action="store_true", help="with add: a general lesson, not tied to one project")
    p.add_argument("--workspace", default=str(ROOT / "workspace"), help="with add: the project the note is about")
    p.add_argument("--all", action="store_true", help="also show retired notes and test-task notes")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("runs", help="list past runs")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("show", help="show one run in detail")
    p.add_argument("run_id", type=int)
    p.add_argument("--full", action="store_true", help="print full event payloads")
    p.set_defaults(func=cmd_show)

    args = parser.parse_args()
    load_env_file(ROOT / ".env")
    return args.func(args, Store(Path(args.db)))


if __name__ == "__main__":
    sys.exit(main())
