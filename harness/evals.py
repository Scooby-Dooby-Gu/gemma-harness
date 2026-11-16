"""Fixed test tasks: the same goals, run in fresh folders and graded by code written here.

The grader is independent of the harness's own verifier, so the results show two
things: how often the model really finishes a task, and how often the harness says
"done" when the work is actually wrong (a false done).
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .client import Ollama
from .runner import run
from .store import Store
from .tools import Toolbox, sandbox_prefix


@dataclass
class Task:
    name: str
    goal: str
    files: dict  # path -> content, written into the fresh workspace
    grade: object  # (workspace: Path) -> None; raises AssertionError with the reason on failure
    web: bool = False  # needs the internet; only run with --web


def sh(ws: Path, *argv: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command in the workspace, sandboxed like the model's own commands."""
    return subprocess.run(
        sandbox_prefix(ws.resolve()) + list(argv), cwd=ws, capture_output=True, text=True,
        timeout=timeout, stdin=subprocess.DEVNULL,
    )


def py(ws: Path, code: str) -> None:
    """Run Python assertions against the workspace's code."""
    proc = sh(ws, sys.executable, "-c", code)
    assert proc.returncode == 0, (proc.stderr or proc.stdout).strip()[-400:]


# --- 1. fix a bug ----------------------------------------------------------------

CALC = """def add(a, b):
    return a + b


def average(numbers):
    return sum(numbers) / (len(numbers) + 1)
"""

TEST_CALC = """import unittest
from calc import add, average


class CalcTest(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_average(self):
        self.assertEqual(average([2, 4, 6]), 4)


if __name__ == "__main__":
    unittest.main()
"""


def grade_fix_bug(ws: Path) -> None:
    assert (ws / "test_calc.py").read_text() == TEST_CALC, "test_calc.py was changed."
    py(ws, "from calc import add, average; assert average([2, 4, 6]) == 4; assert average([5]) == 5; assert add(2, 3) == 5")


# --- 2. write a module with tests ------------------------------------------------

def grade_slugify(ws: Path) -> None:
    py(ws, "from slugify import slugify\n"
           "assert slugify('  Hello,  World!! ') == 'hello-world', slugify('  Hello,  World!! ')\n"
           "assert slugify('A--b__C') == 'a-b-c', slugify('A--b__C')\n"
           "assert slugify('') == ''")
    assert (ws / "test_slugify.py").is_file(), "test_slugify.py is missing."
    proc = sh(ws, sys.executable, "-m", "unittest", "test_slugify")
    assert proc.returncode == 0, "its own tests fail: " + proc.stderr.strip()[-300:]


# --- 3. rename across files ------------------------------------------------------

USERS = """USERS = {1: {"name": "Ada"}, 2: {"name": "Lin"}}


def get_usr(user_id):
    return USERS[user_id]
"""

APP = """from users import get_usr


def greet(user_id):
    return "Hello, " + get_usr(user_id)["name"]
"""

REPORT = """import users


def names(user_ids):
    return [users.get_usr(i)["name"] for i in user_ids]
"""


def grade_rename(ws: Path) -> None:
    for f in ws.rglob("*.py"):
        assert "get_usr" not in f.read_text(), f"{f.name} still uses get_usr."
    py(ws, "from users import get_user; from app import greet; from report import names\n"
           "assert greet(1) == 'Hello, Ada'; assert names([1, 2]) == ['Ada', 'Lin']")


# --- 4. data transform -----------------------------------------------------------

SALES = "category,amount\nbooks,12.50\nfood,4.25\ngames,30\nbooks,7.50\nfood,5.75\nfood,10\n"


def grade_totals(ws: Path) -> None:
    assert (ws / "summarize.py").is_file(), "summarize.py is missing."
    assert (ws / "totals.json").is_file(), "totals.json is missing."
    totals = json.loads((ws / "totals.json").read_text())
    expected = {"books": 20.0, "food": 20.0, "games": 30.0}
    assert set(totals) == set(expected), f"categories are {sorted(totals)}"
    for k, v in expected.items():
        assert abs(float(totals[k]) - v) < 1e-6, f"{k} is {totals[k]}, expected {v}"


# --- 5. small command-line app (the longest chain) -------------------------------

def grade_todo(ws: Path) -> None:
    assert (ws / "todo.py").is_file(), "todo.py is missing."
    (ws / "todo.json").unlink(missing_ok=True)  # start from an empty list
    for args in (["add", "buy milk"], ["add", "call Sam"], ["done", "2"]):
        proc = sh(ws, sys.executable, "todo.py", *args)
        assert proc.returncode == 0, f"`todo.py {' '.join(args)}` failed: {proc.stderr.strip()[-300:]}"
    out = sh(ws, sys.executable, "todo.py", "list").stdout
    assert "1. [ ] buy milk" in out, f"list printed: {out!r}"
    assert "2. [x] call Sam" in out, f"list printed: {out!r}"
    assert (ws / "README.md").is_file(), "README.md is missing."
    readme = (ws / "README.md").read_text().lower()
    for word in ("add", "list", "done"):
        assert word in readme, f"README.md does not mention '{word}'."


# --- 6. gather facts from several files ------------------------------------------

def grade_release(ws: Path) -> None:
    assert (ws / "release.md").is_file(), "release.md is missing."
    text = (ws / "release.md").read_text()
    for label, value in (("port", "8417"), ("database name", "orchid"), ("owner", "Priya"), ("date", "2027-03-09")):
        assert value in text, f"release.md is missing the {label} ({value})."


# --- 7. web research (needs the internet, so only run with --web) -----------------

def grade_python_history(ws: Path) -> None:
    assert (ws / "answer.md").is_file(), "answer.md is missing."
    text = (ws / "answer.md").read_text()
    assert "2008" in text, "answer.md does not give the release year (2008)."
    assert "guido" in text.lower(), "answer.md does not name Guido van Rossum."
    assert "http" in text, "answer.md gives no source URL."


TASKS = [
    Task(
        "fix-bug",
        "The tests in test_calc.py fail. Find and fix the bug in calc.py so that they pass. Do not change test_calc.py.",
        {"calc.py": CALC, "test_calc.py": TEST_CALC},
        grade_fix_bug,
    ),
    Task(
        "slugify",
        "Create slugify.py with a function slugify(text) that lowercases the text, replaces each run of "
        "non-alphanumeric characters with a single hyphen, and strips leading and trailing hyphens. "
        "Add test_slugify.py with unittest tests for it, and make sure the tests pass.",
        {},
        grade_slugify,
    ),
    Task(
        "rename",
        "Rename the function get_usr to get_user everywhere in the workspace. Several Python files use it. "
        "Behaviour must not change.",
        {"users.py": USERS, "app.py": APP, "report.py": REPORT},
        grade_rename,
    ),
    Task(
        "totals",
        "Write summarize.py, which reads sales.csv and writes totals.json: a JSON object mapping each category "
        "to the sum of its amounts. Then run it so that totals.json exists.",
        {"sales.csv": SALES},
        grade_totals,
    ),
    Task(
        "todo-app",
        "Build todo.py, a command-line to-do list that stores its items in todo.json in the current folder. "
        '`python3 todo.py add "text"` adds an item. `python3 todo.py list` prints one line per item in the form '
        "`1. [ ] text`, numbered from 1, with `[x]` instead of `[ ]` for finished items. "
        "`python3 todo.py done 1` marks item 1 as finished. Also write README.md explaining the three commands.",
        {},
        grade_todo,
    ),
    Task(
        "release-notes",
        "Write release.md stating four things that are recorded in the files of this workspace: the server port, "
        "the database name, the release owner, and the release date.",
        {
            "config/server.txt": "# server settings\nhost = 0.0.0.0\nport = 8417\nworkers = 4\n",
            "config/db.txt": "engine: postgres\ndatabase name: orchid\npool size: 10\n",
            "docs/team.md": "# Team\n\n- Design lead: Tomas\n- Release owner: Priya\n- QA: Wen\n",
            "docs/schedule.md": "# Schedule\n\nCode freeze: 2027-03-02\nRelease date: 2027-03-09\n",
            "notes/misc.txt": "Reminder: the old port was 8080 before the migration. Order lunch for Friday.\n",
        },
        grade_release,
    ),
    Task(
        "web-research",
        "Using the web, find out in which year Python 3.0 was released and who created Python. "
        "Write answer.md stating both facts, with the URL of each page you used as a source.",
        {},
        grade_python_history,
        web=True,
    ),
]


def run_evals(
    store: Store,
    root: Path,
    names: list[str],
    repeat: int,
    label: str,
    model: str,
    num_ctx: int,
    think: bool,
    run_options: dict,
    verbose: bool = False,
    web: bool = False,
) -> str:
    """Run the chosen tasks, grade them, and record the results. Returns the batch id."""
    tasks = [t for t in TASKS if (t.name in names if names else True) and (web or not t.web)]
    batch = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config = json.dumps({"num_ctx": num_ctx, "think": think, **run_options})
    print(f"batch {batch} · {label or 'no label'} · {model} · {len(tasks)} task(s) × {repeat}")

    for task in tasks:
        for n in range(1, repeat + 1):
            ws = root / batch / (task.name if repeat == 1 else f"{task.name}-{n}")
            for path, content in task.files.items():
                (ws / path).parent.mkdir(parents=True, exist_ok=True)
                (ws / path).write_text(content)
            ws.mkdir(parents=True, exist_ok=True)

            client = Ollama(model=model, num_ctx=num_ctx, think=think)
            # 'auto' with nobody at the terminal: ordinary commands run sandboxed, risky ones are refused.
            toolbox = Toolbox(ws, commands="auto", web=task.web)
            # A stable memory key per task: every batch uses a new folder, but it is the same "project".
            run_id = store.create_run(task.goal, str(ws), model, project=f"eval:{task.name}")
            started = time.monotonic()
            quiet = (lambda *_: None)
            status = run(
                store, client, toolbox, run_id,
                report=print if verbose else quiet, on_tool=None, **run_options,
            )
            seconds = time.monotonic() - started

            try:
                task.grade(ws)
                passed, detail = True, ""
            except AssertionError as e:
                passed, detail = False, str(e)
            except Exception as e:  # a crash in the graded code is a failure of the task
                passed, detail = False, f"{type(e).__name__}: {e}"

            store.add_eval_result(batch, label, task.name, run_id, status, passed, detail, seconds, config)
            mark = "PASS" if passed else "FAIL"
            note = "" if passed == (status == "done") else "  <-- harness and grader disagree"
            print(f"  {task.name:<14} {mark}  harness said {status:<7} {seconds:5.0f}s  run {run_id}{note}")
            if detail:
                print(f"      {detail[:200]}")
    return batch
