"""Runner logic tested with a scripted fake model, so no Ollama is needed.

Run with: python3 -m unittest discover tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.checks import run_checks  # noqa: E402
from harness.memory import recall_notes  # noqa: E402
from harness.runner import run  # noqa: E402
from harness.store import Store  # noqa: E402
from harness.tools import Toolbox  # noqa: E402


def call(name, **args):
    return {"message": {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}}


def plan(*descriptions):
    return call("submit_plan", steps=[{"description": d, "done_when": f"{d} is done"} for d in descriptions])


def plan_with(description, checks):
    return call("submit_plan", steps=[{"description": description, "done_when": "it is done", "checks": checks}])


def notes(project=(), lessons=()):
    return call("save_notes", project_notes=list(project), lessons=list(lessons))


def verdict(passed, reason):
    return call("verdict", passed=passed, reason=reason)


def finish(status, summary="ok", facts=()):
    return call("finish", status=status, summary=summary, facts=list(facts))


class FakeModel:
    num_ctx = 16384

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.tools = []

    def chat(self, messages, tools=None):
        self.tools.append(tools)
        self.prompts.append(list(messages))  # copy: the caller keeps appending
        return self.replies.pop(0)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name).resolve()  # the same path form the command line uses
        self.store = Store(root / "test.db")
        self.toolbox = Toolbox(root / "ws")
        self.run_id = self.store.create_run("make two files", str(root / "ws"), "fake")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def go(self, model, **kw):
        kw.setdefault("verify", False)
        kw.setdefault("memory", False)
        return run(self.store, model, self.toolbox, self.run_id, report=lambda *_: None, **kw)

    def statuses(self):
        return [s["status"] for s in self.store.list_steps(self.run_id)]

    def test_plans_and_completes(self):
        model = FakeModel([
            plan("write a.txt", "write b.txt"),
            call("write_file", path="a.txt", content="A"),
            finish("done", facts=["a.txt holds A"]),
            call("write_file", path="b.txt", content="B"),
            finish("done"),
        ])
        self.assertEqual(self.go(model), "done")
        self.assertEqual(self.statuses(), ["done", "done"])
        self.assertEqual((self.toolbox.workspace / "b.txt").read_text(), "B")
        # The second step's prompt carries the fact from the first, in a fresh conversation.
        second_step_prompt = model.prompts[3]
        self.assertEqual(len(second_step_prompt), 2)
        self.assertIn("a.txt holds A", second_step_prompt[1]["content"])

    def test_failure_triggers_replan(self):
        model = FakeModel([
            plan("write a.txt", "write b.txt"),
            finish("failed", "could not write a.txt"),
            plan("write a.txt another way", "write b.txt"),
            finish("done"),
            finish("done"),
        ])
        self.assertEqual(self.go(model), "done")
        self.assertEqual(self.statuses(), ["failed", "skipped", "done", "done"])
        self.assertIn("could not write a.txt", model.prompts[2][1]["content"])

    def test_replan_limit_stops_the_run(self):
        model = FakeModel([plan("a"), finish("failed"), plan("b"), finish("failed")])
        self.assertEqual(self.go(model, max_replans=1), "failed")
        self.assertEqual(self.store.get_run(self.run_id)["status"], "failed")

    def test_blocked_stops_and_resume_continues(self):
        model = FakeModel([plan("a", "b"), finish("blocked", "need the API name")])
        self.assertEqual(self.go(model), "blocked")
        self.assertEqual(self.statuses(), ["blocked", "pending"])
        self.store.retry_failed_steps(self.run_id)
        self.assertEqual(self.go(FakeModel([finish("done"), finish("done")])), "done")
        self.assertEqual(self.statuses(), ["done", "done"])

    def test_interrupted_step_is_rerun(self):
        step_id = self.store.add_step(self.run_id, "a")
        self.store.start_step(step_id)  # simulate a crash mid-step
        self.assertEqual(self.go(FakeModel([finish("done")])), "done")

    # --- verification -------------------------------------------------------

    def test_done_without_any_successful_tool_call_is_rejected_then_retried(self):
        model = FakeModel([
            plan("write a.txt"),
            finish("done", "wrote it", facts=["a.txt exists"]),  # a lie: nothing was written
            call("write_file", path="a.txt", content="A"),       # retry does the work
            finish("done", facts=["a.txt holds A"]),
            call("read_file", path="a.txt"),                     # verifier looks
            verdict(True, "a.txt exists and holds A"),
        ])
        self.assertEqual(self.go(model, verify=True), "done")
        self.assertEqual(self.statuses(), ["done"])
        # The lie's fact was never saved; only the verified attempt's fact was.
        self.assertEqual(self.store.list_facts(self.run_id), ["a.txt holds A"])
        # The retry was told what the check found.
        self.assertIn("no tool call succeeded", model.prompts[2][1]["content"])

    def test_verifier_sees_recorded_changes_but_not_the_claim(self):
        model = FakeModel([
            plan("write a.txt"),
            call("write_file", path="a.txt", content="A"),
            finish("done", "SECRET-CLAIM"),
            verdict(True, "fine"),
        ])
        self.assertEqual(self.go(model, verify=True), "done")
        verifier_prompt = model.prompts[3][1]["content"]
        self.assertIn("Files created: a.txt", verifier_prompt)
        self.assertNotIn("SECRET-CLAIM", verifier_prompt)
        tools_offered = [t["function"]["name"] for t in model.tools[3]]
        self.assertNotIn("write_file", tools_offered)

    def test_rejected_twice_fails_the_step_and_replans(self):
        model = FakeModel([
            plan("write a.txt"),
            call("list_dir", path="."), finish("done"), verdict(False, "a.txt is missing"),
            call("list_dir", path="."), finish("done"), verdict(False, "a.txt is still missing"),
            plan("write a.txt properly"),
            call("write_file", path="a.txt", content="A"), finish("done"), verdict(True, "ok"),
        ])
        self.assertEqual(self.go(model, verify=True), "done")
        self.assertEqual(self.statuses(), ["failed", "done"])
        self.assertIn("still missing", model.prompts[7][1]["content"])  # the re-plan saw the finding

    def test_time_limit_stops_the_run(self):
        model = FakeModel([plan("a", "b")])
        self.assertEqual(self.go(model, max_minutes=0), "failed")
        self.assertEqual(self.statuses(), ["pending", "pending"])

    # --- code checks --------------------------------------------------------

    def test_passing_code_checks_decide_without_a_model_inspection(self):
        checks = [{"type": "file_exists", "path": "a.txt"}, {"type": "file_contains", "path": "a.txt", "text": "A"}]
        model = FakeModel([
            plan_with("write a.txt", checks),
            call("write_file", path="a.txt", content="A"),
            finish("done"),
        ])  # no verdict reply scripted: the inspector must not be called
        self.assertEqual(self.go(model, verify=True), "done")
        self.assertEqual(len(model.prompts), 3)
        # The executor was told what would be checked.
        self.assertIn("the file a.txt contains the text: A", model.prompts[1][1]["content"])

    def test_failing_code_check_rejects_and_the_retry_sees_why(self):
        checks = [{"type": "file_contains", "path": "a.txt", "text": "def main"}]
        model = FakeModel([
            plan_with("write a.txt", checks),
            call("write_file", path="a.txt", content="nothing useful"), finish("done"),
            verdict(False, "a.txt has no main function"),  # a failed text check gets a second opinion
            call("write_file", path="a.txt", content="def  main():\n    pass"), finish("done"),
        ])
        self.assertEqual(self.go(model, verify=True), "done")
        self.assertIn("does not contain the expected text: def main", model.prompts[4][1]["content"])

    def test_wrong_text_check_is_overruled_by_inspection(self):
        # The plan guessed text that belongs in another file. The work is right, so inspection passes it.
        model = FakeModel([
            plan_with("rename in app.py", [{"type": "file_contains", "path": "app.py", "text": "def get_user"}]),
            call("write_file", path="app.py", content="from users import get_user\n"),
            finish("done"),
            call("read_file", path="app.py"),
            verdict(True, "app.py uses get_user"),
        ])
        self.assertEqual(self.go(model, verify=True), "done")
        self.assertEqual(self.statuses(), ["done"])
        self.assertIn("may be a wrong guess", model.prompts[3][1]["content"])
        self.assertEqual(self.store.count_events(self.run_id, "warning"), 1)

    def test_other_failed_checks_are_not_open_to_dispute(self):
        model = FakeModel([
            plan_with("write a.txt", [{"type": "file_exists", "path": "a.txt"}]),
            call("write_file", path="b.txt", content="B"),
            finish("done"),
            call("write_file", path="a.txt", content="A"),
            finish("done"),
        ])
        self.assertEqual(self.go(model, verify=True), "done")
        self.assertEqual(len(model.prompts), 5)  # no inspection call was made

    def test_half_done_rename_is_caught_and_rereading_after_an_edit_is_allowed(self):
        ws = self.toolbox.workspace
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "app.py").write_text("from users import get_usr\nprint(get_usr(1))\n")
        read = call("read_file", path="app.py")
        model = FakeModel([
            plan_with("rename in app.py", [{"type": "file_lacks", "path": "app.py", "text": "get_usr"}]),
            read, read,
            call("edit_file", path="app.py", old_text="import get_usr", new_text="import get_user"),
            read,  # a third identical read, allowed because the file changed
            finish("done"),
            call("edit_file", path="app.py", old_text="print(get_usr(1))", new_text="print(get_user(1))"),
            finish("done"),
        ])
        self.assertEqual(self.go(model, verify=True), "done")
        self.assertIn("still contains 'get_usr' on line(s) 2", model.prompts[6][1]["content"])
        self.assertIn("get_user", model.prompts[5][-1]["content"])  # the re-read returned the file

    def test_malformed_checks_are_dropped_and_the_inspector_takes_over(self):
        checks = [{"type": "file_is_nice", "path": "a.txt"}, {"type": "file_contains", "path": "a.txt"},
                  {"type": "command_succeeds", "command": "true"}, "nonsense"]  # commands are off here
        model = FakeModel([
            plan_with("write a.txt", checks),
            call("write_file", path="a.txt", content="A"), finish("done"),
            verdict(True, "looked at it"),
        ])
        self.assertEqual(self.go(model, verify=True), "done")
        self.assertEqual(self.store.list_steps(self.run_id)[0]["checks"], "[]")

    def test_command_check_runs_and_reports_failure_output(self):
        box = Toolbox(self.toolbox.workspace, commands="auto")
        ok, bad, unapproved, outside = run_checks(box, [
            {"type": "command_succeeds", "command": "true"},
            {"type": "command_succeeds", "command": "echo boom && exit 3"},
            {"type": "command_succeeds", "command": "rm -rf ."},
            {"type": "file_exists", "path": "../../etc/hosts"},
        ])
        self.assertEqual(ok.status, "passed")
        self.assertEqual(bad.status, "failed")
        self.assertIn("code 3", bad.detail)
        self.assertIn("boom", bad.detail)
        self.assertEqual(unapproved.status, "skipped")  # could not run, so it proves nothing
        self.assertEqual(outside.status, "skipped")

    def test_old_database_gains_the_checks_column(self):
        path = Path(self.tmp.name) / "old.db"
        import sqlite3
        db = sqlite3.connect(path)
        db.executescript("CREATE TABLE steps (id INTEGER PRIMARY KEY, run_id INTEGER, position INTEGER, "
                         "description TEXT, done_when TEXT, status TEXT, summary TEXT, attempts INTEGER, "
                         "started_at TEXT, finished_at TEXT); INSERT INTO steps (run_id, position, description) VALUES (1,1,'x');")
        db.commit(); db.close()
        store = Store(path)
        self.assertEqual(store.get_step(1)["checks"], "[]")
        store.db.close()

    def test_replan_limit_is_per_session(self):
        model = FakeModel([plan("a"), finish("failed"), plan("b"), finish("failed")])
        self.assertEqual(self.go(model, max_replans=1), "failed")
        # Resuming gets a fresh allowance, so the run can re-plan again and finish.
        self.store.retry_failed_steps(self.run_id)
        model = FakeModel([finish("failed"), plan("c"), finish("done")])
        self.assertEqual(self.go(model, max_replans=1), "done")

    # --- memory -------------------------------------------------------------

    def new_run(self, goal, project=None):
        return self.store.create_run(goal, str(self.toolbox.workspace), "fake", project)

    def test_notes_are_saved_then_recalled_by_the_next_run(self):
        self.run_id = self.new_run("write tests for calc.py")
        first = FakeModel([
            plan("write the tests for calc.py"),
            call("write_file", path="test_calc.py", content="x"), finish("done"),
            notes(project=["Tests for calc.py are run with python3 -m unittest test_calc.py."],
                  lessons=["A tool error means the action did not happen, so retry or report failure."]),
        ])
        self.assertEqual(self.go(first, memory=True), "done")
        saved = self.store.list_memory("user")
        self.assertEqual([m["scope"] for m in saved], ["project", "global"])
        self.assertEqual(saved[0]["source_run_id"], self.run_id)

        self.run_id = self.new_run("add more tests for calc.py")
        second = FakeModel([plan("extend the tests for calc.py"), call("list_dir", path="."), finish("done"), notes()])
        self.assertEqual(self.go(second, memory=True), "done")
        plan_prompt, step_prompt = second.prompts[0][1]["content"], second.prompts[1][1]["content"]
        self.assertIn("python3 -m unittest test_calc.py", plan_prompt)       # project note
        self.assertIn("A tool error means", plan_prompt)                     # lesson
        self.assertIn('Earlier task: "write tests for calc.py" ended DONE', plan_prompt)  # similar earlier run
        self.assertNotIn("make two files", plan_prompt)                      # unrelated run is not recalled
        self.assertIn("python3 -m unittest test_calc.py", step_prompt)
        self.assertGreaterEqual(self.store.list_memory("user")[0]["uses"], 2)
        self.assertEqual(self.store.count_events(self.run_id, "memory"), 2)  # what was injected is logged

    def test_memory_guards(self):
        # A run with no successful step may save a lesson, but not "facts" about the project.
        model = FakeModel([
            plan("a"), finish("failed", "could not"), plan("b"), finish("failed", "still could not"),
            notes(project=["The project keeps its settings in config.yaml files."],
                  lessons=["Reading a file first avoids guessing what the file contains."]),
        ])
        self.assertEqual(self.go(model, memory=True, max_replans=1), "failed")
        self.assertEqual([m["scope"] for m in self.store.list_memory("user")], ["global"])

        # Repeats and near-repeats are not saved twice.
        self.run_id = self.new_run("another task")
        model = FakeModel([
            plan("a"), call("list_dir", path="."), finish("done"),
            notes(lessons=["Reading a file first avoids guessing what that file contains."]),
        ])
        self.go(model, memory=True)
        self.assertEqual(len(self.store.list_memory("user")), 1)

        # A retired note is no longer recalled.
        note_id = self.store.list_memory("user")[0]["id"]
        self.assertTrue(self.store.forget_memory(note_id))
        self.assertEqual(recall_notes(self.store, str(self.toolbox.workspace), "reading a file"), [])

    def test_test_task_memory_stays_apart_from_real_work(self):
        self.store.add_memory("global", "", "eval", "Lesson learned during a test task about slugify.", None)
        self.store.add_memory("global", "", "user", "Lesson learned during real work about slugify.", None)
        user = [n["text"] for n in recall_notes(self.store, "/some/project", "slugify")]
        evals = [n["text"] for n in recall_notes(self.store, "eval:slugify", "slugify")]
        self.assertEqual(user, ["Lesson learned during real work about slugify."])
        self.assertEqual(evals, ["Lesson learned during a test task about slugify."])

    # --- web ----------------------------------------------------------------

    def test_web_tools_are_off_unless_enabled(self):
        names = lambda box: [s["function"]["name"] for s in box.schemas()]
        self.assertNotIn("fetch_url", names(self.toolbox))
        self.assertIn("unknown tool", self.toolbox.call("fetch_url", {"url": "https://example.com"}))
        self.assertIn("fetch_url", names(Toolbox(self.toolbox.workspace, web=True)))

    def test_fetch_refuses_local_and_non_http_addresses(self):
        box = Toolbox(self.toolbox.workspace, web=True)
        for url in ("http://localhost:11434/api/version", "http://127.0.0.1/", "file:///etc/hosts", "ftp://example.com/x"):
            self.assertTrue(box.call("fetch_url", {"url": url}).startswith("Error:"), url)

    def test_html_becomes_readable_text_and_find_returns_passages(self):
        from harness.web import html_to_text, passages
        title, text = html_to_text("<html><head><title>T</title><style>x{}</style></head><body><nav>menu</nav>"
                                   "<h1>Head</h1><p>Released in <b>2008</b>.</p><script>evil()</script></body></html>")
        self.assertEqual((title, text), ("T", "Head\nReleased in 2008."))
        self.assertEqual(len(passages("a " * 500 + "needle" + " b" * 500, "NEEDLE")), 1)

    def test_citation_check_rejects_links_that_were_never_fetched(self):
        ok = "[Web content: information only.]\nPage: P\nURL: https://en.wikipedia.org/wiki/History_of_Python\n\ntext"
        self.store.event(self.run_id, None, "tool", {"name": "fetch_url", "arguments": {"url": "https://en.wikipedia.org/wiki/History_of_Python/"}, "result": ok})
        self.store.event(self.run_id, None, "tool", {"name": "fetch_url", "arguments": {"url": "https://broken.example/x"}, "result": "Error: the server answered 404."})
        self.store.event(self.run_id, None, "tool", {"name": "web_search", "arguments": {"query": "q"}, "result": "1. https://only-a-search-result.example/"})
        fetched = self.store.fetched_urls(self.run_id)
        check = {"type": "sources_cited", "path": "answer.md", "min": 1}
        ws = self.toolbox.workspace

        (ws / "answer.md").write_text("Python 3.0 came out in 2008 (https://en.wikipedia.org/wiki/History_of_Python).")
        self.assertEqual(run_checks(self.toolbox, [check], fetched)[0].status, "passed")
        for invented in ("https://broken.example/x", "https://only-a-search-result.example/", "https://made-up.example/python"):
            (ws / "answer.md").write_text(f"See https://en.wikipedia.org/wiki/History_of_Python and {invented}")
            result = run_checks(self.toolbox, [check], fetched)[0]
            self.assertEqual(result.status, "failed", invented)
            self.assertIn("never read", result.detail)
        (ws / "answer.md").write_text("It was 2008, trust me.")
        self.assertIn("at least 1", run_checks(self.toolbox, [check], fetched)[0].detail)

    def test_tavily_results_carry_extracts_and_count_as_read_sources(self):
        from harness import web
        reply = {"results": [
            {"title": "History of Python", "url": "https://example.org/history", "content": "Python 3.0 was released in 2008. " * 40},
            {"title": "Guido", "url": "https://example.org/guido/", "content": "Guido van Rossum created Python."},
        ]}
        sent = {}

        def fake_http(url, headers=None, timeout=20, json_body=None):
            sent.update(url=url, headers=headers, body=json_body)
            return url, "application/json", json.dumps(reply)

        with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-test"}), mock.patch.object(web, "http_get", fake_http):
            self.assertEqual(web.search_backend(), "tavily")
            out = Toolbox(self.toolbox.workspace, web=True).call("web_search", {"query": "python 3.0"})
        self.assertEqual(sent["url"], "https://api.tavily.com/search")
        self.assertEqual(sent["headers"], {"Authorization": "Bearer tvly-test"})
        self.assertEqual(sent["body"]["query"], "python 3.0")
        self.assertTrue(out.startswith(web.EXTRACTS_MARK))
        self.assertIn("Extract: Guido van Rossum created Python.", out)
        self.assertNotIn("tvly-test", out)

        # Pages that came back with an extract count as read, so they may be cited without a fetch.
        self.store.event(self.run_id, None, "tool", {"name": "web_search", "arguments": {"query": "q"}, "result": out})
        self.assertEqual(self.store.fetched_urls(self.run_id), {"https://example.org/history", "https://example.org/guido"})

    def test_tavily_errors_are_explained(self):
        from harness import web

        def refuse(*a, **k):
            raise web.WebError("the server answered 401 Unauthorized.")

        with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-bad"}), mock.patch.object(web, "http_get", refuse):
            out = Toolbox(self.toolbox.workspace, web=True).call("web_search", {"query": "x"})
        self.assertEqual(out, "Error: the Tavily API key is missing or not valid.")

    def test_planner_only_offers_checks_for_enabled_tools(self):
        from harness.checks import allowed_types, clean_checks
        raw = [{"type": "sources_cited", "path": "a.md", "min": 2}, {"type": "command_succeeds", "command": "true"}]
        self.assertEqual(clean_checks(raw, allowed_types(["read_file"]))[0], [])
        kept, _ = clean_checks(raw, allowed_types(["read_file", "fetch_url"]))
        self.assertEqual(kept, [{"type": "sources_cited", "path": "a.md", "min": 2}])

    # --- guardrails ---------------------------------------------------------

    def test_command_policy(self):
        box = Toolbox(self.toolbox.workspace, commands="auto")
        self.assertIn("not allowed", box.call("run_command", {"command": "sudo ls"}))
        # Risky commands need approval; with nobody at the terminal they are refused.
        self.assertIn("approval", box.call("run_command", {"command": "rm -rf ."}))
        with mock.patch.object(Toolbox, "_approve", return_value=True):
            self.assertIn("exit code 0", box.call("run_command", {"command": "touch x && rm x"}))
        self.assertIn("exit code 0", box.call("run_command", {"command": "echo hi > made.txt && cat made.txt"}))
        self.assertIn("unknown tool", self.toolbox.call("run_command", {"command": "ls"}))  # off by default

    @unittest.skipUnless(sys.platform == "darwin", "sandbox-exec is macOS only")
    def test_commands_cannot_write_outside_the_workspace(self):
        box = Toolbox(self.toolbox.workspace, commands="auto")
        outside = Path.home() / "harness-sandbox-escape-test.txt"
        with mock.patch.object(Toolbox, "_approve", return_value=True):  # even an approved command is confined
            out = box.call("run_command", {"command": f"echo x > {outside}"})
        self.assertFalse(outside.exists())
        self.assertNotIn("exit code 0", out)

    def test_edit_file(self):
        box, f = self.toolbox, self.toolbox.workspace / "app.py"
        original = "from users import get_usr\n\n\ndef greet(user_id):\n    return \"Hello, \" + get_usr(user_id)[\"name\"]\n"
        f.write_text(original)
        edit = lambda **a: box.call("edit_file", {"path": "app.py", **a})

        # More than one match needs an explicit choice, and nothing changes meanwhile.
        out = edit(old_text="get_usr", new_text="get_user")
        self.assertIn("appears 2 times", out)
        self.assertIn("lines 1, 5", out)
        self.assertEqual(f.read_text(), original)

        out = edit(old_text="get_usr", new_text="get_user", replace_all=True)
        self.assertIn("Replaced 2 occurrences", out)
        self.assertEqual(f.read_text(), original.replace("get_usr", "get_user"))

        # A single exact replacement, with the result shown back.
        out = edit(old_text='"Hello, "', new_text='"Hi, "')
        self.assertIn('5:     return "Hi, " + get_user(user_id)["name"]', out)

        # Line numbers copied from read_file are tolerated.
        out = edit(old_text="4: def greet(user_id):", new_text="4: def greet(uid):")
        self.assertIn("Replaced 1 occurrence", out)
        self.assertIn("def greet(uid):", f.read_text())
        self.assertNotIn("4:", f.read_text())

        # A near miss explains itself; a miss changes nothing.
        before = f.read_text()
        out = edit(old_text='return "Hi, " + get_user(user_id)["name"]  ', new_text="x")
        self.assertIn("closest lines", out)
        self.assertIn("5: ", out)
        self.assertIn("was not found", edit(old_text="no such text anywhere", new_text="x"))
        self.assertIn("does not exist", box.call("edit_file", {"path": "nope.py", "old_text": "a", "new_text": "b"}))
        self.assertIn("outside the workspace", box.call("edit_file", {"path": "../x.py", "old_text": "a", "new_text": "b"}))
        self.assertEqual(f.read_text(), before)

        # The verifier's read-only toolbox cannot edit.
        inspector = Toolbox(box.workspace, read_only=True)
        self.assertIn("unknown tool", inspector.call("edit_file", {"path": "app.py", "old_text": "greet", "new_text": "x"}))

    def test_search_accepts_a_single_file_as_the_path(self):
        (self.toolbox.workspace / "app.py").write_text("from users import get_usr\n")
        self.assertIn("app.py:1:", self.toolbox.call("search", {"pattern": "get_usr", "path": "app.py"}))
        self.assertIn("app.py:1:", self.toolbox.call("search", {"pattern": "get_usr"}))

    def test_tools_cannot_leave_the_workspace(self):
        self.assertIn("outside the workspace", self.toolbox.call("read_file", {"path": "../test.db"}))
        self.assertIn("outside the workspace", self.toolbox.call("write_file", {"path": "/tmp/x", "content": "x"}))


if __name__ == "__main__":
    unittest.main()
