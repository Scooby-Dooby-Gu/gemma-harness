"""SQLite state store: the source of truth for runs, steps, facts and the event log.

The model never reads or writes this directly. Only the harness does.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    goal        TEXT NOT NULL,
    workspace   TEXT NOT NULL,
    model       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',   -- running | done | failed | blocked
    project     TEXT NOT NULL DEFAULT '',          -- memory key: the workspace path, or eval:<task>
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    id           INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    position     INTEGER NOT NULL,
    description  TEXT NOT NULL,
    done_when    TEXT NOT NULL DEFAULT '',
    checks       TEXT NOT NULL DEFAULT '[]',       -- JSON list of code checks, see checks.py
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | failed | blocked | skipped
    summary      TEXT NOT NULL DEFAULT '',
    attempts     INTEGER NOT NULL DEFAULT 0,
    started_at   TEXT,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_id     INTEGER REFERENCES steps(id) ON DELETE SET NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- Append-only log of everything that happened: prompts, responses, tool calls.
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY,
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_id  INTEGER REFERENCES steps(id) ON DELETE SET NULL,
    ts       TEXT NOT NULL,
    kind     TEXT NOT NULL,   -- plan | replan | prompt | model | tool | finish | checks | verify | memory | consolidate | warning | error | run_end
    payload  TEXT NOT NULL    -- JSON
);

-- Notes kept across runs, see memory.py.
CREATE TABLE IF NOT EXISTS memory (
    id             INTEGER PRIMARY KEY,
    scope          TEXT NOT NULL,              -- project | global
    project        TEXT NOT NULL DEFAULT '',   -- which project a project note belongs to
    realm          TEXT NOT NULL DEFAULT 'user',  -- user | eval (test tasks keep separate memory)
    text           TEXT NOT NULL,
    source_run_id  INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    created_at     TEXT NOT NULL,
    last_used_at   TEXT,
    uses           INTEGER NOT NULL DEFAULT 0,
    active         INTEGER NOT NULL DEFAULT 1
);

-- One row per graded test-task run. `passed` is the independent grader's result;
-- `harness_status` is what the harness itself concluded.
CREATE TABLE IF NOT EXISTS eval_results (
    id              INTEGER PRIMARY KEY,
    batch           TEXT NOT NULL,
    label           TEXT NOT NULL DEFAULT '',
    task            TEXT NOT NULL,
    run_id          INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    harness_status  TEXT NOT NULL,
    passed          INTEGER NOT NULL,
    detail          TEXT NOT NULL DEFAULT '',
    seconds         REAL NOT NULL,
    config          TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS steps_run  ON steps(run_id, position);
CREATE INDEX IF NOT EXISTS facts_run  ON facts(run_id);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        # Databases created before code checks existed lack the column.
        if "checks" not in [r["name"] for r in self.db.execute("PRAGMA table_info(steps)")]:
            with self.db:
                self.db.execute("ALTER TABLE steps ADD COLUMN checks TEXT NOT NULL DEFAULT '[]'")
        if "project" not in [r["name"] for r in self.db.execute("PRAGMA table_info(runs)")]:
            with self.db:
                self.db.execute("ALTER TABLE runs ADD COLUMN project TEXT NOT NULL DEFAULT ''")
                self.db.execute("UPDATE runs SET project=workspace")

    # --- runs ---------------------------------------------------------------

    def create_run(self, goal: str, workspace: str, model: str, project: str | None = None) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO runs (goal, workspace, model, project, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (goal, workspace, model, project or workspace, now(), now()),
            )
        return cur.lastrowid

    def set_run_status(self, run_id: int, status: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE id=?", (status, now(), run_id)
            )

    def get_run(self, run_id: int):
        return self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()

    def list_runs(self, limit: int = 30):
        return self.db.execute(
            """SELECT r.*,
                      (SELECT COUNT(*) FROM steps s WHERE s.run_id=r.id) AS n_steps,
                      (SELECT COUNT(*) FROM steps s WHERE s.run_id=r.id AND s.status='done') AS n_done
               FROM runs r ORDER BY r.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    # --- steps --------------------------------------------------------------

    def add_step(self, run_id: int, description: str, done_when: str = "", checks: list | None = None) -> int:
        with self.db:
            pos = self.db.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM steps WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            cur = self.db.execute(
                "INSERT INTO steps (run_id, position, description, done_when, checks) VALUES (?,?,?,?,?)",
                (run_id, pos, description, done_when, json.dumps(checks or [], ensure_ascii=False)),
            )
        return cur.lastrowid

    def get_step(self, step_id: int):
        return self.db.execute("SELECT * FROM steps WHERE id=?", (step_id,)).fetchone()

    def list_steps(self, run_id: int):
        return self.db.execute(
            "SELECT * FROM steps WHERE run_id=? ORDER BY position", (run_id,)
        ).fetchall()

    def next_pending_step(self, run_id: int):
        return self.db.execute(
            "SELECT * FROM steps WHERE run_id=? AND status='pending' ORDER BY position LIMIT 1",
            (run_id,),
        ).fetchone()

    def skip_pending_steps(self, run_id: int) -> None:
        """Retire the not-yet-run steps when the plan is replaced."""
        with self.db:
            self.db.execute(
                "UPDATE steps SET status='skipped' WHERE run_id=? AND status='pending'", (run_id,)
            )

    def reset_running_steps(self, run_id: int) -> None:
        with self.db:
            self.db.execute(
                "UPDATE steps SET status='pending' WHERE run_id=? AND status='running'", (run_id,)
            )

    def retry_failed_steps(self, run_id: int) -> None:
        with self.db:
            self.db.execute(
                "UPDATE steps SET status='pending' WHERE run_id=? AND status IN ('failed','blocked')",
                (run_id,),
            )

    def start_step(self, step_id: int) -> None:
        with self.db:
            self.db.execute(
                "UPDATE steps SET status='running', attempts=attempts+1, started_at=? WHERE id=?",
                (now(), step_id),
            )

    def requeue_step(self, step_id: int, summary: str) -> None:
        """Put a step back in the queue, keeping a note of why the last attempt didn't count."""
        with self.db:
            self.db.execute("UPDATE steps SET status='pending', summary=? WHERE id=?", (summary, step_id))

    def finish_step(self, step_id: int, status: str, summary: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE steps SET status=?, summary=?, finished_at=? WHERE id=?",
                (status, summary, now(), step_id),
            )

    # --- facts --------------------------------------------------------------

    def add_fact(self, run_id: int, step_id: int | None, text: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO facts (run_id, step_id, text, created_at) VALUES (?,?,?,?)",
                (run_id, step_id, text, now()),
            )

    def list_facts(self, run_id: int) -> list[str]:
        rows = self.db.execute(
            "SELECT text FROM facts WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [r["text"] for r in rows]

    def finished_runs(self, exclude: int, limit: int = 300):
        return self.db.execute(
            "SELECT * FROM runs WHERE status IN ('done','failed') AND id != ? ORDER BY id DESC LIMIT ?",
            (exclude, limit),
        ).fetchall()

    def fetched_urls(self, run_id: int) -> set[str]:
        """Every page whose content the model was shown in this run: pages fetch_url loaded
        (as asked for, and after redirects), and search results that came with page extracts."""
        from .web import EXTRACTS_MARK, find_urls, normalize_url

        urls = set()
        rows = self.db.execute("SELECT payload FROM events WHERE run_id=? AND kind='tool'", (run_id,)).fetchall()
        for row in rows:
            p = json.loads(row["payload"])
            result = str(p.get("result", ""))
            if p.get("name") == "web_search" and result.startswith(EXTRACTS_MARK):
                for line in result.splitlines():
                    if line.strip().startswith("URL: "):
                        urls.update(find_urls(line))
                continue
            if p.get("name") != "fetch_url" or result.startswith("Error:"):
                continue
            asked = (p.get("arguments") or {}).get("url")
            if asked:
                urls.add(normalize_url(str(asked)))
            for line in result.splitlines()[:4]:
                if line.startswith("URL: "):
                    urls.update(find_urls(line))
        return urls

    # --- memory -------------------------------------------------------------

    def add_memory(self, scope: str, project: str, realm: str, text: str, run_id: int | None) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO memory (scope, project, realm, text, source_run_id, created_at) VALUES (?,?,?,?,?,?)",
                (scope, project, realm, text, run_id, now()),
            )
        return cur.lastrowid

    def list_memory(self, realm: str | None = None, include_inactive: bool = False):
        sql, args = "SELECT * FROM memory WHERE 1=1", []
        if realm:
            sql += " AND realm=?"
            args.append(realm)
        if not include_inactive:
            sql += " AND active=1"
        return self.db.execute(sql + " ORDER BY id", args).fetchall()

    def touch_memory(self, ids: list[int]) -> None:
        with self.db:
            self.db.executemany(
                "UPDATE memory SET uses=uses+1, last_used_at=? WHERE id=?", [(now(), i) for i in ids]
            )

    def forget_memory(self, memory_id: int) -> bool:
        with self.db:
            return self.db.execute("UPDATE memory SET active=0 WHERE id=?", (memory_id,)).rowcount > 0

    def trim_memory(self, project: str, keep: int) -> None:
        """Retire a project's least recently useful notes beyond the cap."""
        with self.db:
            self.db.execute(
                """UPDATE memory SET active=0 WHERE active=1 AND scope='project' AND project=? AND id NOT IN (
                       SELECT id FROM memory WHERE active=1 AND scope='project' AND project=?
                       ORDER BY COALESCE(last_used_at, created_at) DESC, id DESC LIMIT ?)""",
                (project, project, keep),
            )

    # --- test-task results --------------------------------------------------

    def add_eval_result(self, batch, label, task, run_id, harness_status, passed, detail, seconds, config) -> None:
        with self.db:
            self.db.execute(
                """INSERT INTO eval_results
                   (batch, label, task, run_id, harness_status, passed, detail, seconds, config, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (batch, label, task, run_id, harness_status, int(passed), detail, seconds, config, now()),
            )

    def eval_batches(self, limit: int = 20):
        return self.db.execute(
            """SELECT e.batch, e.label,
                      COUNT(*) AS n,
                      SUM(e.passed) AS passed,
                      SUM(e.harness_status='done' AND NOT e.passed) AS false_done,
                      SUM(e.harness_status!='done' AND e.passed) AS false_fail,
                      SUM(e.seconds) AS seconds,
                      (SELECT COUNT(*) FROM events v WHERE v.kind='model'
                         AND v.run_id IN (SELECT run_id FROM eval_results x WHERE x.batch=e.batch)) AS model_calls
               FROM eval_results e GROUP BY e.batch ORDER BY e.batch DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    def eval_rows(self, batch: str):
        return self.db.execute(
            """SELECT e.*,
                      (SELECT COUNT(*) FROM events v WHERE v.run_id=e.run_id AND v.kind='model') AS model_calls,
                      (SELECT COUNT(*) FROM steps s WHERE s.run_id=e.run_id AND s.status!='skipped') AS steps,
                      (SELECT COUNT(*) FROM events v WHERE v.run_id=e.run_id AND v.kind='replan') AS replans,
                      (SELECT COUNT(*) FROM events v WHERE v.run_id=e.run_id AND v.kind='verify'
                         AND json_extract(v.payload, '$.passed')=0) AS rejections
               FROM eval_results e WHERE e.batch=? ORDER BY e.id""",
            (batch,),
        ).fetchall()

    # --- events -------------------------------------------------------------

    def event(self, run_id: int, step_id: int | None, kind: str, payload: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO events (run_id, step_id, ts, kind, payload) VALUES (?,?,?,?,?)",
                (run_id, step_id, now(), kind, json.dumps(payload, ensure_ascii=False)),
            )

    def count_events(self, run_id: int, kind: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND kind=?", (run_id, kind)
        ).fetchone()[0]

    def list_events(self, run_id: int):
        return self.db.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
