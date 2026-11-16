"""Code checks: conditions the planner attaches to a step, which the harness tests itself.

The types:
  file_exists       {"type": "file_exists", "path": "a.py"}            the file exists and is not empty
  file_contains     {"type": "file_contains", "path": "a.py", "text": "def f"}
  file_lacks        {"type": "file_lacks", "path": "a.py", "text": "old_name"}   the text is gone from the file
  command_succeeds  {"type": "command_succeeds", "command": "python3 -m unittest"}   exits with code 0
  sources_cited     {"type": "sources_cited", "path": "answer.md", "min": 2}
                    the file contains at least `min` URLs, and every URL in it is a page whose
                    content the model was really shown during this run (so no invented links)

No model is involved in running them, so they are fast and cannot be talked round.
"""

from dataclasses import dataclass

from .tools import Toolbox, ToolError

TYPES = ("file_exists", "file_contains", "file_lacks", "command_succeeds", "sources_cited")


def allowed_types(tool_names: list[str]) -> set[str]:
    """Which check types make sense given the tools that are switched on."""
    allowed = {"file_exists", "file_contains", "file_lacks"}
    if "run_command" in tool_names:
        allowed.add("command_succeeds")
    if "fetch_url" in tool_names:
        allowed.add("sources_cited")
    return allowed


@dataclass
class CheckResult:
    check: dict
    status: str  # passed | failed | skipped (could not be run, so it proves nothing)
    detail: str


def describe(check: dict) -> str:
    kind = check["type"]
    if kind == "file_exists":
        return f"the file {check['path']} exists and is not empty"
    if kind == "file_contains":
        return f"the file {check['path']} contains the text: {check['text']}"
    if kind == "file_lacks":
        return f"the file {check['path']} no longer contains the text: {check['text']}"
    if kind == "sources_cited":
        return (f"the file {check['path']} gives at least {check['min']} source URL(s), "
                "and every URL in it is a page that was actually read")
    return f"the command `{check['command']}` exits with code 0"


def clean_checks(raw, allowed: set[str]) -> tuple[list[dict], list]:
    """Keep the well-formed checks from the planner's output. Returns (kept, dropped)."""
    kept, dropped = [], []
    if not isinstance(raw, list):
        return kept, ([raw] if raw else [])
    for c in raw:
        kind = c.get("type") if isinstance(c, dict) else None
        path = str(c.get("path") or "").strip() if isinstance(c, dict) else ""
        if kind not in allowed:
            dropped.append(c)
        elif kind == "file_exists" and path:
            kept.append({"type": kind, "path": path})
        elif kind in ("file_contains", "file_lacks") and path and str(c.get("text") or "").strip():
            kept.append({"type": kind, "path": path, "text": str(c["text"]).strip()})
        elif kind == "command_succeeds" and str(c.get("command") or "").strip():
            kept.append({"type": kind, "command": str(c["command"]).strip()})
        elif kind == "sources_cited" and path:
            try:
                minimum = max(1, min(10, int(c.get("min") or 1)))
            except (TypeError, ValueError):
                minimum = 1
            kept.append({"type": kind, "path": path, "min": minimum})
        else:
            dropped.append(c)
    return kept[:5], dropped


def _squash(text: str) -> str:
    return " ".join(text.split())


def run_check(toolbox: Toolbox, check: dict, fetched: set[str] = frozenset()) -> CheckResult:
    kind = check["type"]
    try:
        if kind == "sources_cited":
            from .web import find_urls

            path = toolbox._resolve(check["path"])
            if not path.is_file():
                return CheckResult(check, "failed", f"the file {check['path']} does not exist.")
            cited = find_urls(path.read_text(errors="replace"))
            invented = [u for u in cited if u not in fetched]
            if invented:
                return CheckResult(
                    check, "failed",
                    f"the file {check['path']} cites pages that were never read in this task: "
                    + ", ".join(invented[:4]) + ". Only cite pages whose content you were shown.",
                )
            if len(cited) < check["min"]:
                return CheckResult(
                    check, "failed",
                    f"the file {check['path']} gives {len(cited)} source URL(s); at least {check['min']} are needed.",
                )
            return CheckResult(check, "passed", f"{check['path']} cites {len(cited)} source(s) that were read.")

        if kind in ("file_exists", "file_contains", "file_lacks"):
            path = toolbox._resolve(check["path"])
            if not path.is_file():
                return CheckResult(check, "failed", f"the file {check['path']} does not exist.")
            if kind == "file_exists":
                if path.stat().st_size == 0:
                    return CheckResult(check, "failed", f"the file {check['path']} is empty.")
                return CheckResult(check, "passed", f"{check['path']} exists.")
            try:
                content = path.read_text()
            except UnicodeDecodeError:
                return CheckResult(check, "failed", f"the file {check['path']} is not a text file.")
            if kind == "file_lacks":
                lines = [i for i, line in enumerate(content.splitlines(), 1) if check["text"] in line]
                if not lines:
                    return CheckResult(check, "passed", f"{check['path']} no longer contains '{check['text']}'.")
                return CheckResult(
                    check, "failed",
                    f"the file {check['path']} still contains '{check['text']}' on line(s) "
                    + ", ".join(map(str, lines[:8])) + ".",
                )
            # Whitespace differences (indentation, line breaks) don't count as a mismatch.
            if _squash(check["text"]) in _squash(content):
                return CheckResult(check, "passed", f"{check['path']} contains the expected text.")
            return CheckResult(
                check, "failed", f"the file {check['path']} does not contain the expected text: {check['text']}"
            )

        if toolbox.commands == "off":
            return CheckResult(check, "skipped", "commands are switched off.")
        code, output = toolbox.exec_command(check["command"])
        if code == 0:
            return CheckResult(check, "passed", f"`{check['command']}` exited with code 0.")
        tail = output[-600:]
        return CheckResult(check, "failed", f"the command `{check['command']}` exited with code {code}:\n{tail}")
    except ToolError as e:
        if "timed out" in str(e):
            return CheckResult(check, "failed", f"the command `{check.get('command')}` timed out.")
        return CheckResult(check, "skipped", str(e))  # outside the workspace, not approved, not allowed


def run_checks(toolbox: Toolbox, checks: list[dict], fetched: set[str] = frozenset()) -> list[CheckResult]:
    return [run_check(toolbox, c, fetched) for c in checks]
