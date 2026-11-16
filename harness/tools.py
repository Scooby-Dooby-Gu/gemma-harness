"""Tools the model can call. File tools are confined to one workspace folder."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MAX_OUTPUT = 4000  # characters returned to the model per tool call
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".next"}


# Commands that are never run, whatever the mode.
BLOCKED_COMMANDS = [r"\bsudo\b", r"\bsu\b\s", r"\bshutdown\b", r"\breboot\b", r"\bmkfs", r"\bdiskutil\b"]

# Commands that need the user's approval even in 'auto' mode.
RISKY_COMMANDS = [
    r"\brm\b",
    r"\bgit\s+(push|reset|clean|rebase|checkout\s+--|restore|branch\s+-D)",
    r"\b(curl|wget|ssh|scp|rsync|nc)\b",
    r"\b(pip3?|npm|pnpm|yarn|brew|uv)\s+(install|add|uninstall|remove|i)\b",
    r"\b(chmod|chown|kill|killall|pkill|dd|launchctl|osascript|open)\b",
    r"\.\./",
    r"(^|\s)~",
    r">\s*/",
]


def command_risk(command: str) -> str:
    """Classify a shell command as 'blocked', 'risky' or 'normal'. A heuristic, not a guarantee."""
    if any(re.search(p, command) for p in BLOCKED_COMMANDS):
        return "blocked"
    if any(re.search(p, command) for p in RISKY_COMMANDS):
        return "risky"
    return "normal"


def sandbox_prefix(workspace: Path) -> list[str]:
    """On macOS, confine a command's file writes to the workspace and temp folders."""
    if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
        return []
    profile = (
        "(version 1)(allow default)(deny file-write*)"
        f'(allow file-write* (subpath "{workspace}") (subpath "/private/tmp") '
        '(subpath "/private/var/folders") (literal "/dev/null") (regex #"^/dev/tty"))'
    )
    return ["sandbox-exec", "-p", profile]


def diff_snapshots(before: dict, after: dict) -> dict:
    return {
        "Files created": sorted(set(after) - set(before)),
        "Files changed": sorted(p for p in after if p in before and after[p] != before[p]),
        "Files deleted": sorted(set(before) - set(after)),
    }


class ToolError(Exception):
    """An error the model should see and can recover from."""


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


SCHEMAS = [
    _fn(
        "list_dir",
        "List files and folders at a path in the workspace.",
        {"path": {"type": "string", "description": "Folder path relative to the workspace. Use '.' for the root."}},
        ["path"],
    ),
    _fn(
        "read_file",
        "Read a text file from the workspace. Returns numbered lines.",
        {
            "path": {"type": "string", "description": "File path relative to the workspace."},
            "start_line": {"type": "integer", "description": "First line to read (1-based). Default 1."},
            "max_lines": {"type": "integer", "description": "How many lines to read. Default 200."},
        },
        ["path"],
    ),
    _fn(
        "edit_file",
        "Change part of an existing file by replacing one piece of text with another. "
        "Use this for small changes instead of rewriting the whole file.",
        {
            "path": {"type": "string", "description": "File path relative to the workspace."},
            "old_text": {"type": "string", "description": "The exact text to replace, copied from the file without line numbers."},
            "new_text": {"type": "string", "description": "The text to put in its place."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of exactly one. Default false."},
        },
        ["path", "old_text", "new_text"],
    ),
    _fn(
        "write_file",
        "Create a new text file, or overwrite a whole file, with the full given content. "
        "To change part of an existing file, use edit_file.",
        {
            "path": {"type": "string", "description": "File path relative to the workspace."},
            "content": {"type": "string", "description": "The complete new content of the file."},
        },
        ["path", "content"],
    ),
    _fn(
        "search",
        "Search file contents in the workspace for a regular expression.",
        {
            "pattern": {"type": "string", "description": "Regular expression to search for."},
            "path": {"type": "string", "description": "Folder to search in. Default '.'."},
        },
        ["pattern"],
    ),
    _fn(
        "run_command",
        "Run a shell command in the workspace folder and return its output and exit code.",
        {"command": {"type": "string", "description": "The shell command to run."}},
        ["command"],
    ),
]

WEB_SCHEMAS = [
    _fn(
        "web_search",
        "Search the web. Returns titles and URLs, with either a short snippet or an extract of the page.",
        {"query": {"type": "string", "description": "What to search for."}},
        ["query"],
    ),
    _fn(
        "fetch_url",
        "Load a web page and return its text. Long pages come in parts; use 'find' to jump to the relevant part.",
        {
            "url": {"type": "string", "description": "The full http or https URL."},
            "find": {"type": "string", "description": "Optional word or phrase; returns the passages around it."},
            "start": {"type": "integer", "description": "Character position to continue reading from. Default 0."},
        },
        ["url"],
    ),
]

FINISH_SCHEMA = _fn(
    "finish",
    "End this step. Call it once the step is complete, or when you cannot complete it.",
    {
        "status": {
            "type": "string",
            "enum": ["done", "failed", "blocked"],
            "description": "done = step complete; failed = tried and could not; blocked = need something from the user.",
        },
        "summary": {"type": "string", "description": "One or two sentences on what you did or why you stopped."},
        "facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short facts you learned that later steps will need (file names, commands, decisions).",
        },
    },
    ["status", "summary"],
)


def _cap(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n[truncated: {len(text) - MAX_OUTPUT} more characters]"


class Toolbox:
    def __init__(self, workspace: Path, commands: str = "off", read_only: bool = False, web: bool = False):
        """commands: 'off' (run_command hidden), 'ask' (confirm each one), 'auto' (confirm only risky ones).
        read_only hides write_file; the verifier uses it."""
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.commands = commands
        self.read_only = read_only
        self.web = web  # adds web_search and fetch_url

    def schemas(self) -> list[dict]:
        hidden = set()
        if self.commands == "off":
            hidden.add("run_command")
        if self.read_only:
            hidden.update({"write_file", "edit_file"})
        return [s for s in SCHEMAS if s["function"]["name"] not in hidden] + (WEB_SCHEMAS if self.web else [])

    def snapshot(self, max_files: int = 5000) -> dict:
        """Size and modified time of every file, to see what a step really changed."""
        snap = {}
        for dirpath, dirnames, filenames in os.walk(self.workspace):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                f = Path(dirpath) / name
                try:
                    st = f.stat()
                except OSError:
                    continue
                snap[self._rel(f)] = (st.st_size, st.st_mtime_ns)
                if len(snap) >= max_files:
                    return snap
        return snap

    def call(self, name: str, args: dict) -> str:
        """Run a tool. Always returns a string for the model, including on error."""
        handler = getattr(self, f"_{name}", None)
        known = {s["function"]["name"] for s in self.schemas()}
        if handler is None or name not in known:
            return f"Error: unknown tool '{name}'. Available tools: {', '.join(sorted(known))}."
        if not isinstance(args, dict):
            return "Error: tool arguments must be a JSON object."
        try:
            return _cap(handler(**args))
        except ToolError as e:
            return f"Error: {e}"
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:  # never let a tool crash the run
            return f"Error: {type(e).__name__}: {e}"

    # --- helpers ------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = (self.workspace / path).resolve()
        if p != self.workspace and not p.is_relative_to(self.workspace):
            raise ToolError(f"'{path}' is outside the workspace. Use paths relative to the workspace.")
        return p

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.workspace)) or "."

    # --- tools --------------------------------------------------------------

    def _list_dir(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.is_dir():
            raise ToolError(f"'{path}' is not a folder.")
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        if not entries:
            return "(empty folder)"
        return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)

    def _read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"'{path}' does not exist or is not a file.")
        try:
            lines = p.read_text().splitlines()
        except UnicodeDecodeError:
            raise ToolError(f"'{path}' is not a text file.")
        start = max(1, int(start_line))
        chunk = lines[start - 1 : start - 1 + int(max_lines)]
        if not chunk:
            return f"(no lines at {start}; the file has {len(lines)} lines)"
        body = "\n".join(f"{start + i}: {line}" for i, line in enumerate(chunk))
        end = start + len(chunk) - 1
        if end < len(lines):
            body += f"\n[lines {start}-{end} of {len(lines)}; read more with start_line={end + 1}]"
        return body

    def _write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if p.is_dir():
            raise ToolError(f"'{path}' is a folder.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content.splitlines())} lines to {self._rel(p)}."

    def _edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"'{path}' does not exist. Use write_file to create a new file.")
        try:
            content = p.read_text()
        except UnicodeDecodeError:
            raise ToolError(f"'{path}' is not a text file.")
        if not old_text:
            raise ToolError("old_text is empty. Give the exact text to replace.")
        if isinstance(replace_all, str):
            replace_all = replace_all.strip().lower() == "true"

        count = content.count(old_text)
        if count == 0:
            # read_file shows "12: code"; the model sometimes copies those line numbers along.
            unnumbered = "\n".join(re.sub(r"^\s*\d+: ?", "", line) for line in old_text.splitlines())
            if unnumbered != old_text and unnumbered and content.count(unnumbered):
                old_text = unnumbered
                new_text = "\n".join(re.sub(r"^\s*\d+: ?", "", line) for line in new_text.splitlines())
                count = content.count(old_text)
        if count == 0:
            raise ToolError(self._not_found_hint(path, content, old_text))
        if count > 1 and not replace_all:
            lines = [str(n) for n, line in enumerate(content.splitlines(), 1) if old_text.splitlines()[0] in line]
            raise ToolError(
                f"old_text appears {count} times in {path} (lines {', '.join(lines[:8])}). "
                "Set replace_all to true to change all of them, or include more surrounding text to pick one."
            )
        if old_text == new_text:
            raise ToolError("old_text and new_text are the same, so nothing would change.")

        first_line = content[: content.index(old_text)].count("\n") + 1
        p.write_text(content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1))
        after = p.read_text().splitlines()
        shown = after[first_line - 1 : first_line - 1 + max(1, min(4, len(new_text.splitlines())))]
        preview = "\n".join(f"{first_line + i}: {line}" for i, line in enumerate(shown))
        times = f"{count} occurrences" if replace_all and count > 1 else "1 occurrence"
        return f"Replaced {times} in {self._rel(p)}. It now reads, from line {first_line}:\n{preview}"

    @staticmethod
    def _not_found_hint(path: str, content: str, old_text: str) -> str:
        """Explain a failed match in a way that lets the model fix its next attempt."""
        wanted = next((line.strip() for line in old_text.splitlines() if line.strip()), "")
        near = [
            f"{n}: {line}" for n, line in enumerate(content.splitlines(), 1)
            if wanted and (line.strip() == wanted or wanted in line or line.strip() in wanted and len(line.strip()) > 8)
        ]
        if near:
            return (
                f"old_text was not found in {path} exactly as given. Spaces, indentation and line breaks must match. "
                "The closest lines in the file are:\n" + "\n".join(near[:5])
            )
        return f"old_text was not found in {path}. Read the file first and copy the text exactly."

    def _search(self, pattern: str, path: str = ".") -> str:
        root = self._resolve(path)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"invalid regular expression: {e}")
        hits = []
        # The path may name one file instead of a folder.
        walk = [(str(root.parent), [], [root.name])] if root.is_file() else os.walk(root)
        for dirpath, dirnames, filenames in walk:
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in sorted(filenames):
                f = Path(dirpath) / fname
                try:
                    if f.stat().st_size > 1_000_000:
                        continue
                    text = f.read_text()
                except (UnicodeDecodeError, OSError):
                    continue
                for n, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{self._rel(f)}:{n}: {line.strip()[:200]}")
                        if len(hits) >= 50:
                            return "\n".join(hits) + "\n[stopped at 50 matches]"
        return "\n".join(hits) if hits else "No matches."

    def _web_search(self, query: str) -> str:
        from . import web

        try:
            return web.search(query)
        except web.WebError as e:
            raise ToolError(str(e))

    def _fetch_url(self, url: str, find: str = "", start: int = 0) -> str:
        from . import web

        try:
            return web.fetch(url, start=start, find=find)[1]
        except web.WebError as e:
            raise ToolError(str(e))

    def _approve(self, command: str, risk: str) -> bool:
        if not sys.stdin.isatty():  # nobody at the terminal to approve it
            return False
        note = " (flagged as risky)" if risk == "risky" else ""
        try:
            return input(f"\n  Model wants to run{note}: {command}\n  Allow? [y/N] ").strip().lower() == "y"
        except EOFError:
            return False

    def _run_command(self, command: str) -> str:
        code, out = self.exec_command(command)
        return f"exit code {code}\n{out}"

    def exec_command(self, command: str) -> tuple[int, str]:
        """Run a command under the approval rules and the sandbox. Returns (exit code, output)."""
        risk = command_risk(command)
        if risk == "blocked":
            raise ToolError("this command is not allowed.")
        if self.commands == "ask" or risk == "risky":
            if not self._approve(command, risk):
                raise ToolError("this command needs the user's approval, and it was not given.")
        try:
            proc = subprocess.run(
                sandbox_prefix(self.workspace) + ["/bin/sh", "-c", command],
                cwd=self.workspace, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise ToolError("command timed out after 120 seconds.")
        return proc.returncode, (proc.stdout + proc.stderr).strip() or "(no output)"
