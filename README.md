# gemma-harness

A small harness that lets a local model (built around `gemma4:12b-mlx` on Ollama) work through multi-step tasks. The model is not trusted to hold a long task in its head: the goal, plan and progress live in a SQLite database, and the model is given one small step at a time with a fresh prompt.

Python 3.11+, standard library only.

## How it works

1. **Plan.** The model turns the goal into at most 8 steps. Each step carries checks the harness can run itself.
2. **Execute.** One step at a time, with a fresh prompt built from the database (goal, plan, recent results, saved facts) and a budget of 8 tool calls.
3. **Verify.** A "done" claim is not believed until it is checked:
   - no successful tool call means the step is rejected;
   - code checks from the plan run with no model involved (`file_exists`, `file_contains`, `file_lacks`, `command_succeeds`, `sources_cited`);
   - a separate model call with a clean context and read-only tools inspects the workspace when no code check could run, or when only a text check failed.
4. **Recover.** A rejected step is retried once with the finding, then the remaining work is re-planned (twice per session at most).
5. **Remember.** At the end of a run the model distils a few notes. Later runs recall notes for the same project and digests of similar earlier runs.

## Tools the model gets

`list_dir`, `read_file`, `edit_file`, `write_file`, `search`, and optionally `run_command`, `web_search` and `fetch_url`.

- File tools cannot leave the workspace folder.
- Commands run under macOS `sandbox-exec`, which limits file writes to the workspace and temp folders. Network access is not blocked. Risky commands need approval at a terminal and are refused otherwise.
- Web fetches refuse local and private addresses, including after redirects.

## Usage

```bash
ollama pull gemma4:12b-mlx
python3 -m harness run "Create slugify.py with tests" --workspace ./workspace --commands auto
python3 -m harness runs
python3 -m harness show 1 --full
python3 -m harness resume 1
python3 -m harness memory
```

Useful options: `--web`, `--no-memory`, `--no-verify`, `--always-inspect`, `--max-minutes`, `--num-ctx`, `--no-think`.

For web search, put `TAVILY_API_KEY=...` (or `BRAVE_API_KEY=...`) in a `.env` file in this folder. With neither, search falls back to Wikipedia only.

## Test tasks

Six fixed coding tasks and one web task, each run in a fresh folder and graded by code that is independent of the harness's own verifier.

```bash
python3 -m harness eval --label "my change"
python3 -m harness eval --task rename --repeat 3 --verbose
python3 -m harness eval --task web-research --web
python3 -m harness evals
```

The number to watch is **false done**: runs where the harness said "done" but the grader failed the work.

Unit tests need no model:

```bash
python3 -m unittest discover tests
```

## Known limits

- The code checks are only as good as what the planner writes. A weak check passes unfinished work and a wrong check rejects good work; both have been observed.
- Nothing checks the overall goal at the end of a run, so a re-plan that drops the real work can end in a false done.
- Results so far come from one or two runs per task on one machine, so pass rates are rough.
- macOS only for sandboxed commands.
