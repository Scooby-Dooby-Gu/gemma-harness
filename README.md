# gemma-harness

A harness that lets a small local model finish multi-step tasks.

A 12B model running on a laptop loses track of a long task. It forgets the goal, repeats itself, and says "done" when the work isn't. gemma-harness takes the long task away from it. The goal, the plan and the progress live in a SQLite database. The model gets one small step at a time with a fresh prompt, and nothing it claims is believed until the harness has checked it.

It is built around `gemma4:12b-mlx` on [Ollama](https://ollama.com) and written in plain Python with no dependencies.

**[Interactive walkthrough](https://ai.gufamily.org/harness)**: every layer of the harness, stepped through with a real run.

## Quick start

You need Python 3.11 or later and Ollama running locally.

```bash
git clone https://github.com/Scooby-Dooby-Gu/gemma-harness
cd gemma-harness
ollama pull gemma4:12b-mlx
python3 -m harness run "Create slugify.py with tests" --workspace ./workspace --commands auto
```

The harness prints the plan, then each step as it is executed and checked. Everything it does is recorded, so you can look afterwards:

```bash
python3 -m harness runs             # list past runs
python3 -m harness show 1 --full    # one run, step by step
python3 -m harness resume 1         # continue a run that stopped
```

## How it works

A run passes through these layers. The middle four repeat for every step.

| Layer | What it does |
|---|---|
| **Plan** | The model turns the goal into at most 8 small steps. Each step carries checks the harness can run itself. |
| **Late checks** | Just before a step runs, a separate call rewrites its checks with the files in view. The planner wrote them before it had seen any file. |
| **Execute** | The model works on one step with a fresh prompt built from the database: the goal, the plan, recent results and saved facts. It gets 8 tool calls. |
| **Verify** | A "done" claim is checked before it counts. See below. |
| **Recover** | A rejected step is retried once with the verifier's finding. If it fails again, the remaining work is re-planned. |
| **Goal check** | When no steps are left, an inspector with a clean context checks the task itself. Every step can pass while the plan left the real work out. If it finds a gap, the missing work is planned. |
| **Remember** | The model distils a few notes from the run. Later runs in the same project recall them. |

### Verification

Verification runs in three layers, cheapest first.

1. **Evidence.** A step with no successful tool call is rejected outright.
2. **Code checks.** These run with no model involved: `file_exists`, `file_contains`, `file_lacks`, `command_succeeds` and `sources_cited`. They are also run *before* the step, and a check that was already true doesn't count as proof.
3. **Inspection.** A separate model call with a clean context and read-only tools looks at the workspace. It runs when no code check could, when only a text check failed, or when a step changed files but none of its checks newly became true.

### Memory

Memory is the only thing that connects one run to the next. It is written once at the end of a run and read by the planner, the step prompt and the re-planner in later runs. The roles that check work never see it, so a wrong note can't talk a checker into passing something.

```bash
python3 -m harness memory                     # list notes for this project
python3 -m harness memory add "Tests run with pytest -q"
python3 -m harness memory forget 12
```

## Tools and safety

The model gets `list_dir`, `read_file`, `edit_file`, `write_file` and `search`. Shell commands and web access are off unless you turn them on.

| Option | Effect |
|---|---|
| `--commands off` | No shell commands. This is the default. |
| `--commands ask` | The model can run commands, and you approve each one. |
| `--commands auto` | Commands run without asking, except risky ones. |
| `--web` | Adds `web_search` and `fetch_url`. |

- File tools can't leave the workspace folder.
- On macOS, commands run under `sandbox-exec`, which limits file writes to the workspace and temp folders. Network access is not blocked. On other systems there is no write restriction, so use `--commands ask`.
- Web fetches refuse local and private addresses, including after redirects.

### Web search

Search works out of the box against Wikipedia. For general web search, add a key for [Tavily](https://tavily.com) or [Brave Search](https://brave.com/search/api/) to a `.env` file in the project folder:

```bash
TAVILY_API_KEY=your-key
# or
BRAVE_API_KEY=your-key
```

## Options

| Option | Effect |
|---|---|
| `--workspace PATH` | The folder the model works in. |
| `--model NAME` | Any Ollama model that supports tool calls. |
| `--num-ctx N` | Context window in tokens. |
| `--max-tool-calls N` | Tool calls allowed per step. |
| `--max-minutes N` | Time limit for the run. |
| `--no-think` | Turn off the model's thinking. |
| `--no-memory` | Neither use nor save notes from other runs. |
| `--no-late-checks` | Use the plan's checks as written. |
| `--always-inspect` | Inspect every step, including ones whose code checks passed. |
| `--no-verify` | Skip verification. Useful only for comparison. |

## Evaluation

The repo includes six fixed coding tasks and one web task. Each runs in a fresh folder and is graded by code that is independent of the harness's own verifier.

| Task | What the model has to do |
|---|---|
| `fix-bug` | Find and fix a bug so the failing tests pass. |
| `slugify` | Write a function to a spec, with tests. |
| `rename` | Rename a function across several files. |
| `totals` | Write and run a script that summarises a CSV file. |
| `todo-app` | Build a small command-line app with stored state. |
| `release-notes` | Gather four facts from the workspace into one file. |
| `web-research` | Answer two questions from the web and cite the sources. |

```bash
python3 -m harness eval --label "my change"             # every coding task once
python3 -m harness eval --task rename --repeat 3 --verbose
python3 -m harness eval --task web-research --web
python3 -m harness evals                                # compare batches
```

The number to watch is **false done**: runs where the harness said "done" but the grader failed the work. A harness that fails honestly is more useful than one that passes wrongly.

### Results

The six coding tasks, run twice each, with `gemma4:12b-mlx` on an M5 MacBook with 24 GB of memory:

| Configuration | Passed | False done | Model calls |
|---|---|---|---|
| Step verification only | 11 of 12 | 1 | 143 |
| With the goal check | 12 of 12 | 0 | 317 |

The goal check removed the one false "done" at more than double the model calls. Two runs per task is a small sample, so read these as a direction and not a benchmark.

## Project layout

```text
harness/
  runner.py      Plans a goal, then executes and verifies the steps one at a time.
  planner.py     Turns a goal into steps, and re-plans when a step fails.
  checker.py     Rewrites a step's checks just before it runs.
  executor.py    Runs one step with a fresh prompt and a small tool budget.
  verifier.py    Checks a step's "done" claim, and the goal at the end.
  checks.py      The code checks the harness runs itself.
  review.py      Shared set-up for the read-only checking roles.
  context.py     Builds each step's prompt from the database.
  loop.py        The shared model and tool loop.
  memory.py      Notes and digests carried between runs.
  store.py       The SQLite store for runs, steps, facts and events.
  tools.py       File and command tools, confined to the workspace.
  web.py         Search and page fetching.
  client.py      A thin wrapper around Ollama's chat API.
  evals.py       The fixed test tasks and their graders.
tests/           Unit tests. They need no model.
```

## Development

```bash
python3 -m unittest discover tests
```

The unit tests use a scripted stand-in for the model, so they run in a few seconds without Ollama. To measure a change against the real model, run `python3 -m harness eval` with a `--label` before and after, then compare with `python3 -m harness evals`.
