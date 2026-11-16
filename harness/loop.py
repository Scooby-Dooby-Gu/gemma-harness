"""The shared model/tool loop: let the model call tools until it calls a terminal tool.

The executor ends on `finish`; the verifier ends on `verdict`. Both get the same
budget, repeat-call block and plain-text nudge.
"""

import json
from collections import Counter
from dataclasses import dataclass

from .client import ModelError, Ollama


@dataclass
class LoopResult:
    args: dict | None  # the terminal tool's arguments, or None if the loop ended without them
    reason: str = ""  # why it ended without them
    calls: int = 0  # tool calls attempted
    ok_calls: int = 0  # tool calls that ran without an error


def tool_loop(
    client: Ollama,
    messages: list[dict],
    schemas: list[dict],
    call_tool,  # (name, args) -> str
    terminal: str,  # name of the tool that ends the loop
    check_terminal,  # (args) -> error string, or None if the arguments are acceptable
    max_tool_calls: int,
    log,  # (kind, payload) -> None
    on_tool=None,
) -> LoopResult:
    say = on_tool or (lambda *_: None)
    out = LoopResult(args=None)
    seen = Counter()
    nudged = False

    # A few turns beyond the tool budget so the model can still call the terminal tool.
    for _ in range(max_tool_calls + 4):
        try:
            resp = client.chat(messages, tools=schemas)
        except ModelError as e:
            log("error", {"error": str(e)})
            out.reason = f"Model call failed: {e}"
            return out

        msg = resp.get("message", {})
        tool_calls = msg.get("tool_calls") or []
        tokens_in = resp.get("prompt_eval_count", 0)
        log(
            "model",
            {
                "content": msg.get("content", ""),
                "thinking": msg.get("thinking", ""),
                "tool_calls": tool_calls,
                "tokens_in": tokens_in,
                "tokens_out": resp.get("eval_count", 0),
                "seconds": round(resp.get("total_duration", 0) / 1e9, 1),
                # Time spent reading the prompt. Small when the model's cache covered most of it.
                "prompt_seconds": round(resp.get("prompt_eval_duration", 0) / 1e9, 2),
            },
        )
        if tokens_in > 0.85 * client.num_ctx:
            log("warning", {"warning": f"prompt used {tokens_in} of {client.num_ctx} context tokens"})

        # Earlier thinking is dropped from the history to save context.
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})

        if not tool_calls:
            if nudged:
                out.reason = f"The model stopped without calling {terminal}."
                return out
            nudged = True
            if resp.get("done_reason") == "length":
                nudge = f"Your reply was cut off because it was too long. Think briefly, then call a tool or call {terminal}."
            else:
                nudge = f"Do not answer in text. Call a tool to continue, or call {terminal}."
            messages.append({"role": "user", "content": nudge})
            continue

        for tc in tool_calls:
            fn = tc.get("function", {})
            name, args = fn.get("name", ""), fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None

            if name == terminal and isinstance(args, dict):
                problem = check_terminal(args)
                if problem is None:
                    out.args = args
                    return out
                result = f"Error: {problem}"
            else:
                out.calls += 1
                key = json.dumps([name, args], sort_keys=True)
                seen[key] += 1
                if out.calls > max_tool_calls:
                    result = f"Error: the tool budget is used up. Call {terminal} now."
                elif seen[key] > 2:
                    result = f"Error: you already made this exact call twice. Try something different or call {terminal}."
                else:
                    result = call_tool(name, args)
                    if not result.startswith("Error:"):
                        out.ok_calls += 1
                        # After a change, looking at the same thing again is no longer a repeat.
                        if name in ("edit_file", "write_file", "run_command"):
                            seen.clear()
                say(name, args, result)
                log("tool", {"name": name, "arguments": args, "result": result})

            messages.append({"role": "tool", "tool_name": name, "content": result})

    out.reason = "Ran out of turns without finishing."
    return out
