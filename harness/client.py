"""Thin wrapper around Ollama's /api/chat."""

import json
import time
import urllib.error
import urllib.request


class ModelError(RuntimeError):
    pass


class Ollama:
    def __init__(
        self,
        model: str = "gemma4:12b-mlx",
        host: str = "http://localhost:11434",
        num_ctx: int = 16384,
        think: bool = True,
        timeout: int = 600,
        retries: int = 2,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        self.retries = retries
        # Cap on one reply (thinking included). Stops a stuck model thinking for many minutes.
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send one chat turn. Returns Ollama's response dict (message, token counts)."""
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            # Set explicitly: Ollama's default window is far smaller than the model's maximum.
            "options": {"num_ctx": self.num_ctx, "num_predict": self.max_tokens},
        }
        if tools:
            body["tools"] = tools
        data = json.dumps(body).encode()

        last_error = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                f"{self.host}/api/chat", data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                # 4xx means the request itself is wrong; retrying won't help.
                if 400 <= e.code < 500:
                    raise ModelError(f"Ollama rejected the request ({e.code}): {detail}") from e
                last_error = f"HTTP {e.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_error = str(e)
            if attempt < self.retries:
                time.sleep(2 * (attempt + 1))
        raise ModelError(f"Ollama call failed after {self.retries + 1} attempts: {last_error}")
