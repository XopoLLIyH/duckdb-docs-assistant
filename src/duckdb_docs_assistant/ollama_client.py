from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class JsonTransport(Protocol):
    def post(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]: ...


class UrllibJsonTransport:
    def post(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Ollama request failed: {error}") from error


@dataclass(frozen=True)
class OllamaResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ns: int
    load_duration_ns: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int


class OllamaClient:
    def __init__(self, config: dict[str, Any], transport: JsonTransport | None = None) -> None:
        self.config = config
        self.transport = transport or UrllibJsonTransport()

    def chat(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> OllamaResponse:
        payload = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": schema,
            "options": self.config["options"],
            "think": self.config["think"],
            "keep_alive": self.config["keep_alive"],
        }
        response = self.transport.post(
            self.config["base_url"].rstrip("/") + "/api/chat",
            payload,
            float(self.config["timeout_seconds"]),
        )
        try:
            return OllamaResponse(
                content=response["message"]["content"],
                model=response["model"],
                prompt_tokens=int(response.get("prompt_eval_count", 0)),
                completion_tokens=int(response.get("eval_count", 0)),
                total_duration_ns=int(response.get("total_duration", 0)),
                load_duration_ns=int(response.get("load_duration", 0)),
                prompt_eval_duration_ns=int(response.get("prompt_eval_duration", 0)),
                eval_duration_ns=int(response.get("eval_duration", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Ollama returned an unexpected response shape") from error
