from __future__ import annotations

from typing import Any

from duckdb_docs_assistant.ollama_client import OllamaClient


class FakeTransport:
    def __init__(self) -> None:
        self.request: tuple[str, dict[str, Any], float] | None = None

    def post(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.request = (url, payload, timeout)
        return {
            "model": "qwen3:8b-q4_K_M",
            "message": {"role": "assistant", "content": '{"status":"answered"}'},
            "prompt_eval_count": 100,
            "eval_count": 20,
            "total_duration": 123,
            "load_duration": 12,
            "prompt_eval_duration": 34,
            "eval_duration": 77,
        }


def test_ollama_client_sends_non_streaming_structured_request() -> None:
    config = {
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen3:8b-q4_K_M",
        "options": {"num_ctx": 4096, "temperature": 0.1},
        "think": False,
        "keep_alive": "2m",
        "timeout_seconds": 180,
    }
    transport = FakeTransport()
    client = OllamaClient(config, transport)
    schema = {"type": "object"}

    response = client.chat("system", "user", schema)

    assert transport.request is not None
    url, payload, timeout = transport.request
    assert url == "http://127.0.0.1:11434/api/chat"
    assert payload["stream"] is False
    assert payload["format"] == schema
    assert payload["think"] is False
    assert payload["messages"][0] == {"role": "system", "content": "system"}
    assert timeout == 180.0
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 20
    assert response.load_duration_ns == 12
    assert response.prompt_eval_duration_ns == 34
    assert response.eval_duration_ns == 77
