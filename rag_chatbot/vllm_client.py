"""Talk to the already-running local vLLM server (OpenAI-compatible API)."""

from __future__ import annotations

import json
from typing import Iterator

import requests

from rag_chatbot import config

_SESSION = requests.Session()


def chat_stream_vllm(messages: list[dict]) -> Iterator[str]:
    """Stream a chat completion from vLLM. Does not start or change the server."""
    with _SESSION.post(
        f"{config.VLLM_URL}/chat/completions",
        json={
            "model": config.VLLM_MODEL,
            "messages": messages,
            "temperature": config.TEMPERATURE,
            "max_tokens": config.NUM_PREDICT,
            "stream": True,
        },
        stream=True,
        timeout=config.CHAT_TIMEOUT_S,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            delta = (payload.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                yield piece
