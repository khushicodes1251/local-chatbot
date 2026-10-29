"""Thin wrappers around the local Ollama HTTP API."""

from __future__ import annotations

import json
import time
from typing import Iterator

import requests

from rag_chatbot import config

_SESSION = requests.Session()
_MODELS_CHECKED_AT = 0.0
_MODELS_CACHE_TTL_S = 300.0
_CACHED_MODEL_NAMES: set[str] | None = None


def list_models(force: bool = False) -> set[str]:
    global _CACHED_MODEL_NAMES, _MODELS_CHECKED_AT
    now = time.monotonic()
    if (
        not force
        and _CACHED_MODEL_NAMES is not None
        and (now - _MODELS_CHECKED_AT) < _MODELS_CACHE_TTL_S
    ):
        return _CACHED_MODEL_NAMES

    response = _SESSION.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
    response.raise_for_status()
    _CACHED_MODEL_NAMES = {model["name"] for model in response.json().get("models", [])}
    _MODELS_CHECKED_AT = now
    return _CACHED_MODEL_NAMES


def _has_model(names: set[str], prefix: str) -> bool:
    return any(name.startswith(prefix) for name in names)


def require_models(need_chat: bool = True, force: bool = False) -> None:
    """Fail fast if Ollama or required models are missing (cached briefly)."""
    try:
        names = list_models(force=force)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Cannot reach Ollama at 127.0.0.1:11434. Start it with: ollama serve"
        ) from exc

    if not _has_model(names, config.EMBED_MODEL):
        raise RuntimeError(
            f"Missing embedding model. Run: ollama pull {config.EMBED_MODEL}"
        )
    if need_chat and not _has_model(names, config.CHAT_MODEL):
        raise RuntimeError(
            f"Missing chat model. Run: ollama pull {config.CHAT_MODEL}"
        )


def embed_text(text: str) -> list[float]:
    response = _SESSION.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json={
            "model": config.EMBED_MODEL,
            "prompt": text,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        },
        timeout=config.EMBED_TIMEOUT_S,
    )
    response.raise_for_status()
    vector = response.json().get("embedding") or []
    if not vector:
        raise RuntimeError("Ollama returned an empty embedding")
    return vector


def _chat_payload(
    messages: list[dict],
    stream: bool,
    *,
    model: str | None = None,
    num_predict: int | None = None,
    temperature: float | None = None,
    keep_alive: str | None = None,
) -> dict:
    """Build an Ollama /api/chat payload. Defaults come from config (app default unchanged)."""
    return {
        "model": model or config.CHAT_MODEL,
        "stream": stream,
        "messages": messages,
        "keep_alive": config.OLLAMA_KEEP_ALIVE if keep_alive is None else keep_alive,
        "options": {
            "num_predict": config.NUM_PREDICT if num_predict is None else num_predict,
            "temperature": config.TEMPERATURE if temperature is None else temperature,
        },
    }


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    num_predict: int | None = None,
    temperature: float | None = None,
    keep_alive: str | None = None,
) -> str:
    """Non-streaming chat (full answer when complete)."""
    response = _SESSION.post(
        f"{config.OLLAMA_URL}/api/chat",
        json=_chat_payload(
            messages,
            stream=False,
            model=model,
            num_predict=num_predict,
            temperature=temperature,
            keep_alive=keep_alive,
        ),
        timeout=config.CHAT_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def chat_stream(
    messages: list[dict],
    *,
    model: str | None = None,
    num_predict: int | None = None,
    temperature: float | None = None,
    keep_alive: str | None = None,
) -> Iterator[str]:
    """Yield text chunks as they arrive from Ollama (NDJSON stream)."""
    with _SESSION.post(
        f"{config.OLLAMA_URL}/api/chat",
        json=_chat_payload(
            messages,
            stream=True,
            model=model,
            num_predict=num_predict,
            temperature=temperature,
            keep_alive=keep_alive,
        ),
        stream=True,
        timeout=config.CHAT_TIMEOUT_S,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            message = data.get("message") or {}
            piece = message.get("content") or ""
            if piece:
                yield piece
            if data.get("done"):
                break
