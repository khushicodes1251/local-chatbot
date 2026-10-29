"""Open the local Chroma store and search uploaded PDF chunks."""

import logging
import os
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

from rag_chatbot import config
from rag_chatbot.ollama_client import embed_text

_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

_CLIENT_CACHE: dict[str, chromadb.PersistentClient] = {}


def get_client(persist_dir: Path) -> chromadb.PersistentClient:
    path = str(persist_dir.resolve())
    if path not in _CLIENT_CACHE:
        Path(path).mkdir(parents=True, exist_ok=True)
        _CLIENT_CACHE[path] = chromadb.PersistentClient(path=path, settings=_CHROMA_SETTINGS)
    return _CLIENT_CACHE[path]


def retrieve(
    question: str,
    *,
    collection,
    top_k: int | None = None,
    where: dict | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """Return the closest chunks from the given upload collection."""
    top_k = top_k or config.TOP_K
    if where is None:
        count = collection.count()
    else:
        matched = collection.get(where=where)
        count = len(matched.get("ids") or [])
    if count <= 0:
        return []

    query = {
        "query_embeddings": [query_embedding if query_embedding is not None else embed_text(question)],
        "n_results": min(top_k, count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query["where"] = where
    result = collection.query(**query)

    hits = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        distance = float(dist)
        hit = {
            "text": doc,
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "distance": distance,
            "similarity": 1.0 - distance,
        }
        if meta.get("page") is not None:
            hit["page"] = int(meta["page"])
        hits.append(hit)
    return hits
