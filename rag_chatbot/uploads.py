"""Index PDFs the user uploads and answer only from those files.

Embeddings stay on the local Ollama model. Answers use the local vLLM server,
with the local Ollama chat model as a fallback. No built-in document set is searched.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from rag_chatbot import config
from rag_chatbot.ollama_client import embed_text, require_models
from rag_chatbot.vectorstore import get_client, retrieve

ALLOWED_EXTENSIONS = {".pdf"}
_MAX_CHARS = 700
_CHUNK_CHARS = 700
_CHUNK_OVERLAP = 120
_MANIFEST = "manifest.json"
# Sample file previously indexed for demos. Never retrieve it.
_SAMPLE_NAME = "harbor-note.txt"


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf" if name else "upload.pdf"
    return name


def _manifest_path() -> Path:
    return config.UPLOAD_DIR / _MANIFEST


def _load_manifest() -> list[dict]:
    path = _manifest_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save_manifest(rows: list[dict]) -> None:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _window_end(para: str, start: int) -> int:
    """End a window on a space when that does not make it much shorter."""
    end = min(start + _CHUNK_CHARS, len(para))
    if end >= len(para):
        return end
    space = para.rfind(" ", start + 1, end)
    if space >= start + (_CHUNK_CHARS // 2):
        return space
    return end


def chunk_document_text(text: str) -> list[str]:
    """Paragraph chunks. Long paragraphs overlap so a fact on the cut stays searchable."""
    pieces: list[str] = []
    for para in _paragraphs(text):
        if len(para) <= _CHUNK_CHARS:
            pieces.append(para)
            continue
        start = 0
        while start < len(para):
            end = _window_end(para, start)
            piece = para[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(para):
                break
            next_start = end - _CHUNK_OVERLAP
            if next_start <= start:
                next_start = end
            start = next_start
    return [p for p in pieces if p]


def extract_text(path: Path) -> str:
    """Join extractable PDF page text. Empty when the PDF has no text layer."""
    pages = extract_pdf_pages(path)
    return "\n\n".join(page["text"] for page in pages if page["text"]).strip()


def extract_pdf_pages(path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("This PDF is encrypted and could not be read.") from exc
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"page": index, "text": text})
    return pages


def _upload_collection():
    client = get_client(config.UPLOAD_CHROMA_DIR)
    name = config.UPLOAD_COLLECTION
    names = {str(n) for n in client.list_collections()}
    if name not in names:
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return client.get_collection(name)


def _delete_source(source: str) -> None:
    try:
        collection = _upload_collection()
    except Exception:
        return
    found = collection.get(where={"source": source})
    ids = found.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def purge_sample_note() -> None:
    """Drop the old demo note so it cannot answer document questions."""
    path = config.UPLOAD_DIR / _SAMPLE_NAME
    if path.is_file():
        path.unlink()
    _delete_source(_SAMPLE_NAME)
    rows = [row for row in _load_manifest() if row.get("name") != _SAMPLE_NAME]
    if rows != _load_manifest():
        _save_manifest(rows)


def list_uploaded_files() -> list[dict]:
    purge_sample_note()
    by_name = {row["name"]: row for row in _load_manifest()}
    folder = config.UPLOAD_DIR
    if not folder.is_dir():
        return []
    files = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        meta = by_name.get(path.name, {})
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "pages": meta.get("pages"),
                "chunks": meta.get("chunks"),
                "status": "ready",
            }
        )
    return files


def _chunks_for_pdf(path: Path) -> tuple[list[dict], int]:
    try:
        pages = extract_pdf_pages(path)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read this PDF: {exc}") from exc
    if not pages:
        raise ValueError("This PDF has no pages.")
    chunks = []
    for page in pages:
        if not page["text"]:
            continue
        for piece in chunk_document_text(page["text"]):
            chunks.append({"text": piece, "page": page["page"]})
    if not chunks:
        raise ValueError(
            "No text could be extracted. This PDF looks scanned or image-only. "
            "OCR is not enabled, so upload a PDF with selectable text."
        )
    return chunks, len(pages)


def index_upload(filename: str, data: bytes) -> dict:
    """Save a PDF, embed its chunks, and add them to the upload collection."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Upload a PDF. Other file types are not accepted.")
    if not data:
        raise ValueError("The file is empty.")

    purge_sample_note()
    digest = hashlib.sha256(data).hexdigest()
    existing = next((row for row in _load_manifest() if row.get("sha256") == digest), None)
    if existing and (config.UPLOAD_DIR / existing["name"]).is_file():
        return {
            "name": existing["name"],
            "chunks": existing.get("chunks", 0),
            "pages": existing.get("pages"),
            "duplicate": True,
            "status": "ready",
        }

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(filename)
    path = config.UPLOAD_DIR / safe
    path.write_bytes(data)

    try:
        chunks, page_count = _chunks_for_pdf(path)
    except ValueError:
        path.unlink(missing_ok=True)
        raise

    require_models(need_chat=False)
    collection = _upload_collection()
    _delete_source(safe)
    ids, documents, metadatas, embeddings = [], [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{safe}::{digest[:12]}::{i}")
        documents.append(chunk["text"])
        metadatas.append({"source": safe, "chunk_id": i, "page": int(chunk["page"]), "sha256": digest})
        embeddings.append(embed_text(chunk["text"]))
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    rows = [row for row in _load_manifest() if row.get("name") != safe]
    rows.append(
        {
            "name": safe,
            "sha256": digest,
            "pages": page_count,
            "chunks": len(chunks),
        }
    )
    _save_manifest(rows)
    return {
        "name": safe,
        "chunks": len(chunks),
        "pages": page_count,
        "duplicate": False,
        "status": "ready",
    }


def remove_upload(name: str) -> None:
    """Delete one uploaded PDF and its chunks."""
    safe = Path(name).name
    if safe == _SAMPLE_NAME or not safe.lower().endswith(".pdf"):
        raise ValueError("That document is not in the uploaded PDF library.")
    _delete_source(safe)
    path = config.UPLOAD_DIR / safe
    if path.is_file():
        path.unlink()
    _save_manifest([row for row in _load_manifest() if row.get("name") != safe])


def clear_uploads() -> None:
    """Remove saved PDFs and the upload vector collection."""
    client = get_client(config.UPLOAD_CHROMA_DIR)
    names = {str(n) for n in client.list_collections()}
    if config.UPLOAD_COLLECTION in names:
        client.delete_collection(config.UPLOAD_COLLECTION)
    if config.UPLOAD_DIR.is_dir():
        for path in config.UPLOAD_DIR.iterdir():
            if path.is_file():
                path.unlink()


def _hit_key(hit: dict) -> tuple:
    return (hit.get("source"), hit.get("chunk_id"), hit.get("page"))


def _retrieve_selected(question: str, collection, chosen: list[str]) -> list[dict]:
    """Keep the best chunk from each selected PDF, then fill the rest by distance."""
    vector = embed_text(question)

    def hits_for(where: dict, top_k: int) -> list[dict]:
        return [
            hit
            for hit in retrieve(
                question,
                collection=collection,
                top_k=top_k,
                where=where,
                query_embedding=vector,
            )
            if hit.get("source") in chosen
        ]

    if len(chosen) == 1:
        return hits_for({"source": chosen[0]}, config.TOP_K)

    guaranteed = []
    seen = set()
    for name in chosen:
        for hit in hits_for({"source": name}, 1):
            key = _hit_key(hit)
            if key in seen:
                continue
            seen.add(key)
            guaranteed.append(hit)
    guaranteed.sort(key=lambda hit: hit["distance"])
    guaranteed = guaranteed[: config.TOP_K]
    seen = {_hit_key(hit) for hit in guaranteed}

    extras = []
    if len(guaranteed) < config.TOP_K:
        where = {"source": chosen[0]} if len(chosen) == 1 else {"source": {"$in": chosen}}
        for hit in hits_for(where, config.TOP_K):
            key = _hit_key(hit)
            if key in seen:
                continue
            seen.add(key)
            extras.append(hit)
            if len(guaranteed) + len(extras) >= config.TOP_K:
                break
    return guaranteed + extras


_FOLLOW_UP = re.compile(
    r"\b(it|its|that|this|they|them|their|those|he|she|her|his)\b",
    re.IGNORECASE,
)
_HISTORY_LIMIT = 4
_HISTORY_CHARS = 500


def normalize_history(history: list | None) -> list[dict]:
    """Keep the latest user and assistant turns. Drop anything else."""
    turns = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = str(item.get("text") or item.get("content") or "").strip()
        if role not in ("user", "assistant") or not text:
            continue
        turns.append({"role": role, "content": text[:_HISTORY_CHARS]})
    return turns[-_HISTORY_LIMIT:]


def retrieval_question(question: str, history: list | None = None) -> str:
    """Expand a short or pronoun follow-up with the previous user question."""
    turns = normalize_history(history)
    previous = next((turn["content"] for turn in reversed(turns) if turn["role"] == "user"), "")
    if not previous:
        return question
    if len(question.split()) <= 12 or _FOLLOW_UP.search(question):
        return f"{previous}\n{question}"
    return question


def prepare_document_answer(
    question: str,
    selected: list[str] | None = None,
    history: list | None = None,
) -> dict:
    """Retrieve only the PDFs the user selected."""
    purge_sample_note()
    question = (question or "").strip()
    if not question:
        return {"messages": None, "answer": "Please enter a question.", "hits": [], "gated": True}

    ready = {item["name"] for item in list_uploaded_files()}
    chosen = []
    for name in selected or []:
        safe = Path(str(name)).name
        if safe in ready and safe != _SAMPLE_NAME and safe not in chosen:
            chosen.append(safe)
    if not chosen:
        return {
            "messages": None,
            "answer": "Select at least one indexed PDF before asking a question.",
            "hits": [],
            "gated": True,
        }

    collection = _upload_collection()
    turns = normalize_history(history)
    hits = _retrieve_selected(retrieval_question(question, turns), collection, chosen)
    if not hits:
        return {
            "messages": None,
            "answer": "The selected PDFs have no indexed text to search.",
            "hits": [],
            "gated": True,
        }

    blocks = []
    for i, hit in enumerate(hits[: config.TOP_K], start=1):
        text = _truncate(hit["text"], _MAX_CHARS)
        page = hit.get("page")
        where = f"{hit['source']} p.{page}" if page else hit["source"]
        blocks.append(f"[{i}] ({where})\n{text}")
    context = "\n\n".join(blocks)
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using only the selected PDF excerpts in CONTEXT. "
                "Cite the source file name and page number. "
                "Earlier messages only explain what a follow-up is asking. "
                "If CONTEXT does not contain the answer, say it is not in the selected documents. "
                "Do not use outside knowledge and do not invent facts."
            ),
        },
        *turns,
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
        },
    ]
    return {"messages": messages, "answer": None, "hits": hits[: config.TOP_K], "gated": False}


def source_list(hits: list[dict]) -> list[dict]:
    """Filename, page, and the excerpt that was retrieved for the answer."""
    sources = []
    seen = set()
    for hit in hits:
        text = _truncate(hit.get("text") or "", _MAX_CHARS)
        key = (hit.get("source"), hit.get("page"), text)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"name": hit.get("source"), "page": hit.get("page"), "text": text})
    return sources

