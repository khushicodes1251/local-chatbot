#!/usr/bin/env python3
"""Flask app for the local PDF chatbot.

Serves the HTML page and indexes only the PDFs the user uploads.
Retrieval, embeddings, and answers are handled by rag_chatbot.
"""

from __future__ import annotations

import json

from flask import Flask, Response, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from rag_chatbot.ollama_client import chat_stream
from rag_chatbot.uploads import (
    clear_uploads,
    index_upload,
    list_uploaded_files,
    prepare_document_answer,
    purge_sample_note,
    remove_upload,
    source_list,
)
from rag_chatbot.vllm_client import chat_stream_vllm

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
purge_sample_note()


@app.get("/")
def index():
    return render_template("index.html")


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return {"error": "That PDF is larger than 16 MB."}, 413


@app.get("/api/files")
def files():
    return {"files": list_uploaded_files()}


@app.post("/api/upload")
def upload():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return {"error": "Choose a PDF to upload."}, 400
    try:
        result = index_upload(uploaded.filename, uploaded.read())
    except ValueError as exc:
        return {"error": str(exc)}, 400
    except Exception as exc:
        return {"error": f"Could not process that PDF: {exc}"}, 500
    return {"ok": True, **result, "files": list_uploaded_files()}


@app.delete("/api/files/<name>")
def delete_file(name: str):
    try:
        remove_upload(name)
    except ValueError as exc:
        return {"error": str(exc)}, 400
    return {"ok": True, "files": list_uploaded_files()}


@app.post("/api/clear")
def clear():
    clear_uploads()
    return {"ok": True, "files": []}


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return {"error": "Please enter a question."}, 400

    selected = payload.get("documents") or payload.get("selected") or []
    if isinstance(selected, str):
        selected = [selected]
    try:
        history = payload.get("history") or []
        prepared = prepare_document_answer(question, selected, history)
    except Exception as exc:
        message = str(exc)
        if "11434" in message or "Ollama" in message or "Connection" in message:
            message = "Cannot reach the local embedding service. Start Ollama with: ollama serve"

        def failed():
            yield _sse({"error": message, "done": True})

        return Response(failed(), mimetype="text/event-stream")

    if prepared["messages"] is None:
        answer = prepared["answer"] or ""

        def fixed():
            yield _sse({"token": answer, "done": True, "gated": True, "sources": []})

        return Response(fixed(), mimetype="text/event-stream")

    sources = source_list(prepared["hits"])

    def stream():
        try:
            try:
                token_iter = chat_stream_vllm(prepared["messages"])
                first = next(token_iter)
            except StopIteration:
                first = ""
                token_iter = iter(())
            except Exception:
                token_iter = chat_stream(prepared["messages"])
                first = ""
            if first:
                yield _sse({"token": first})
            for piece in token_iter:
                if piece:
                    yield _sse({"token": piece})
            yield _sse({"done": True, "sources": sources})
        except Exception as exc:
            yield _sse({"error": str(exc), "done": True})

    return Response(stream(), mimetype="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
