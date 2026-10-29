"""Paths and local model settings. Edit this file to change defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_CHROMA_DIR = PROJECT_ROOT / ".chroma_uploads"
UPLOAD_COLLECTION = "user_uploads"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
# Used only when the vLLM server cannot be reached.
CHAT_MODEL = "llama3.2:3b"
VLLM_URL = "http://127.0.0.1:8000/v1"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

NUM_PREDICT = 512
TEMPERATURE = 0.2
OLLAMA_KEEP_ALIVE = "30m"
EMBED_TIMEOUT_S = 120
CHAT_TIMEOUT_S = 600
TOP_K = 6
