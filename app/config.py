"""
Central configuration for the Policy Agent.

Everything is overridable via environment variables (see .env.example),
so nothing here is hardcoded that a real deployment would need to change.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Ollama (local LLM + embeddings) ---------------------------------------
# qwen2.5:7b is the default because it was tested and reliably calls the
# right tool for every factual question. qwen2.5:1.5b is faster but, in
# practice, skips tool calls and answers generically often enough to be
# unreliable for Stage 0's "never guess" requirement — drop down to it only
# if you're CPU-constrained and can tolerate that tradeoff.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- Qdrant (local, on-disk — no Docker/server required) -------------------
QDRANT_PATH = os.getenv("QDRANT_PATH", str(BASE_DIR / "qdrant_data"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "policies")

# --- Data sources ------------------------------------------------------------
POLICIES_DIR = os.getenv("POLICIES_DIR", str(BASE_DIR / "data" / "policies"))
EMPLOYEES_CSV = os.getenv("EMPLOYEES_CSV", str(BASE_DIR / "data" / "employees.csv"))

# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Agent loop ---------------------------------------------------------------
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "5"))

# --- FastAPI / Streamlit -------------------------------------------------------
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Stage 4 (optional): swappable LLM provider ------------------------------
# "ollama" (default) or "vllm" — see app/llm.py for the factory that reads
# this. Switching providers is just changing this + the VLLM_* vars below;
# no code changes needed anywhere that calls get_chat_llm().
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-30B-A3B")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")

# --- Stage 4 (optional): Neo4j knowledge graph -------------------------------
# Disabled by default — the app works fully without it. Enable only once you
# have a real Neo4j instance (e.g. Neo4j Aura's free tier) and have run
# scripts/ingest_graph.py to populate it. See app/graph_tool.py.
NEO4J_ENABLED = os.getenv("NEO4J_ENABLED", "false").lower() == "true"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
