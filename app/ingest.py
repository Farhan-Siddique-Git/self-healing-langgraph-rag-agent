"""
Stage 0, step 1-2: parse the sample policy PDFs (PyMuPDF), split into chunks,
embed them (via a local Ollama embedding model), and load into a local
on-disk Qdrant collection.

Qdrant runs here in "local mode" (qdrant_client.QdrantClient(path=...)) —
this persists to a folder on disk with no Docker container or server needed,
which keeps Stage 0 dependency-free. Swap to a real Qdrant server later by
changing QdrantClient(path=...) to QdrantClient(url=...); nothing else in
the app needs to change.

Run:  python -m app.ingest
"""
from __future__ import annotations

import glob
import os
import uuid

import fitz  # PyMuPDF
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config


def load_pdf_text(pdf_path: str) -> str:
    """Extract plain text from every page of a PDF using PyMuPDF."""
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def chunk_text(text: str, source: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    return [{"text": c, "source": source, "chunk_index": i} for i, c in enumerate(chunks)]


def get_qdrant_client() -> QdrantClient:
    os.makedirs(config.QDRANT_PATH, exist_ok=True)
    return QdrantClient(path=config.QDRANT_PATH)


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION in existing:
        client.delete_collection(config.QDRANT_COLLECTION)
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=vector_size, distance=qmodels.Distance.COSINE
        ),
    )


def run_ingestion() -> None:
    pdf_paths = sorted(glob.glob(os.path.join(config.POLICIES_DIR, "*.pdf")))
    if not pdf_paths:
        raise SystemExit(
            f"No PDFs found in {config.POLICIES_DIR}. "
            "Run scripts/generate_sample_data.py first, or drop your own PDFs there."
        )

    print(f"Found {len(pdf_paths)} policy PDF(s):")
    for p in pdf_paths:
        print(f"  - {os.path.basename(p)}")

    all_chunks: list[dict] = []
    for path in pdf_paths:
        text = load_pdf_text(path)
        chunks = chunk_text(text, source=os.path.basename(path))
        print(f"  {os.path.basename(path)}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Embedding with Ollama model '{config.OLLAMA_EMBED_MODEL}' "
          f"(this calls your local Ollama server at {config.OLLAMA_BASE_URL}) ...")

    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    vectors = embeddings.embed_documents([c["text"] for c in all_chunks])
    vector_size = len(vectors[0])

    client = get_qdrant_client()
    ensure_collection(client, vector_size=vector_size)

    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk, vec in zip(all_chunks, vectors)
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)

    print(
        f"\nIngested {len(points)} chunks from {len(pdf_paths)} documents into "
        f"Qdrant collection '{config.QDRANT_COLLECTION}' at {config.QDRANT_PATH}"
    )


if __name__ == "__main__":
    run_ingestion()
