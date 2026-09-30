"""
Stage 4 (optional): one-time script to extract a knowledge graph from the
policy PDFs and load it into Neo4j, using LangChain's LLMGraphTransformer.

Requires NEO4J_ENABLED=true and NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD set
in .env, plus:
    pip install langchain-neo4j langchain-experimental neo4j

Run once (or whenever the policy documents change):
    python scripts/ingest_graph.py

NOTE: entity/relationship extraction quality depends heavily on the LLM.
qwen2.5:7b was reliable for Stage 0/1's tool-calling, but hasn't been
validated here for structured graph extraction specifically — this script
was written but not run against a live Neo4j instance (none was available
in the sandboxed dev environment). After running it, inspect the resulting
graph in the Neo4j browser and, if the extraction looks noisy, consider
passing allowed_nodes / allowed_relationships to LLMGraphTransformer below
to constrain what it extracts.
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config


def main() -> None:
    if not config.NEO4J_ENABLED:
        print(
            "NEO4J_ENABLED is not set to true in your .env — set it first, "
            "along with NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD."
        )
        sys.exit(1)

    try:
        import fitz  # PyMuPDF
        from langchain_core.documents import Document
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        from langchain_neo4j import Neo4jGraph
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        print(
            f"Missing dependency: {exc}. Run: "
            "pip install langchain-neo4j langchain-experimental neo4j"
        )
        sys.exit(1)

    from app.llm import get_chat_llm

    pdf_paths = sorted(glob.glob(os.path.join(config.POLICIES_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {config.POLICIES_DIR}")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    documents: list[Document] = []
    for path in pdf_paths:
        with fitz.open(path) as pdf:
            text = "".join(page.get_text() for page in pdf)
        for chunk in splitter.split_text(text):
            documents.append(
                Document(page_content=chunk, metadata={"source": os.path.basename(path)})
            )

    print(
        f"Extracted {len(documents)} chunks from {len(pdf_paths)} PDF(s). "
        "Running graph extraction (one LLM call per chunk — can take a "
        "while on a local model)..."
    )

    llm = get_chat_llm(temperature=0)
    transformer = LLMGraphTransformer(llm=llm)
    graph_documents = transformer.convert_to_graph_documents(documents)

    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
    )
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    total_nodes = sum(len(gd.nodes) for gd in graph_documents)
    total_rels = sum(len(gd.relationships) for gd in graph_documents)
    print(f"Loaded {total_nodes} nodes and {total_rels} relationships into Neo4j.")


if __name__ == "__main__":
    main()
