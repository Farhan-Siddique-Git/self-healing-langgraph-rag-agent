"""
Stage 0, step 3: the three tools, as plain Python functions.

Each is wrapped with LangChain's @tool decorator so it can be handed to the
LLM for function-calling — but the underlying logic is ordinary Python you
could call and test directly with no LLM involved, e.g.:

    >>> from app.tools import search_employee
    >>> search_employee.invoke({"name": "Alice"})

Keeping the tool bodies as plain functions (rather than baking retrieval
logic into prompts) is what makes Stage 1's self-healing loop possible later:
the graph in Stage 1 wraps *around* these same three functions unchanged.
"""
from __future__ import annotations

import glob
import os

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

from app import config
from app.db import get_connection

_qdrant_client: QdrantClient | None = None
_embeddings: OllamaEmbeddings | None = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=config.QDRANT_PATH)
    return _qdrant_client


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL, base_url=config.OLLAMA_BASE_URL
        )
    return _embeddings


@tool
def search_policies(query: str, k: int = 4) -> str:
    """Search the company's HR policy documents (remote work, PTO, expenses,
    code of conduct, security) for text relevant to the query. Use this for
    any question about company policy, rules, or procedures — e.g. "can
    interns work remotely?" or "how many PTO days do I get?". Returns the
    top matching passages along with which document they came from."""
    client = _get_qdrant_client()
    embeddings = _get_embeddings()

    query_vector = embeddings.embed_query(query)
    hits = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=k,
    ).points

    if not hits:
        return "No relevant policy passages were found for this query."

    formatted = []
    for hit in hits:
        payload = hit.payload or {}
        source = payload.get("source", "unknown document")
        text = payload.get("text", "")
        formatted.append(f"[source: {source}, score: {hit.score:.3f}]\n{text}")

    return "\n\n---\n\n".join(formatted)


@tool
def search_employee(name: str) -> str:
    """Look up an employee by name (full or partial match) in the company
    directory. Use this for any question about a specific person — e.g. who
    manages them, what department or title they have, or their email. Not
    for questions about policy or rules."""
    con = get_connection()
    rows = con.execute(
        """
        SELECT id, name, department, title, manager, email, location, employment_type
        FROM employees
        WHERE lower(name) LIKE lower(?)
        """,
        [f"%{name}%"],
    ).fetchall()

    if not rows:
        return f"No employee found matching '{name}'."

    columns = ["id", "name", "department", "title", "manager", "email", "location", "employment_type"]
    results = []
    for row in rows:
        record = dict(zip(columns, row))
        results.append(
            f"{record['name']} — {record['title']}, {record['department']}. "
            f"Manager: {record['manager'] or 'none'}. Email: {record['email']}. "
            f"Location: {record['location']}. Type: {record['employment_type']}."
        )
    return "\n".join(results)


@tool
def list_documents() -> str:
    """List every policy document currently available to search_policies.
    Use this if the user asks what documents/policies exist, or to check
    whether a topic is covered before claiming there's no information on
    it."""
    pdf_paths = sorted(glob.glob(os.path.join(config.POLICIES_DIR, "*.pdf")))
    if not pdf_paths:
        return "No policy documents are currently loaded."
    names = [os.path.basename(p) for p in pdf_paths]
    return "Available policy documents:\n" + "\n".join(f"- {n}" for n in names)


ALL_TOOLS = [search_policies, search_employee, list_documents]

# --- Stage 4 (optional): knowledge-graph tool --------------------------------
# Only added if NEO4J_ENABLED=true AND its dependencies import cleanly AND
# the Neo4j connection actually succeeds (see app/graph_tool.py). If any of
# that fails, ALL_TOOLS just stays as the three tools above — Stage 0-3
# behave identically whether or not this block does anything.
if config.NEO4J_ENABLED:
    try:
        from app.graph_tool import search_relationships

        ALL_TOOLS.append(search_relationships)
    except Exception as exc:  # missing deps, bad config, etc.
        print(f"[tools] Stage 4 knowledge-graph tool not loaded: {exc}")
