"""
Stage 4 (optional): a knowledge-graph lookup tool backed by Neo4j.

Disabled by default (config.NEO4J_ENABLED = False), so the rest of the app
has zero Neo4j dependency unless you opt in. To enable:

  1. Get a Neo4j instance — Neo4j Aura's free tier works and needs no local
     install: https://neo4j.com/cloud/platform/aura-graph-database/
  2. Set NEO4J_ENABLED=true and NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD in .env.
  3. pip install langchain-neo4j langchain-experimental neo4j
  4. Run `python scripts/ingest_graph.py` once to extract entities/relationships
     from the policy PDFs (via LangChain's LLMGraphTransformer) and load them
     into Neo4j.
  5. Restart the app — app/tools.py appends search_relationships to ALL_TOOLS
     automatically once NEO4J_ENABLED is true and the import succeeds.

NOTE: this was not run against a live Neo4j instance during this project (no
hosted graph DB was available in the sandboxed dev environment) — treat it
as a solid starting point to wire up and validate against your own
instance, not as pre-verified code. It's written defensively (every failure
mode returns a plain-text message instead of raising) specifically so a
misconfigured or unreachable Neo4j never takes down the rest of the app.
"""
from __future__ import annotations

from langchain_core.tools import tool

from app import config

_graph_client = None
_graph_client_failed = False


def _get_graph_client():
    """Lazily connect to Neo4j, and remember a failed connection so we don't
    retry (and re-raise) on every single tool call."""
    global _graph_client, _graph_client_failed
    if _graph_client is not None or _graph_client_failed:
        return _graph_client
    try:
        from langchain_neo4j import Neo4jGraph

        _graph_client = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
        )
    except Exception:
        _graph_client_failed = True
        _graph_client = None
    return _graph_client


@tool
def search_relationships(entity: str) -> str:
    """Look up how a person, team, or policy topic relates to other entities
    in the knowledge graph — e.g. which department a policy applies to, or
    what topics are connected to a given person or team. Use this for
    relationship/connection questions the other tools can't answer with a
    plain text search. Only meaningful once the Stage 4 knowledge graph has
    been populated via scripts/ingest_graph.py."""
    graph = _get_graph_client()
    if graph is None:
        return "The knowledge graph is not available (Neo4j not configured or unreachable)."

    cypher = """
    MATCH (n)-[r]-(m)
    WHERE toLower(n.id) CONTAINS toLower($term)
    RETURN n.id AS from_entity, type(r) AS relationship, m.id AS to_entity
    LIMIT 15
    """
    try:
        results = graph.query(cypher, params={"term": entity})
    except Exception as exc:  # unreachable server, bad query, etc.
        return f"Error querying the knowledge graph: {exc}"

    if not results:
        return f"No relationships found for '{entity}' in the knowledge graph."

    lines = [
        f"{row['from_entity']} --{row['relationship']}--> {row['to_entity']}"
        for row in results
    ]
    return "\n".join(lines)
