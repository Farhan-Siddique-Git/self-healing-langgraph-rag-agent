"""
Stage 1: a LangGraph self-healing retrieve -> generate -> critique -> retry
loop, built on top of the exact same three tools from app/tools.py.

Why this exists: Stage 0's agent.py trusts the LLM's own answer completely.
During testing, qwen2.5:1.5b reliably skipped tool calls on real questions
(PTO and "can I walk my neighbor's dog on lunch" both got generic,
ungrounded answers with tool_calls: []) — switching the default model to
qwen2.5:7b fixed that specific failure, but nothing stops a *future* model
swap, prompt regression, or edge-case question from reintroducing it. Stage 1
adds a second LLM call — a critique step — that checks whether the generated
answer is actually supported by the retrieved context, and retries with a
reformulated query if not, instead of just trusting the first pass.

Graph shape:

    START -> retrieve -> generate -> critique -> decide -+-> END (answer is grounded)
                ^                                         |
                +---------- (retry: reformulate query) <--+
                                                            |
                                                            +-> END (gave up after max_retries)

Run directly:  python -c "from app.graph import run_self_healing_agent as r; print(r('Can interns work remotely?'))"
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import SystemMessage

from app import config
from app.llm import get_chat_llm
from app.tools import ALL_TOOLS

_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


class AgentState(TypedDict):
    question: str  # the user's original question, never overwritten
    current_query: str  # the query actually sent to retrieval this round (may be reformulated)
    context: str  # raw tool output gathered this round
    answer: str  # the generated answer this round
    grounded: bool | None  # critique's verdict; None until critique has run
    give_up: bool  # set once max_retries is exhausted without a grounded answer
    retries: int  # how many retrieve/generate/critique rounds have run so far
    max_retries: int
    tool_calls: list[dict]  # accumulated across all rounds, same shape as Stage 0's log


RETRIEVE_SYSTEM_PROMPT = """You are the retrieval step of an internal HR assistant for Acme Corp.

You have three tools:
- search_policies: for questions about company policy/rules (remote work, PTO, expenses, code of conduct, security).
- search_employee: for questions about a specific person (their manager, department, title, email).
- list_documents: to see what policy documents exist.

Your ONLY job this turn is to call the single most relevant tool for the
query below and return its raw result. Always call exactly one tool — never
answer from memory, and never skip the tool call. If you are unsure which
tool applies, prefer search_policies.

Query: {query}
"""

GENERATE_SYSTEM_PROMPT = """You are an internal HR assistant for Acme Corp. Answer the user's
question using ONLY the context below — do not use outside knowledge and do
not guess. If the context does not contain enough information to answer,
say plainly that you don't have enough information on that, rather than
making something up. Keep the answer concise and cite which document it
came from when relevant.

Context:
{context}

Question: {question}
"""

CRITIQUE_SYSTEM_PROMPT = """You are a strict fact-checker. Given a context and an answer that was
supposed to be derived from it, decide whether the answer is actually
grounded in the context — i.e. every factual claim in the answer is
supported by something in the context, and the answer isn't a generic
non-answer to a question the context *does* actually cover.

An answer that honestly says "I don't have enough information" when the
context truly doesn't cover the question counts as GROUNDED (that's the
correct behavior, not a failure).

Respond with exactly one word: GROUNDED or UNGROUNDED.

Context:
{context}

Answer to check:
{answer}
"""

REFORMULATE_PROMPT = """The previous search for the query below did not return context that
supports a grounded answer. Rewrite the query to be more specific and more
likely to match relevant HR policy or employee-directory text. Return ONLY
the rewritten query, nothing else.

Original question: {question}
Previous query tried: {previous_query}
"""


def _llm(temperature: float = 0.0):
    # Stage 4: goes through the swappable-provider factory (Ollama by
    # default, vLLM if LLM_PROVIDER=vllm) instead of building ChatOllama
    # directly, so a provider swap applies here automatically.
    return get_chat_llm(temperature=temperature)


def retrieve_node(state: AgentState) -> dict:
    """Call the LLM with tool-binding restricted to a single required tool
    call, then execute whatever it picks. Logs into tool_calls the same way
    Stage 0's agent.py does, so the eval harness and Streamlit UI can render
    both stages identically."""
    query = state["current_query"]
    llm_with_tools = _llm().bind_tools(ALL_TOOLS)

    response = llm_with_tools.invoke(
        [SystemMessage(content=RETRIEVE_SYSTEM_PROMPT.format(query=query))]
    )
    tool_calls = getattr(response, "tool_calls", None) or []

    if not tool_calls:
        # The model answered without calling a tool despite the instruction —
        # treat this as "no context found" rather than trusting it, which is
        # exactly the failure mode Stage 1 exists to catch.
        return {
            "context": "No tool was called; no context was retrieved.",
            "tool_calls": state["tool_calls"],
        }

    call = tool_calls[0]
    tool_name = call["name"]
    tool_args = call.get("args", {})
    tool_fn = _TOOLS_BY_NAME.get(tool_name)

    if tool_fn is None:
        result = f"Error: unknown tool '{tool_name}'"
    else:
        try:
            result = tool_fn.invoke(tool_args)
        except Exception as exc:  # keep the graph alive on a bad tool call
            result = f"Error running {tool_name}: {exc}"

    new_log = state["tool_calls"] + [{"name": tool_name, "args": tool_args, "result": result}]
    return {"context": str(result), "tool_calls": new_log}


def generate_node(state: AgentState) -> dict:
    llm = _llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=GENERATE_SYSTEM_PROMPT.format(
                    context=state["context"], question=state["question"]
                )
            )
        ]
    )
    return {"answer": response.content}


def critique_node(state: AgentState) -> dict:
    llm = _llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=CRITIQUE_SYSTEM_PROMPT.format(
                    context=state["context"], answer=state["answer"]
                )
            )
        ]
    )
    verdict = str(response.content).strip().upper()
    grounded = verdict.startswith("GROUNDED")
    return {"grounded": grounded}


def decide_node(state: AgentState) -> dict:
    """Not an LLM call — just decides whether to retry, give up, or finish,
    and reformulates the query when retrying. Kept as its own node (rather
    than folded into the conditional edge) so the reformulation LLM call is
    visible as a distinct graph step."""
    if state["grounded"]:
        return {}

    if state["retries"] >= state["max_retries"]:
        return {"give_up": True}

    llm = _llm(temperature=0.3)
    response = llm.invoke(
        [
            SystemMessage(
                content=REFORMULATE_PROMPT.format(
                    question=state["question"], previous_query=state["current_query"]
                )
            )
        ]
    )
    new_query = str(response.content).strip() or state["question"]
    return {"current_query": new_query, "retries": state["retries"] + 1}


def _route_after_decide(state: AgentState) -> str:
    if state["grounded"] or state.get("give_up"):
        return "end"
    return "retry"


def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique", critique_node)
    graph.add_node("decide", decide_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "decide")
    graph.add_conditional_edges("decide", _route_after_decide, {"retry": "retrieve", "end": END})

    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_self_healing_agent(question: str, max_retries: int = 2) -> dict:
    """Entry point mirroring Stage 0's run_agent(), so app/main.py can call
    either one and return the same shape.

    Returns: {"answer": str, "tool_calls": [...], "retries": int, "grounded": bool}
    """
    graph = get_graph()
    initial_state: AgentState = {
        "question": question,
        "current_query": question,
        "context": "",
        "answer": "",
        "grounded": None,
        "give_up": False,
        "retries": 0,
        "max_retries": max_retries,
        "tool_calls": [],
    }
    final_state = graph.invoke(initial_state)

    answer = final_state["answer"]
    if final_state.get("give_up") and not final_state.get("grounded"):
        answer = (
            f"{answer}\n\n(Note: I retried this search {final_state['retries']} time(s) and "
            "still couldn't fully confirm this answer against the source documents — please "
            "double-check with HR.)"
        )

    return {
        "answer": answer,
        "tool_calls": final_state["tool_calls"],
        "retries": final_state["retries"],
        "grounded": final_state.get("grounded"),
    }
