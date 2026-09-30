"""
Stage 0, step 4: give an LLM function-calling access to the three tools.

This uses a local, open-weight model served by Ollama (see README for setup)
rather than OpenAI/Claude, so there's no API key or billing required to run
this. Stage 4 makes the provider swappable (Ollama vs. a self-hosted vLLM
endpoint) via app/llm.py's get_chat_llm() — see that file for details.

The loop below is intentionally simple (no LangGraph yet — that's Stage 1):
  1. send the user's message + tool definitions to the LLM
  2. if the LLM asks to call a tool, run it and feed the result back
  3. repeat until the LLM answers directly, or MAX_TOOL_ITERATIONS is hit
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app import config
from app.llm import get_chat_llm
from app.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are an internal HR assistant for Acme Corp.

You have three tools:
- search_policies: for questions about company policy/rules (remote work, PTO, expenses, code of conduct, security).
- search_employee: for questions about a specific person (their manager, department, title, email).
- list_documents: to see what policy documents exist.

For every question about company policy, rules, procedures, or a specific
person, you MUST call the relevant tool before answering — even if you
think you already know the answer. Never answer a factual question from
memory alone. Do not skip the tool call because the question sounds
generic or you're unsure a document covers it — call search_policies (or
list_documents first, if unsure what's covered) and let the result decide.
The only messages you may answer without calling a tool are ones with no
factual content to look up, such as greetings or thanks.

If a tool returns nothing relevant, say plainly that you don't have that
information — do not guess or make something up. Keep answers concise and
cite which document a policy answer came from when relevant.
"""

_TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def _build_llm():
    # Stage 4: goes through the swappable-provider factory (Ollama by
    # default, vLLM if LLM_PROVIDER=vllm) instead of building ChatOllama
    # directly, so a provider swap applies here automatically.
    return get_chat_llm(temperature=0).bind_tools(ALL_TOOLS)


def run_agent(user_message: str) -> dict:
    """Run the tool-calling loop for a single user message.

    Returns a dict: {"answer": str, "tool_calls": [ {name, args, result}, ... ]}
    so the FastAPI layer (and Streamlit UI) can show which tools fired —
    useful for the Stage 0 acceptance test ("does the right question hit the
    right tool?").
    """
    llm_with_tools = _build_llm()
    messages: list = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    tool_call_log: list[dict] = []

    for _ in range(config.MAX_TOOL_ITERATIONS):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return {"answer": response.content, "tool_calls": tool_call_log}

        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args", {})
            tool_fn = _TOOLS_BY_NAME.get(tool_name)

            if tool_fn is None:
                result = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    result = tool_fn.invoke(tool_args)
                except Exception as exc:  # keep the loop alive on a bad tool call
                    result = f"Error running {tool_name}: {exc}"

            tool_call_log.append({"name": tool_name, "args": tool_args, "result": result})
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    # Ran out of iterations without a final answer.
    return {
        "answer": (
            "I wasn't able to reach a final answer within the allowed number of "
            "tool calls. Please try rephrasing your question."
        ),
        "tool_calls": tool_call_log,
    }
