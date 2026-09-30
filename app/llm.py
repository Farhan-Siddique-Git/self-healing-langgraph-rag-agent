"""
Stage 4: a single factory for the chat LLM, so swapping providers (local
Ollama vs. a self-hosted, OpenAI-compatible vLLM endpoint serving a bigger
open-weight model) is a one-line config change instead of touching every
file that builds a chat model directly.

Set LLM_PROVIDER=vllm (plus VLLM_BASE_URL / VLLM_MODEL) in .env to switch.
Both app/agent.py (Stage 0) and app/graph.py (Stage 1) call get_chat_llm()
instead of constructing a chat model themselves, so the switch applies
everywhere at once, with no other code changes.

NOTE: the vLLM path is written against vLLM's OpenAI-compatible server API
(https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) but
was not run against a live vLLM deployment in this project — no GPU was
available during development, and the plan always treated this as an
optional swap for later. Wire it up and sanity-check it against your own
endpoint before relying on it for anything beyond a demo.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama

from app import config


def get_chat_llm(temperature: float = 0.0):
    """Return a chat model configured per LLM_PROVIDER. Every call site in
    the app should go through this function rather than importing
    ChatOllama/ChatOpenAI directly, so provider swaps stay one config
    change."""
    if config.LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=config.OLLAMA_CHAT_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    if config.LLM_PROVIDER == "vllm":
        # vLLM's OpenAI-compatible server works with the standard ChatOpenAI
        # client, just pointed at a different base_url. api_key is required
        # by the client's constructor but unused by vLLM unless you've
        # configured it to check one.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.VLLM_MODEL,
            base_url=config.VLLM_BASE_URL,
            api_key=config.VLLM_API_KEY or "not-needed",
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER {config.LLM_PROVIDER!r}. Expected 'ollama' or 'vllm'."
    )
