"""
Streamlit chat UI on top of the FastAPI server. Lets you switch between
Stage 0's simple agent and Stage 1's self-healing graph, and shows Stage 2
guardrail blocks when they happen.

Run:  streamlit run app/streamlit_app.py
(Make sure `uvicorn app.main:app --port 8000` is already running in another
terminal — this UI just calls that API.)
"""
import os
import sys

# When Streamlit launches this file directly (`streamlit run app/streamlit_app.py`),
# it can add this script's own folder to sys.path instead of the project
# root, which would break the `from app import config` import below. Make
# sure the project root (the parent of this file's folder) is on sys.path
# regardless of how Streamlit bootstrapped us.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import streamlit as st

from app import config

API_URL = f"http://{config.API_HOST}:{config.API_PORT}/chat"

st.set_page_config(page_title="Acme Corp HR Assistant", page_icon="💬")
st.title("💬 Acme Corp HR Assistant")
st.caption(
    "Ask about company policy (remote work, PTO, expenses, conduct, security) "
    "or about a colleague (manager, department, title)."
)

with st.sidebar:
    st.subheader("Agent stage")
    stage_label = st.radio(
        "Which agent runs your question?",
        options=["Stage 1 — self-healing (recommended)", "Stage 0 — simple tool-calling"],
        index=0,
    )
    stage = 1 if stage_label.startswith("Stage 1") else 0
    if stage == 1:
        st.caption(
            "Retrieves, generates, then critiques its own answer against the "
            "retrieved context — retries with a reformulated query if the "
            "answer isn't grounded, instead of just trusting the first pass."
        )
    else:
        st.caption("A single retrieve → answer pass, with no self-check.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role", "content", "tool_calls", "meta"}

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        meta = turn.get("meta") or {}
        if meta.get("blocked"):
            st.caption(f"🛡️ Blocked by guardrails ({meta.get('blocked_reason')})")
        elif meta.get("stage") == 1 and meta.get("grounded") is not None:
            if meta["grounded"]:
                st.caption("✅ grounded")
            else:
                st.caption(f"⚠️ gave up after {meta.get('retries', 0)} retry(ies)")
        if turn.get("tool_calls"):
            with st.expander("Tool calls"):
                for call in turn["tool_calls"]:
                    st.markdown(f"**{call['name']}**(`{call['args']}`)")
                    st.text(call["result"])

user_input = st.chat_input("Ask a question, e.g. 'Can interns work remotely?'")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input, "meta": {}})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    API_URL, json={"message": user_input, "stage": stage}, timeout=180
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                tool_calls = data.get("tool_calls", [])
                meta = {
                    "stage": data.get("stage"),
                    "retries": data.get("retries"),
                    "grounded": data.get("grounded"),
                    "blocked": data.get("blocked", False),
                    "blocked_reason": data.get("blocked_reason"),
                }
            except requests.RequestException as exc:
                answer = (
                    f"Couldn't reach the API at {API_URL} ({exc}). "
                    "Is `uvicorn app.main:app --port 8000` running?"
                )
                tool_calls = []
                meta = {}

        st.markdown(answer)
        if meta.get("blocked"):
            st.caption(f"🛡️ Blocked by guardrails ({meta.get('blocked_reason')})")
        elif meta.get("stage") == 1 and meta.get("grounded") is not None:
            if meta["grounded"]:
                st.caption("✅ grounded")
            else:
                st.caption(f"⚠️ gave up after {meta.get('retries', 0)} retry(ies)")
        if tool_calls:
            with st.expander("Tool calls"):
                for call in tool_calls:
                    st.markdown(f"**{call['name']}**(`{call['args']}`)")
                    st.text(call["result"])

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "tool_calls": tool_calls, "meta": meta}
    )
