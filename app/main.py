"""
FastAPI server. Stage 0's simple tool-calling loop and Stage 1's
self-healing graph are both wired in here, selected per-request via
`stage`, with Stage 2's input/output guardrails applied around whichever
one runs.

Run directly:      uvicorn app.main:app --reload --port 8000
Or:                 python -m app.main
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app import config
from app.agent import run_agent
from app.graph import run_self_healing_agent
from app.guardrails import check_input, check_output

app = FastAPI(title="Policy Agent")


class ChatRequest(BaseModel):
    message: str
    stage: int = 1  # 0 = Stage 0 simple loop, 1 = Stage 1 self-healing graph


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    result: str


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord]
    stage: int
    retries: int | None = None  # Stage 1 only
    grounded: bool | None = None  # Stage 1 only
    blocked: bool = False
    blocked_reason: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # --- Stage 2: input guardrail, checked before any LLM call ---
    input_check = check_input(request.message)
    if not input_check.allowed:
        return ChatResponse(
            answer=input_check.reason or "This request can't be processed.",
            tool_calls=[],
            stage=request.stage,
            blocked=True,
            blocked_reason=input_check.rule,
        )

    # --- Route to the requested stage's agent ---
    if request.stage == 0:
        result = run_agent(request.message)
        answer = result["answer"]
        tool_calls = result["tool_calls"]
        retries = None
        grounded = None
    else:
        result = run_self_healing_agent(request.message)
        answer = result["answer"]
        tool_calls = result["tool_calls"]
        retries = result["retries"]
        grounded = result["grounded"]

    # --- Stage 2: output guardrail, can override the generated answer ---
    output_check = check_output(answer)
    blocked = False
    blocked_reason = None
    if not output_check.allowed:
        answer = output_check.reason or "This answer can't be shared."
        blocked = True
        blocked_reason = output_check.rule

    return ChatResponse(
        answer=answer,
        tool_calls=tool_calls,
        stage=request.stage,
        retries=retries,
        grounded=grounded,
        blocked=blocked,
        blocked_reason=blocked_reason,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
