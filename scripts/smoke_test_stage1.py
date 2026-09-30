"""
Stage 1 smoke test — same spirit as scripts/smoke_test.py from Stage 0, but
against the self-healing graph. Run with:

    python scripts/smoke_test_stage1.py

Requires `ollama serve` running and the documents already ingested
(python -c "from app.ingest import run_ingestion; run_ingestion()").
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph import run_self_healing_agent

CASES = [
    {
        "question": "Can interns work remotely?",
        "expected_tool": "search_policies",
    },
    {
        "question": "Who is Alice's manager?",
        "expected_tool": "search_employee",
    },
]

UNANSWERABLE_QUESTION = "Is it okay if I walk my neighbor's dog during my lunch break?"
UNANSWERABLE_PHRASE = "don't have"


def main() -> int:
    failures = 0

    for case in CASES:
        print(f"--- {case['question']!r} ---")
        result = run_self_healing_agent(case["question"])
        called = {c["name"] for c in result["tool_calls"]}
        print(f"answer: {result['answer']}")
        print(f"tools called: {called}")
        print(f"grounded: {result['grounded']}, retries: {result['retries']}")

        if case["expected_tool"] not in called:
            print(f"FAIL: expected tool {case['expected_tool']!r} to be called")
            failures += 1
        elif not result["grounded"]:
            print("FAIL: expected a grounded answer")
            failures += 1
        else:
            print("PASS")
        print()

    print(f"--- {UNANSWERABLE_QUESTION!r} ---")
    result = run_self_healing_agent(UNANSWERABLE_QUESTION)
    print(f"answer: {result['answer']}")
    print(f"grounded: {result['grounded']}, retries: {result['retries']}")
    if UNANSWERABLE_PHRASE not in result["answer"].lower():
        print(
            f"WARN: expected the answer to contain {UNANSWERABLE_PHRASE!r} — the model "
            "may have phrased the decline differently. Read the answer above and judge "
            "for yourself; this is a WARN, not a FAIL."
        )
    else:
        print("PASS")
    print()

    if failures:
        print(f"{failures} case(s) FAILED.")
        return 1

    print("All required cases PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
