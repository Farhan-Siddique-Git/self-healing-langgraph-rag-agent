"""
Stage 0's own acceptance test, straight from the build plan:

  "Can interns work remotely?" should hit search_policies
  "Who manages Alice?" should hit search_employee

Run this after ingestion, with Ollama running:
    python scripts/smoke_test.py

It doesn't need the FastAPI server running — it calls the agent directly.
"""
import sys

sys.path.insert(0, ".")

from app.agent import run_agent  # noqa: E402

TEST_CASES = [
    ("Can interns work remotely?", "search_policies"),
    ("Who manages Alice?", "search_employee"),
]


def main() -> None:
    failures = 0

    for question, expected_tool in TEST_CASES:
        print(f"\n{'=' * 70}\nQ: {question}")
        result = run_agent(question)
        tools_used = [c["name"] for c in result["tool_calls"]]

        print(f"A: {result['answer']}")
        print(f"Tools used: {tools_used}")

        if expected_tool in tools_used:
            print(f"PASS — used {expected_tool} as expected")
        else:
            print(f"FAIL — expected {expected_tool}, got {tools_used}")
            failures += 1

    print(f"\n{'=' * 70}")
    if failures:
        print(f"{failures} test(s) failed.")
        sys.exit(1)
    else:
        print("All smoke tests passed.")


if __name__ == "__main__":
    main()
