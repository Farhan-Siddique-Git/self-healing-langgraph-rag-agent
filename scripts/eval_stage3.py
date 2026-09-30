"""
Stage 3 — Golden-dataset evaluation harness.

Runs every case in eval/golden_dataset.json through the Stage 1 self-healing
agent (app.graph.run_self_healing_agent) and scores it on three axes:

  1. Tool accuracy   — did the agent call the tool we expect it to call?
                        (skipped for cases where expected_tool is null, i.e.
                        cases the agent should NOT be able to answer from
                        the knowledge base)
  2. Keyword coverage — a cheap, judge-free proxy for faithfulness/relevancy.
                        Checks whether any/all of a small set of expected
                        keywords or refusal phrases show up in the answer.
                        This is deliberately NOT an LLM-as-judge scorer
                        (see README note at the bottom of this file for why),
                        so it will never be as good as Ragas/human eval, but
                        it needs no extra model calls and is fully
                        deterministic, which is what a CI gate needs.
  3. Latency          — wall-clock time per question, reported so we can spot
                        regressions if the model or prompt changes.

This script is meant to run both locally (`python scripts/eval_stage3.py`)
and inside CI (see .github/workflows/eval.yml), and it exits non-zero when
the aggregate pass rate falls below a threshold, so CI can gate on it.

NOTE ON THE GOLDEN DATASET: the `expected_keywords` in eval/golden_dataset.json
are generic HR-policy terms (pto, remote, expense, etc). They're a reasonable
starting point but were written without seeing the exact wording of the
synthetic policy PDFs your `scripts/generate_sample_data.py` generated on
your machine. After your first real run, open eval_report.json, read the
actual answers, and tighten/loosen expected_keywords to match your real
documents — that calibration step is expected and normal for this kind of
harness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make `app` importable when this script is run directly (python scripts/eval_stage3.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph import run_self_healing_agent  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "eval" / "golden_dataset.json"
DEFAULT_REPORT = Path(__file__).resolve().parent.parent / "eval_report.json"

# Minimum fraction of cases that must pass for this script to exit 0.
# Overridable via the EVAL_PASS_THRESHOLD env var so CI can tune it without
# touching code.
PASS_THRESHOLD = float(os.environ.get("EVAL_PASS_THRESHOLD", "0.75"))


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_case(case: dict, result: dict) -> dict:
    """Score a single case against the agent's result. Returns a dict with
    per-check booleans plus an overall `passed` bool."""
    answer = (result.get("answer") or "").lower()
    tool_calls = result.get("tool_calls") or []
    called_tools = {tc.get("tool") or tc.get("name") for tc in tool_calls if isinstance(tc, dict)}

    expected_tool = case.get("expected_tool")
    expected_keywords = [k.lower() for k in case.get("expected_keywords", [])]

    checks = {}

    # --- Tool accuracy check ---
    if expected_tool is None:
        checks["tool_check"] = None  # not applicable
        tool_ok = True
    else:
        tool_ok = expected_tool in called_tools
        checks["tool_check"] = tool_ok

    # --- Keyword / refusal-phrase check ---
    if not expected_keywords:
        checks["keyword_check"] = None  # not applicable, don't penalize
        keyword_ok = True
        coverage = None
    else:
        hits = [kw for kw in expected_keywords if kw in answer]
        coverage = len(hits) / len(expected_keywords)
        # Unanswerable/refusal cases: expected_keywords is a list of
        # alternative refusal phrasings (OR semantics) — one hit is enough.
        # Answerable cases: same OR semantics, since we don't know the
        # exact wording of the user's synthetic policy docs.
        keyword_ok = len(hits) > 0
        checks["keyword_check"] = keyword_ok
        checks["keyword_coverage"] = round(coverage, 2)
        checks["keyword_hits"] = hits

    passed = tool_ok and keyword_ok
    checks["passed"] = passed
    return checks


def run_eval(dataset_path: Path, report_path: Path, max_retries: int = 2) -> int:
    cases = load_dataset(dataset_path)
    print(f"Loaded {len(cases)} cases from {dataset_path}")
    print(f"Pass threshold: {PASS_THRESHOLD:.0%}\n")

    results = []
    for i, case in enumerate(cases, 1):
        question = case["question"]
        print(f"[{i}/{len(cases)}] ({case['category']}) {question!r} ... ", end="", flush=True)

        start = time.monotonic()
        try:
            result = run_self_healing_agent(question, max_retries=max_retries)
            error = None
        except Exception as exc:  # noqa: BLE001 — we want to record and continue
            result = {"answer": "", "tool_calls": [], "retries": 0, "grounded": None}
            error = str(exc)
        latency = time.monotonic() - start

        scoring = score_case(case, result) if error is None else {"passed": False}
        status = "PASS" if scoring.get("passed") else "FAIL"
        print(f"{status} ({latency:.1f}s)")
        if error:
            print(f"    ERROR: {error}")

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": question,
                "expected_tool": case.get("expected_tool"),
                "expected_keywords": case.get("expected_keywords", []),
                "answer": result.get("answer"),
                "tool_calls": result.get("tool_calls"),
                "grounded": result.get("grounded"),
                "retries": result.get("retries"),
                "latency_s": round(latency, 2),
                "error": error,
                **scoring,
            }
        )

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    pass_rate = passed / total if total else 0.0
    avg_latency = sum(r["latency_s"] for r in results) / total if total else 0.0

    tool_checked = [r for r in results if r.get("tool_check") is not None]
    tool_accuracy = (
        sum(1 for r in tool_checked if r["tool_check"]) / len(tool_checked)
        if tool_checked
        else None
    )

    by_category = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, {"total": 0, "passed": 0})
        by_category[cat]["total"] += 1
        if r.get("passed"):
            by_category[cat]["passed"] += 1

    summary = {
        "total_cases": total,
        "passed": passed,
        "pass_rate": round(pass_rate, 3),
        "pass_threshold": PASS_THRESHOLD,
        "tool_accuracy": round(tool_accuracy, 3) if tool_accuracy is not None else None,
        "avg_latency_s": round(avg_latency, 2),
        "by_category": {
            cat: {
                "total": v["total"],
                "passed": v["passed"],
                "pass_rate": round(v["passed"] / v["total"], 3) if v["total"] else 0.0,
            }
            for cat, v in by_category.items()
        },
    }

    report = {"summary": summary, "cases": results}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"Total cases:    {total}")
    print(f"Passed:         {passed} ({pass_rate:.1%})")
    print(f"Tool accuracy:  {tool_accuracy:.1%}" if tool_accuracy is not None else "Tool accuracy:  n/a")
    print(f"Avg latency:    {avg_latency:.2f}s")
    print("By category:")
    for cat, v in summary["by_category"].items():
        print(f"  {cat:14s} {v['passed']}/{v['total']}  ({v['pass_rate']:.1%})")
    print(f"\nFull report written to {report_path}")

    if pass_rate < PASS_THRESHOLD:
        print(
            f"\nFAIL: pass rate {pass_rate:.1%} is below threshold {PASS_THRESHOLD:.0%}"
        )
        return 1

    print(f"\nPASS: pass rate {pass_rate:.1%} meets threshold {PASS_THRESHOLD:.0%}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 3 golden-dataset eval harness")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    exit_code = run_eval(args.dataset, args.report, max_retries=args.max_retries)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
