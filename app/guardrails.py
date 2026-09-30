"""
Stage 2: input/output guardrails driven entirely by guardrails.yaml (see that
file's comments for what each rule does and why it's config, not code).

Usage:
    from app.guardrails import check_input, check_output

    result = check_input(user_message)
    if not result.allowed:
        return result.reason  # show this instead of calling the LLM at all

    ...

    result = check_output(answer)
    if not result.allowed:
        answer = result.reason  # replace the LLM's answer before returning it
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_RULES_PATH = Path(__file__).resolve().parent.parent / "guardrails.yaml"
_rules_cache: dict[str, Any] | None = None


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None  # set when allowed is False: the message to show the user
    rule: str | None = None  # which rule triggered, for logging/debugging


def _load_rules() -> dict[str, Any]:
    global _rules_cache
    if _rules_cache is None:
        with open(_RULES_PATH, "r", encoding="utf-8") as f:
            _rules_cache = yaml.safe_load(f) or {}
    return _rules_cache


def reload_rules() -> None:
    """Force the next check_input/check_output call to re-read guardrails.yaml
    from disk. Call this after editing the YAML without restarting the app."""
    global _rules_cache
    _rules_cache = None


def check_input(message: str) -> GuardrailResult:
    rules = _load_rules().get("input_guardrails", {})
    lowered = message.lower()

    for phrase in rules.get("prompt_injection_phrases", []):
        if phrase.lower() in lowered:
            return GuardrailResult(
                allowed=False,
                reason=rules.get("block_message", "This request can't be processed."),
                rule=f"prompt_injection_phrase:{phrase}",
            )

    for pii_name, pattern in (rules.get("pii_patterns") or {}).items():
        if re.search(pattern, message):
            return GuardrailResult(
                allowed=False,
                reason=rules.get("block_message", "This request can't be processed."),
                rule=f"pii_pattern:{pii_name}",
            )

    return GuardrailResult(allowed=True)


def check_output(answer: str) -> GuardrailResult:
    rules = _load_rules().get("output_guardrails", {})
    lowered = answer.lower()

    for topic in rules.get("banned_topics", []):
        if topic.lower() in lowered:
            return GuardrailResult(
                allowed=False,
                reason=rules.get("block_message", "This answer can't be shared."),
                rule=f"banned_topic:{topic}",
            )

    return GuardrailResult(allowed=True)
