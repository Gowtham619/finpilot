import json

from ..engines.verify import ungrounded_numbers
from ..llm import LLMUnavailable


def inr(x):
    return f"₹{x:,.0f}"


def narrate(llm, system, payload, fallback):
    if llm is None:
        return fallback
    try:
        return llm.json(system, json.dumps(payload, default=str))
    except (LLMUnavailable, ValueError, AttributeError):
        return fallback


def grounded_or(text, evidence, fallback):
    """Accept LLM prose only if every number in it appears in the evidence."""
    if not isinstance(text, str) or not text.strip():
        return fallback
    return text if not ungrounded_numbers(text, [], evidence) else fallback


def fmt_bal(x):
    """Balance as text; negatives are worded (not signed) because the grounding check compares unsigned figures."""
    return "below zero (overdrawn)" if x < 0 else inr(x)
