import json
import re

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


MAX_PROSE = 600
_BAD_CHARS = re.compile(r"[<>`]")
_URL = re.compile(r"https?|www\.|://|\b[\w-]+\.(?:com|in|io|org|net|co)/|\b[\w-]+\.(?:com|io|org|net)\b", re.I)
_MOVE_MONEY = re.compile(r"\b(?:transfer\w*|wire[sd]?|wiring|invest\w*|crypto\w*|borrow\w*|loans?|lenders?|"
                         r"send(?:ing)? (?:the |some |your )?money|move (?:your )?savings)\b", re.I)


def clean_prose(text):
    """Best-effort hygiene for LLM prose: bounded length, no markup/backticks, no URLs or domains, no
    instructions to move money. Number-free prose cannot be verified; this only filters the obvious."""
    return (isinstance(text, str) and bool(text.strip()) and len(text) <= MAX_PROSE
            and not _BAD_CHARS.search(text) and not _URL.search(text) and not _MOVE_MONEY.search(text))


def grounded_or(text, evidence, fallback, strict=False):
    """Accept LLM prose only if it passes clean_prose and every number in it appears in the evidence.
    strict=True uses the stricter percent/currency/number-word matching of verify.ungrounded_numbers (callers
    then pass only intentional numbers as evidence). This is best-effort: prose that states no numbers
    (claims, advice, tone) cannot be verified, so it is only filtered for obvious hygiene problems."""
    if not clean_prose(text):
        return fallback
    return text if not ungrounded_numbers(text, [], evidence, strict=strict) else fallback


def fmt_bal(x):
    """Balance as text; negatives are worded (not signed) because the grounding check compares unsigned figures."""
    return "below zero (overdrawn)" if x < 0 else inr(x)
