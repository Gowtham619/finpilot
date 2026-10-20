import json
import math
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


def grounded_or(text, evidence, fallback, strict=False, ignore=()):
    """Accept LLM prose only if it passes clean_prose and every number in it appears in the evidence.
    strict=True uses the stricter percent/currency/number-word matching of verify.ungrounded_numbers (callers
    then pass only intentional numbers as evidence). `ignore`: user-supplied strings (e.g. option names) that
    are blanked out before the number check. This is best-effort: prose that states no numbers (claims,
    advice, tone) cannot be verified, so it is only filtered for obvious hygiene problems (and, where the
    caller applies it, risk_claim_consistent)."""
    if not clean_prose(text):
        return fallback
    checked = text
    for s in ignore:
        if isinstance(s, str) and s.strip():
            checked = checked.replace(s, " ")
    return text if not ungrounded_numbers(checked, [], evidence, strict=strict) else fallback


def fmt_bal(x):
    """Balance as text; negatives are worded (not signed) because the grounding check compares unsigned figures."""
    return "below zero (overdrawn)" if x < 0 else inr(x)


def pct_text(p):
    """Probability (0..1) as a percent string, like the UI's pct(): a tiny non-zero value is '<1%', never '0%'."""
    r = _round_half_up(float(p) * 100)
    return "<1%" if p > 0 and r == 0 else f"{r}%"


def pct_values(p):
    """Percent values a narration may state for probability p: the exact value and the displayed integer
    (1 for '<1%'). Used to build strict evidence."""
    v = float(p) * 100
    r = _round_half_up(v)
    return [round(v, 1), 1.0 if v > 0 and r == 0 else float(r)]


def _round_half_up(v):
    return int(math.floor(v + 0.5))


# ---------------------------------------------------------------- risk-claim consistency (best-effort)
def risk_level(p):
    """Level of a computed probability: none (<1%), low (1-5%), some (5-50%), high (>=50%)."""
    p = float(p)
    return "none" if p < 0.01 else "low" if p <= 0.05 else "some" if p < 0.5 else "high"


_ABS_NO_RISK = re.compile(
    r"\bno (?:\w+ )?risk\b|\brisk[- ]free\b|\bzero (?:\w+ )?(?:chance|risk|probability)\b|\bno chance\b|"
    r"\bdefinitely not\b|\bimpossible\b|\bguarantee[ds]?\b|\bno (?:breach\w*|overdraft\w*|dips?|shortfall)\b|"
    r"\b(?:will not|won'?t|cannot|can'?t|never|could not|couldn'?t) (?:\w+ )?(?:go|be|get|end up|fall|dip|run)"
    r"(?: \w+)? (?:overdrawn|below|short|negative|into the red)\b|"
    r"\b(?:will not|won'?t|wouldn'?t|does not|doesn'?t|cannot|can'?t|will never) (?:\w+ )?"
    r"(?:hurt|harm|affect|dent|strain)\b",
    re.I)
_SOFT_NO_RISK = re.compile(
    r"\bnothing to worry\b|\bno (?:need to )?worr(?:y|ies)\b|\b(?:look|looks|looking|seem|seems) (?:fine|good|ok(?:ay)?)\b|"
    r"\ball good\b|\bgo ahead\b|\b(?:completely|perfectly|totally|entirely|absolutely) (?:safe|fine|ok(?:ay)?)\b|"
    r"\bno problem\b|\bnot (?:a|any) (?:problem|concern|worry)\b",
    re.I)
_CERTAIN = re.compile(
    r"\bcertain(?:ly|ty)?\b|\bdefinitely\b|\bsurely\b|\binevitabl\w*|\bbound to\b|\bfor sure\b|"
    r"\bwill (?:\w+ )?(?:be|go|end up|run) (?:overdrawn|out of money|broke|negative|into the red)\b|\bwill run out\b",
    re.I)
_QTY = re.compile(
    r"\b(?P<q>half|most|majority|nearly all|almost all|all|every|none|no)\b"
    r"(?: (?:of the|of your|of|the|your|these|those))?(?: \w+){0,2}? (?:futures?|paths?|outcomes?|simulations?|"
    r"scenarios?|runs?)\b(?P<rest>[^.;!?]*)",
    re.I)
_BAD_POL = re.compile(r"\b(?:fail\w*|dips?|dipping|breach\w*|overdrawn|below|drop\w*|fall\w*|run\w* out|short\w*|"
                      r"negative|broke|storm\w*)\b", re.I)
_GOOD_POL = re.compile(r"\b(?:safe|safely|fine|above|stay\w*|surviv\w*|clear|ok(?:ay)?|succeed\w*|pass\w*)\b", re.I)


def _qty_ok(q, share):
    q = q.lower()
    if q == "half":
        return 0.4 <= share <= 0.6
    if q in ("most", "majority"):
        return share > 0.5
    if q in ("nearly all", "almost all"):
        return share >= 0.85
    if q in ("all", "every"):
        return share >= 0.995
    return share <= 0.005  # none / no


def risk_claim_consistent(text, level, shares=()):
    """Best-effort wording check of LLM prose against computed risk (`level` from risk_level()):
    - absolute no-risk wording ('zero chance', 'definitely not', 'will not be overdrawn', 'no breach',
      'will not hurt', 'risk-free') is allowed only at level 'none';
    - soft no-risk wording ('looks fine', 'all good', 'nothing to worry', 'go ahead') not at 'some'/'high';
    - certainty wording ('definitely', 'certain', 'will be overdrawn') only at 'high';
    - quantity words paired with futures/paths/outcomes ('half of futures fail', 'most paths stay above')
      must match one of the computed failing `shares` (0..1); the rest of the sentence decides whether the
      failing share (fail/dip/below) or its complement (safe/stay/above) is meant; unclear polarity is skipped.
    Number-free prose cannot be verified in general; this only catches common contradictions."""
    if not isinstance(text, str):
        return False
    if level != "none" and _ABS_NO_RISK.search(text):
        return False
    if level in ("some", "high") and _SOFT_NO_RISK.search(text):
        return False
    if level != "high" and _CERTAIN.search(text):
        return False
    shares = [float(s) for s in shares if s is not None]
    for m in (_QTY.finditer(text) if shares else ()):
        if re.search(r"\bnot\s*$", text[:m.start()], re.I):
            continue
        rest = m.group("rest")
        bad, good = bool(_BAD_POL.search(rest)), bool(_GOOD_POL.search(rest))
        if bad == good:
            continue  # polarity unknown or mixed: cannot judge
        if not any(_qty_ok(m.group("q"), s if bad else 1.0 - s) for s in shares):
            return False
    return True
