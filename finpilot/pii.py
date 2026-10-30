"""Mask personal data in statement text before it is stored or sent anywhere.

Regex/heuristic, best effort: it catches the common Indian-statement identifiers (account, card, IFSC, PAN,
Aadhaar, GSTIN, phone, email, UPI ids) and person names in transfer narrations. It cannot promise to find
every name; merchants are kept so categorization still works.
"""
from __future__ import annotations

import re
from collections import Counter

from .engines.categorize import RULES, STOP

_BUSINESS = {
    "LTD", "LIMITED", "PVT", "PRIVATE", "LLP", "INC", "CORP", "CORPORATION", "CO", "COMPANY", "TECHNOLOGIES",
    "TECHNOLOGY", "TECH", "SOLUTIONS", "SERVICES", "SYSTEMS", "ENTERPRISES", "TRADERS", "TRADING", "STORE",
    "STORES", "MART", "FOODS", "FOOD", "COFFEE", "BAKERY", "SALON", "PAYMENTS", "PAY", "BANK", "SALARY", "PAYROLL",
    "RENT", "EMI", "LOAN", "BILL", "RECHARGE", "INTEREST", "CHARGES", "REFUND", "CASHBACK", "SIP", "MUTUAL", "FUND",
    "INSURANCE", "TAX", "GST", "ONLINE", "INDIA", "CINEMAS", "FITNESS", "GYM", "TRAVELS", "MOBILE", "WALLET",
    "RESTAURANT", "CAFE", "PHARMACY", "HOSPITAL", "SUPERMARKET", "TRANSFER", "PAYMENT", "FROM", "TO", "REF",
    "MERCHANT", "UPI", "NEFT", "IMPS", "RTGS", "ATM", "CASH", "WITHDRAWAL", "DEPOSIT", "CHARGE", "FEE", "FEES",
}
_KNOWN = {w for ks in RULES.values() for k in ks for w in k.split()} | _BUSINESS | STOP
_BANK_CODES = r"(?:OK)?(?:AXIS|ICICI|SBI|HDFCBANK|YBL|PAYTM|APL|IBL|AXL|HDFC|YESBANK|UPI)"

_PATTERNS = [
    ("email", re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b"), "[EMAIL]"),
    ("upi_id", re.compile(r"\b[\w.+]{2,}@[a-z][a-z0-9]{1,20}\b", re.I), "[UPI-ID]"),
    ("gstin", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z0-9][A-Z0-9]\b"), "[GSTIN]"),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "[IFSC]"),
    ("aadhaar", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b"), "[AADHAAR]"),
    ("phone", re.compile(r"(?<![\d.])(?:\+91[\s-]?|91[\s-]|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"), "[PHONE]"),
]
_CARD = re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,4}\b")
_LONG_DIGITS = re.compile(r"(?<![\d.])\d{9,19}(?![\d.])")
_HONORIFIC = re.compile(r"\b(?:MR|MRS|MS|MISS|SHRI|SMT|SRI|DR)\.?\s+[A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){0,2}\b", re.I)
_LABELLED = re.compile(r"(?im)^(\s*(?:account holder|customer name|name|holder|address|a/c holder|nominee)\s*[:\-]\s*).+$")
_PREFIX = re.compile(r"^(UPI|IMPS|NEFT|RTGS|P2P|TPT|MB|IB|FT)\b", re.I)
_DIRECTION = re.compile(r"\b(TO|FROM|BY)\s+((?:[A-Za-z]{2,}\s?){1,3})", re.I)


def _last4(m: re.Match) -> str:
    d = re.sub(r"\D", "", m.group(0))
    return "X" * max(len(d) - 4, 4) + d[-4:]


def _looks_like_person(seg: str) -> bool:
    words = seg.split()
    if not 1 <= len(words) <= 4 or not all(re.fullmatch(r"[A-Za-z]{2,}", w) for w in words):
        return False
    if any(w.upper() in _KNOWN or any(len(k) >= 5 and k in w.upper() for k in _KNOWN) for w in words):
        return False
    if re.fullmatch(_BANK_CODES, seg.replace(" ", ""), re.I):
        return False
    return True


def mask_text(text: str, counts: Counter | None = None) -> str:
    counts = counts if counts is not None else Counter()
    s, n = _CARD.subn(_last4, str(text))  # before Aadhaar: a 16-digit card also contains 12-digit groups
    counts["card_or_account"] += n
    for name, rx, repl in _PATTERNS:
        s, n = rx.subn(repl, s)
        counts[name] += n
    s, n = _LONG_DIGITS.subn(_last4, s)
    counts["card_or_account"] += n
    s, n = _HONORIFIC.subn("[PERSON]", s)
    counts["person"] += n
    return s


def mask_description(desc: str, counts: Counter | None = None) -> str:
    """Mask identifiers, then person names in transfer narrations (UPI-RAHUL SHARMA-..., 'TO PRIYA N')."""
    counts = counts if counts is not None else Counter()
    s = mask_text(desc, counts)
    if _PREFIX.match(s.strip()):
        parts = re.split(r"([-/])", s)
        for i in range(0, len(parts), 2):
            if i > 0 and _looks_like_person(parts[i].strip()):
                parts[i] = "[PERSON]"
                counts["person"] += 1
        s = "".join(parts)

    def direction(m: re.Match) -> str:
        name = m.group(2).strip()
        if _looks_like_person(name):
            counts["person"] += 1
            return f"{m.group(1)} [PERSON]" + (" " if m.group(2).endswith(" ") else "")
        return m.group(0)
    return _DIRECTION.sub(direction, s)


def mask_free_text(text: str, counts: Counter | None = None) -> str:
    """For headers/free text: labelled name and address lines are blanked in addition to identifiers."""
    counts = counts if counts is not None else Counter()
    s, n = _LABELLED.subn(lambda m: m.group(1) + "[REDACTED]", str(text))
    counts["name_or_address"] += n
    return mask_text(s, counts)


def summarize(counts: Counter) -> str | None:
    total = sum(counts.values())
    if not total:
        return None
    detail = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(counts.items()) if v)
    return f"masked {total} personal {'identifier' if total == 1 else 'identifiers'} ({detail})"
