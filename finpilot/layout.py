"""Column-aware statement reading for PDFs and OCR output.

Text extraction throws away where a number sat on the page, but in a statement the column *is* the meaning:
a number under "Withdrawal" is money out, under "Deposit" money in. This module keeps word positions, finds the
header row, and assigns each amount to its column. Callers fall back to plain-line parsing when no header is found.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .statement import _num

_ROLE_WORDS = {
    "debit": {"withdrawal", "withdrawals", "debit", "debits", "dr", "withdrawn", "paidout"},
    "credit": {"deposit", "deposits", "credit", "credits", "cr", "paidin"},
    "balance": {"balance", "bal"},
    "amount": {"amount", "amt"},
}
_DESC_WORDS = {"narration", "description", "particulars", "details", "remarks", "transaction"}
_MONEY = re.compile(r"^\(?-?\d[\d,]*\.\d{2}\)?-?$")
_CUR = {"₹", "rs", "rs.", "inr"}
_DRCR = {"dr", "cr", "dr.", "cr.", "(dr)", "(cr)"}
_DATE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{1,2}[ \-/](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]{0,6}[ \-/,]*\d{2,4})(?![\d/])", re.I)
_MONTH_WORD = re.compile(r"^[A-Za-z]{3,9}$")
_SKIP_LINE = re.compile(r"\b(page\s+\d+|generated on|registered office|closing balance|opening balance|statement of|"
                        r"total|brought forward|carried forward|this is a computer)", re.I)


@dataclass
class Word:
    t: str
    x0: float
    x1: float
    y: float          # vertical centre (after any deskew)
    h: float = 10.0

    @property
    def xc(self):
        return (self.x0 + self.x1) / 2


def group_lines(words: list[Word]) -> list[list[Word]]:
    words = sorted(words, key=lambda w: (w.y, w.x0))
    lines: list[list[Word]] = []
    for w in words:
        if lines and abs(w.y - sum(x.y for x in lines[-1]) / len(lines[-1])) <= 0.55 * max(w.h, lines[-1][0].h):
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w.x0) for l in lines]


def deskew(items: list[dict]) -> float:
    """Median baseline angle of the boxes, in radians; items need 'ang'. Returns 0 when it is negligible."""
    angs = sorted(i["ang"] for i in items if "ang" in i)
    if not angs:
        return 0.0
    th = angs[len(angs) // 2]
    return th if abs(th) > math.radians(0.3) else 0.0


def _norm(t: str) -> str:
    return re.sub(r"[^a-z]", "", t.lower())


def _merge_tokens(line: list[Word]) -> list[Word]:
    """Glue '₹' + '1,234.00' and '1,234.00' + 'Dr' into single money words."""
    out: list[Word] = []
    for w in line:
        if out and w.t.lower() in _DRCR and _MONEY.match(out[-1].t.split()[-1] if out[-1].t else ""):
            p = out.pop(); out.append(Word(f"{p.t} {w.t}", p.x0, w.x1, p.y, p.h)); continue
        if out and out[-1].t.lower() in _CUR and _MONEY.match(w.t):
            p = out.pop(); out.append(Word(w.t, p.x0, w.x1, w.y, w.h)); continue
        out.append(w)
    return out


def _is_money(w: Word) -> bool:
    toks = w.t.replace("₹", "").split()
    if not toks:
        return False
    core = re.sub(r"^(rs\.?|inr)", "", toks[0], flags=re.I)
    return bool(_MONEY.match(core)) and (len(toks) == 1 or toks[1].lower() in _DRCR)


def find_header(lines: list[list[Word]]):
    for i, line in enumerate(lines[:60]):
        roles: dict[str, Word] = {}
        for w in line:
            n = _norm(w.t)
            for role, names in _ROLE_WORDS.items():
                if n in names and role not in roles:
                    roles[role] = w
        if "amount" in roles and ({"debit", "credit"} & roles.keys()):
            roles.pop("amount")  # 'Withdrawal Amt.': amt is part of the column title
        if not ({"debit", "credit", "amount"} & roles.keys()):
            continue
        if any(_DATE.match(w.t) for w in line):
            continue
        if not any(_norm(w.t) in _DESC_WORDS or _norm(w.t) == "date" for w in line):
            continue
        return i, roles, line
    return None, None, None


def _assign(w: Word, roles: dict[str, Word], tol: float):
    best, bd = None, 1e9
    for role, h in roles.items():
        d = min(abs(w.x1 - h.x1), abs(w.xc - h.xc))
        if d < bd:
            best, bd = role, d
    return best if bd <= tol else None


def parse_positioned(lines: list[list[Word]]):
    """Returns (recs, header_seen) where recs are dicts for ingest._resolve_signs; None if no header is found."""
    hi, roles, hdr = find_header(lines)
    if hi is None:
        return None
    xs = sorted(h.xc for h in roles.values())
    gaps = [b - a for a, b in zip(xs, xs[1:])] or [120.0]
    tol = max(30.0, 0.55 * min(gaps))
    desc_x = next((w.x0 for w in hdr if _norm(w.t) in _DESC_WORDS), None)
    first_money_x = min(h.x0 for h in roles.values())
    recs: list[dict] = []
    for line in lines[hi + 1:]:
        line = _merge_tokens(line)
        text = " ".join(w.t for w in line)
        h2, r2, _ = find_header([line])
        if h2 is not None:  # repeated header on a new page
            continue
        m = _DATE.match(line[0].t) if line else None
        if not m and len(line) >= 3 and _MONTH_WORD.match(line[1].t) and _DATE.match(" ".join(w.t for w in line[:3])):
            m, ndate = True, 3  # "05 Jan 2026" split into three words
            date_txt = " ".join(w.t for w in line[:3])
        elif m:
            ndate, date_txt = 1, m.group(1)
        glued = ""
        if m is not None and ndate == 1 and len(line[0].t) > len(date_txt):
            glued = line[0].t[len(date_txt):]  # "01/01/2026UPI-..." with no space after the date
        if not m and len(line) >= 3 and sum(_is_money(w) for w in line) >= 1 and recs and not _SKIP_LINE.search(text) \
                and any(_is_money(w) and w.xc >= first_money_x - tol for w in line):
            m, ndate, date_txt = True, 0, recs[-1]["date"]  # date unreadable (OCR): keep the money, borrow the neighbour's date
        if not m:
            # wrapped narration: starts in the narration column, no money in the numeric columns
            if recs and not _SKIP_LINE.search(text) and desc_x is not None and abs(line[0].x0 - desc_x) <= 18 \
                    and not any(_is_money(w) for w in line) and len(text) < 60:
                recs[-1]["desc"] = f"{recs[-1]['desc']} {text}".strip()
            continue
        rest = line[ndate:]
        if glued:
            rest = [Word(glued, line[0].x0, line[0].x1, line[0].y, line[0].h)] + rest
        if rest and _DATE.match(rest[0].t) and not _is_money(rest[0]):
            rest = rest[1:]  # value date column
        desc_words, cols = [], {}
        for w in rest:
            if _is_money(w) and w.xc >= first_money_x - tol:
                role = _assign(w, roles, tol)
                if role and role not in cols:
                    cols[role] = w
                    continue
            desc_words.append(w.t)
        debit, hd = _num(cols["debit"].t) if "debit" in cols else (None, None)
        credit, hc = _num(cols["credit"].t) if "credit" in cols else (None, None)
        amt, known = None, False
        if debit is not None or credit is not None:
            amt, known = abs(credit or 0.0) - abs(debit or 0.0), True
        elif "amount" in cols:
            v, h = _num(cols["amount"].t)
            if v is not None:
                amt, known = (h * abs(v), True) if h is not None else ((v, True) if v < 0 else (abs(v), False))
        bal = None
        if "balance" in cols:
            bal, bh = _num(cols["balance"].t)
            if bal is not None and bh == -1:
                bal = -abs(bal)
        recs.append({"date": date_txt, "desc": " ".join(desc_words), "amt": amt, "known": known, "bal": bal,
                     "n": 0, "repair": amt is None and bal is not None, "date_guess": date_txt == (recs[-1]["date"] if recs else None) and ndate == 0})
    return recs
