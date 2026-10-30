"""Tolerant reader for bank-statement exports.

Turns the CSV a bank actually gives you (odd headers, separate debit/credit
columns, "1,234.50 Dr", preamble lines, day-first dates) into the plain
date/description/amount frame the pipeline expects.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import pandas as pd

DATE = ["transaction date", "txn date", "tran date", "trans date", "posting date", "post date",
        "booking date", "date", "value date", "value dt"]
DESC = ["description", "narration", "particulars", "transaction details", "transaction description",
        "transaction remarks", "details", "remarks", "memo", "payee", "merchant", "desc"]
AMOUNT = ["amount", "transaction amount", "txn amount", "withdrawal deposit", "deposit withdrawal", "debit credit",
          "credit debit", "amount inr", "amt"]
DEBIT = ["debit", "debit amount", "withdrawal", "withdrawals", "withdrawal amt", "withdrawal amount",
         "paid out", "money out", "dr"]
CREDIT = ["credit", "credit amount", "deposit", "deposits", "deposit amt", "deposit amount",
          "paid in", "money in", "cr"]
BALANCE = ["balance", "closing balance", "running balance", "available balance", "bal"]
TYPE = ["type", "transaction type", "txn type", "dr cr", "cr dr", "drcr"]

_NEG_WORDS = {"dr", "debit", "d", "withdrawal", "db"}
_POS_WORDS = {"cr", "credit", "c", "deposit"}
_DELIMS = [",", ";", "\t", "|"]


@dataclass
class ParseResult:
    df: pd.DataFrame
    opening_balance: float | None = None
    notes: list[str] = field(default_factory=list)
    dropped: int = 0
    pii: dict = field(default_factory=dict)
    source: str = "csv"


def _norm(h) -> str:
    h = re.sub(r"\(.*?\)", " ", str(h).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", h)).strip()


def _pick(headers: list[str], names: list[str]) -> int | None:
    for n in names:  # earlier alias wins, so "txn date" beats "value date"
        if n in headers:
            return headers.index(n)
    return None


def _clean_cell(c: str) -> str:
    c = c.strip()
    m = re.fullmatch(r'=\s*"(.*)"', c)  # Excel text wrapper: ="01/01/2026"
    if m:
        c = m.group(1)
    return c.lstrip("'") if re.match(r"^'\d", c) else c


def _decode(raw: bytes) -> str:
    if raw[:4] == b"%PDF":
        raise ValueError("PDF statements are not supported: download the statement as CSV from your bank")
    if raw[:2] == b"PK" or raw[:4] == b"\xd0\xcf\x11\xe0":
        raise ValueError("Excel files are not supported: save or export the statement as CSV")
    if b"\x00" in raw[:4096]:
        raise ValueError("could not read file: it does not look like a text/CSV file")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _rows(text: str) -> list[list[str]]:
    lines = [ln for ln in text.splitlines()[:60] if ln.strip()]
    delim = max(_DELIMS, key=lambda d: sum(ln.count(d) for ln in lines))
    try:
        return [[_clean_cell(c) for c in r] for r in csv.reader(io.StringIO(text), delimiter=delim)]
    except csv.Error as e:
        raise ValueError(f"could not read CSV: {e}")


def _find_header(rows: list[list[str]]):
    for i, row in enumerate(rows[:60]):
        h = [_norm(c) for c in row]
        d, s = _pick(h, DATE), _pick(h, DESC)
        if d is None or s is None:
            continue
        cols = {"date": d, "desc": s, "amount": _pick(h, AMOUNT), "debit": _pick(h, DEBIT),
                "credit": _pick(h, CREDIT), "balance": _pick(h, BALANCE), "type": _pick(h, TYPE)}
        if cols["amount"] is not None or cols["debit"] is not None or cols["credit"] is not None:
            return i, cols
    return None, None


def _num(v):
    """Parse '1,234.50', '₹500 Dr', '(250)', '75-', 'Rs. 75' -> float, or None. Dr/Cr give a sign hint."""
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return None, None
    hint = None
    m = re.search(r"\(?\s*\b(dr|cr|debit|credit)\b\.?\s*\)?\s*$", s, re.I)
    if m:
        hint = -1 if m.group(1).lower() in ("dr", "debit") else 1
        s = s[:m.start()]
    neg = s.strip().startswith("(") and s.strip().endswith(")") or s.strip().endswith("-")
    s = re.sub(r"(?i)(rs\.?|inr|₹)", "", s)
    s = re.sub(r"[,\s()]", "", s).rstrip("-")
    try:
        x = float(s)
    except ValueError:
        return None, None
    return (-abs(x) if neg else x), hint


def _dates(vals: list[str], notes: list[str]) -> pd.Series:
    s = pd.Series([re.sub(r"[T ]\d{1,2}:\d{2}(:\d{2})?.*$", "", str(v).strip()) for v in vals])
    s = s.str.replace(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})$", r"\1-\2-\3", regex=True)
    if s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$").all():
        return pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
    parts = s.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}$").dropna().astype(int)
    dayfirst = True
    if len(parts):
        if (parts[1] > 12).any() and not (parts[0] > 12).any():
            dayfirst = False
        elif not (parts[0] > 12).any():
            notes.append("dates like 03/04/2026 are ambiguous; read as day-first (3 April)")
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")


def _implied_opening(amounts: list[float], balances: list[float | None]):
    """Return (opening, newest_first) if the balance column reconciles with the amounts, else (None, False)."""
    def consistent(a, b):
        ok = tot = 0
        for i in range(1, len(a)):
            if b[i] is None or b[i - 1] is None:
                continue
            tot += 1
            ok += abs(b[i - 1] + a[i] - b[i]) <= max(1.0, abs(b[i]) * 1e-4)
        return ok, tot
    asc_ok, tot = consistent(amounts, balances)
    desc_ok, _ = consistent(amounts[::-1], balances[::-1])
    if tot < 3 or max(asc_ok, desc_ok) < 0.8 * tot:
        return None, False
    newest_first = desc_ok > asc_ok
    a, b = (amounts[::-1], balances[::-1]) if newest_first else (amounts, balances)
    if b[0] is None:
        return None, False
    return b[0] - a[0], newest_first


def parse_statement(raw: bytes, max_rows: int = 50000) -> ParseResult:
    return parse_table(_rows(_decode(raw)), max_rows=max_rows)


def parse_table(rows: list[list[str]], max_rows: int = 50000) -> ParseResult:
    """Rows of cells (header somewhere near the top) -> normalized transactions."""
    hi, c = _find_header(rows)
    if hi is None:
        raise ValueError("CSV must have columns date, description, amount (or debit/credit); "
                         "no usable header row found (missing: date, description, amount)")
    body = [r for r in rows[hi + 1:] if any(x.strip() for x in r)]
    if len(body) > max_rows:
        raise ValueError(f"CSV has too many rows (max {max_rows})")

    def cell(r, k):
        i = c[k]
        return r[i] if i is not None and i < len(r) else ""

    unmarked_debits = False
    if c["amount"] is not None:
        parsed = [_num(cell(r, "amount")) for r in body]
        cr = sum(h == 1 for _, h in parsed)
        dr = sum(h == -1 for _, h in parsed)
        signed = any(v is not None and v < 0 for v, _ in parsed) or any(
            str(cell(r, "amount")).strip().startswith("+") for r in body)
        unmarked_debits = cr > 0 and dr == 0 and not signed  # e.g. card statements: only credits say "Cr"
    recs = []
    for r in body:
        amt = None
        if c["amount"] is not None:
            amt, hint = _num(cell(r, "amount"))
            if amt is not None and unmarked_debits and hint is None and amt > 0:
                amt = -amt
            if amt is not None:
                t = str(cell(r, "type")).strip().lower() if c["type"] is not None else ""
                if t in _NEG_WORDS:
                    amt = -abs(amt)
                elif t in _POS_WORDS:
                    amt = abs(amt)
                elif hint is not None:
                    amt = hint * abs(amt)
        else:
            d, _ = _num(cell(r, "debit"))
            cr, _ = _num(cell(r, "credit"))
            if d is not None or cr is not None:
                amt = abs(cr or 0.0) - abs(d or 0.0)
        bal, bh = _num(cell(r, "balance")) if c["balance"] is not None else (None, None)
        if bal is not None and bh == -1:
            bal = -abs(bal)  # "1,000.00 Dr" on a balance column means overdrawn
        recs.append((cell(r, "date"), cell(r, "desc"), amt, bal))
    return parse_records(recs, max_rows=max_rows)


def parse_records(recs: list[tuple], max_rows: int = 50000, notes: list[str] | None = None) -> ParseResult:
    """(date_text, description, signed_amount, balance|None) tuples -> ParseResult."""
    if len(recs) > max_rows:
        raise ValueError(f"CSV has too many rows (max {max_rows})")
    notes = notes if notes is not None else []
    dates = _dates([x[0] for x in recs], notes) if recs else pd.Series([], dtype="datetime64[ns]")
    df = pd.DataFrame({"date": dates, "description": [x[1] for x in recs],
                       "amount": [x[2] for x in recs], "balance": [x[3] for x in recs]})
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    good = df["date"].notna() & df["amount"].notna()
    dropped = int((~good).sum())
    df = df[good].reset_index(drop=True)
    if len(df) < 5:
        raise ValueError("need at least 5 valid rows")
    if dropped:
        notes.append(f"{dropped} {'row was' if dropped == 1 else 'rows were'} skipped because they had no readable date or amount (totals and footers, for example)")

    opening = None
    bals = [None if pd.isna(b) else float(b) for b in df["balance"]]
    if any(b is not None for b in bals):
        opening, newest_first = _implied_opening(df["amount"].tolist(), bals)
        if newest_first:
            df = df.iloc[::-1].reset_index(drop=True)
        if opening is not None:
            notes.append(f"opening balance {opening:,.2f} read from the statement's balance column")
    df = df[["date", "description", "amount"]].sort_values("date", kind="stable").reset_index(drop=True)
    df["description"] = df["description"].astype(str).str.strip()
    return ParseResult(df, opening, notes, dropped)
