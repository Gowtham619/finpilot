"""Read any statement file (CSV/TSV, text PDF, scanned PDF, photo/screenshot), mask PII, return normalized rows.

Everything runs locally: pdfplumber for text PDFs, RapidOCR (ONNX) for scans and images. The raw statement is
never sent to an LLM; only masked descriptions are stored and used downstream.
"""
from __future__ import annotations

import io
import math
import re
from collections import Counter

import pandas as pd

from . import layout, pii
from .agents.common import plural
from .statement import ParseResult, _decode, _num, _rows, parse_records, parse_table

MAX_PAGES = 30
OCR_DPI = 200

_MONEY = re.compile(r"(?:(?:₹|Rs\.?|INR)\s*)?\(?-?\d[\d,]*\.\d{2}\)?(?:\s*(?:Dr|Cr|DR|CR)\.?)?(?=\s|$)")
_DATE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{1,2}[ \-/](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]{0,6}[ \-/,]*\d{2,4})(?![\d/])", re.I)
_CREDIT_WORDS = re.compile(r"\b(SALARY|REFUND|CASHBACK|INTEREST CREDIT|INTEREST PAID|DEPOSIT|CREDIT|REVERSAL|INWARD|RECEIVED)\b", re.I)
_IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"BM", b"II*\x00", b"MM\x00*", b"RIFF")


def kind(raw: bytes) -> str:
    if raw[:4] == b"%PDF":
        return "pdf"
    if raw[:4] in _IMAGE_MAGIC or raw[:3] == b"\xff\xd8\xff" or raw[:2] == b"BM":
        return "image"
    return "text"


# ---------------------------------------------------------------- OCR

_ocr = None


def _engine():
    global _ocr
    if _ocr is None:
        try:
            from rapidocr import RapidOCR
        except ImportError:
            raise ValueError("OCR is not available on this server (install `rapidocr`); upload a CSV instead")
        _ocr = RapidOCR()
    return _ocr


def ocr_words(image) -> list[layout.Word]:
    """PIL image -> positioned words (deskewed). Multi-token boxes are split with x positions interpolated."""
    import numpy as np
    res = _engine()(np.array(image.convert("RGB")))
    boxes, texts = getattr(res, "boxes", None), getattr(res, "txts", None)
    if boxes is None or texts is None or len(texts) == 0:
        return []
    items = []
    for b, t in zip(boxes, texts):
        (ax, ay), (bx, by) = b[0], b[1]
        items.append({"t": str(t), "ang": math.atan2(by - ay, bx - ax), "pts": b})
    th = layout.deskew(items)
    cs, sn = math.cos(th), math.sin(th)
    words = []
    for it in items:
        pts = [(x * cs + y * sn, -x * sn + y * cs) for x, y in it["pts"]]  # rotate the page flat
        x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
        ys = [p[1] for p in pts]
        yc, h = sum(ys) / 4, max(ys) - min(ys)
        toks = it["t"].split()
        total = max(len(it["t"]), 1)
        pos = 0
        for tok in toks:
            i = it["t"].index(tok, pos)
            pos = i + len(tok)
            parts = [(m.group(0), i + m.start(), i + m.end()) for m in re.finditer(r"\d[\d,]*\.\d{2}", tok)] \
                if re.fullmatch(r"(?:\d[\d,]*\.\d{2}){2,}", tok) else [(tok, i, pos)]  # OCR glued two amounts together
            for text, a, b in parts:
                words.append(layout.Word(text, x0 + (x1 - x0) * a / total, x0 + (x1 - x0) * b / total, yc, h))
    return words


def words_to_lines(words: list[layout.Word]) -> list[str]:
    out = []
    for line in layout.group_lines(words):
        s = line[0].t
        for prev, cur in zip(line, line[1:]):
            s += ("\t" if cur.x0 - prev.x1 > 1.5 * prev.h else " ") + cur.t
        out.append(s)
    return out


def ocr_lines(image) -> list[str]:
    return words_to_lines(ocr_words(image))


# ---------------------------------------------------------------- text lines -> records

def _signed(tokens: list[str], desc: str):
    """Return (amount_magnitude_signed_if_known, sign_known, balance)."""
    vals = [_num(t) for t in tokens]
    nums = [v for v, _ in vals]
    hints = [h for _, h in vals]
    if len(nums) == 1:
        v, h = nums[0], hints[0]
        if v is None:
            return None, False, None
        if h is not None:
            return h * abs(v), True, None
        if v < 0:
            return v, True, None
        return abs(v), False, None
    if len(nums) == 2:
        v, b = nums
        h = hints[0]
        if v is None:
            return None, False, None
        return (h * abs(v) if h is not None else (v if v < 0 else abs(v))), (h is not None or v < 0), b
    d, c, b = nums[-3:]
    return abs(c or 0.0) - abs(d or 0.0), True, b


def parse_text_lines(lines: list[str], max_rows: int = 50000) -> ParseResult:
    recs, last = [], None
    for raw_line in lines:
        line = raw_line.replace("\t", "  ").strip()
        m = _DATE.match(line)
        if m:
            rest = line[m.end():].strip()
            m2 = _DATE.match(rest)  # second (value) date column
            if m2:
                rest = rest[m2.end():].strip()
            toks = []
            while True:  # peel money tokens off the right end
                found = [t for t in _MONEY.finditer(rest)]
                tail = found[-1] if found else None
                if tail is None or rest[tail.end():].strip():
                    break
                toks.insert(0, tail.group(0))
                rest = rest[:tail.start()].strip()
                if len(toks) == 3:
                    break
            if not toks:
                continue
            amt, known, bal = _signed(toks, rest)
            last = {"date": m.group(1), "desc": rest, "amt": amt, "known": known, "bal": bal, "n": len(toks)}
            recs.append(last)
        elif last is not None and not _MONEY.search(line) and len(line) < 80 and len(last["desc"]) < 160:
            last["desc"] = f"{last['desc']} {line}".strip()  # wrapped narration
    if len(recs) < 5:
        raise ValueError("could not find transaction rows in this file (need at least 5 dated rows with amounts)")
    notes = _resolve_signs(recs)
    out = []
    for r in recs:
        out.append((r["date"], r["desc"], r["amt"], r["bal"]))
    return parse_records(out, max_rows=max_rows, notes=notes)


def _resolve_signs(recs: list[dict]) -> list[str]:
    """Use the balance column (when present) to decide debit/credit, else Dr/Cr marks, else keywords."""
    notes = []
    counts = Counter(r["n"] for r in recs)
    repaired = sum(bool(r.get("repair")) for r in recs)
    if counts.most_common(1)[0][0] >= 2:  # most rows show amount+balance; a lone number is the balance (OCR dropped the amount)
        for r in recs:
            if r["n"] == 1 and r["amt"] is not None:
                r["bal"], r["amt"], r["known"] = abs(r["amt"]), None, False
                r["desc"] = re.sub(r"\s+[\d,]+\.?\d?$", "", r["desc"])
                r["repair"] = True
                repaired += 1
    with_bal = [r for r in recs if r["bal"] is not None]
    if len(with_bal) >= 3 and len(with_bal) == len(recs):
        def score(seq):
            return sum(abs(abs(b["bal"] - a["bal"]) - abs(b["amt"])) <= max(1.0, abs(b["bal"]) * 1e-4)
                       for a, b in zip(seq, seq[1:]) if b["amt"] is not None)
        asc, desc = score(recs), score(recs[::-1])
        seq = recs if asc >= desc else recs[::-1]
        best = min(max(asc, desc) + repaired, len(recs) - 1)
        for a, b in zip(seq, seq[1:]):
            if b.get("repair"):
                b["amt"], b["known"] = b["bal"] - a["bal"], True
            elif b["amt"] is not None and not b["known"]:
                b["amt"] = abs(b["amt"]) if b["bal"] >= a["bal"] else -abs(b["amt"])
                b["known"] = True
        if repaired:
            notes.append(f"{plural(repaired, 'row')} had an unreadable amount, recovered from the running balance")
        notes.append(f"{best}/{len(recs) - 1} rows reconcile with the running balance"
                     + ("" if best >= 0.9 * (len(recs) - 1) else " (low: check the extracted values, OCR may have misread digits)"))
    guessed = 0
    for r in recs:
        if r["amt"] is not None and not r["known"]:
            r["amt"] = abs(r["amt"]) if _CREDIT_WORDS.search(r["desc"]) else -abs(r["amt"])
            guessed += 1
    if guessed:
        notes.append(f"{plural(guessed, 'row')} had no Dr/Cr marker or balance, so debit or credit was guessed from the description")
    return notes


# ---------------------------------------------------------------- PDF / image / text entry points

def from_positioned(words: list[layout.Word], max_rows: int):
    """Column-aware parse; None when the page has no recognizable header row."""
    recs = layout.parse_positioned(layout.group_lines(words))
    if not recs or len(recs) < 5:
        return None
    notes = _resolve_signs(recs)
    guessed = sum(bool(r.get("date_guess")) for r in recs)
    if guessed:
        notes.append(f"{plural(guessed, 'row')} had an unreadable date and was placed on the same day as the row above")
    try:
        return parse_records([(r["date"], r["desc"], r["amt"], r["bal"]) for r in recs], max_rows=max_rows, notes=notes)
    except ValueError as e:
        if "too many rows" in str(e):
            raise
        return None  # header found but the columns did not line up: let the line parser try


def _from_pdf(raw: bytes, max_rows: int) -> ParseResult:
    try:
        import pdfplumber
    except ImportError:
        raise ValueError("PDF reading is not available on this server (install `pdfplumber`); upload a CSV instead")
    try:
        pdf = pdfplumber.open(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"could not read PDF: {e}")
    with pdf:
        pages = pdf.pages[:MAX_PAGES]
        notes = [f"only the first {MAX_PAGES} pages were read"] if len(pdf.pages) > MAX_PAGES else []
        table_rows, words, ocr_used = [], [], False
        for i, p in enumerate(pages):
            for t in p.extract_tables() or []:
                table_rows += [[(c or "").replace("\n", " ").strip() for c in r] for r in t]
            words += [layout.Word(w["text"], w["x0"], w["x1"], (w["top"] + w["bottom"]) / 2 + i * 5000, w["bottom"] - w["top"])
                      for w in p.extract_words(x_tolerance=1.5)]
        if not words and not table_rows:
            ocr_used = True
            for i, p in enumerate(pages):
                for w in ocr_words(p.to_image(resolution=OCR_DPI).original):
                    w.y += i * 20000
                    words.append(w)
            notes.append("scanned PDF read with OCR; check amounts and balances")
    res = None
    if table_rows and not ocr_used:
        try:
            res = parse_table(table_rows, max_rows=max_rows)
        except ValueError:
            res = None
    if res is None:
        res = from_positioned(words, max_rows)
    if res is None:
        res = parse_text_lines(words_to_lines(words), max_rows=max_rows)
        res.notes.append("no column headings found; debit and credit were inferred without column positions")
    res.notes = notes + res.notes
    return res


def _from_image(raw: bytes, max_rows: int) -> ParseResult:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:
        raise ValueError(f"could not read image: {e}")
    words = ocr_words(img)
    res = from_positioned(words, max_rows)
    if res is None:
        res = parse_text_lines(words_to_lines(words), max_rows=max_rows)
        res.notes.append("no column headings found; debit and credit were inferred without column positions")
    res.notes = ["image read with OCR; check amounts and balances"] + res.notes
    return res


def _from_text(raw: bytes, max_rows: int) -> ParseResult:
    try:
        return parse_table(_rows(_decode(raw)), max_rows=max_rows)
    except ValueError as e:
        if "no usable header" not in str(e):  # a real data problem, not a missing header
            raise
        try:  # headerless text export (copy/paste from net-banking): try the line parser
            return parse_text_lines(_decode(raw).splitlines(), max_rows=max_rows)
        except ValueError:
            raise e


def ingest(raw: bytes, max_rows: int = 50000, mask: bool = True) -> ParseResult:
    k = kind(raw)
    res = _from_pdf(raw, max_rows) if k == "pdf" else _from_image(raw, max_rows) if k == "image" else _from_text(raw, max_rows)
    res.source = k
    if mask:
        counts: Counter = Counter()
        res.df["description"] = [pii.mask_description(d, counts) for d in res.df["description"]]
        res.pii = {a: b for a, b in counts.items() if b}
        res.notes.append(pii.summarize(counts) or "no personal identifiers found in transaction descriptions")
        res.notes.append("statement headers (name, address, account details) are discarded, never stored")
    return res
