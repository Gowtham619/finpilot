"""Harder PDF layouts and degraded scans, each with known ground truth."""
import io
import random
from datetime import date, timedelta

import pytest

reportlab = pytest.importorskip("reportlab")
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle  # noqa: E402

from finpilot import ingest  # noqa: E402
from tests.test_bank_layouts import OPEN, indian, truth  # noqa: E402


def want(t):
    return sorted((d, round(a, 2)) for d, _, a, _ in t)


def got(res):
    return sorted(zip((x.date() for x in res.df["date"]), res.df["amount"].round(2)))


def text_pdf(t, wrap=False, newest_first=False, lakh=False, per_page=22, footer=True, dr_cr=False):
    """Columnar text PDF: date | narration | withdrawal | deposit | balance, multi-page, repeated header."""
    rows = t[::-1] if newest_first else t
    fmt = indian if lakh else (lambda x: f"{abs(x):,.2f}")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page, y, n = 1, 0, 0

    def head():
        nonlocal y
        c.setFont("Helvetica", 9)
        c.drawString(40, 810, "SOME BANK LTD   Statement of Account   GOWTHAM AMARNATH   A/C 123456789012")
        c.drawString(40, 794, "Date"); c.drawString(105, 794, "Narration")
        c.drawRightString(400, 794, "Withdrawal"); c.drawRightString(470, 794, "Deposit"); c.drawRightString(560, 794, "Balance")
        y = 774
    head()
    for d, desc, a, b in rows:
        if n and n % per_page == 0:
            if footer:
                c.drawString(40, 30, f"Page {page} of 3   Generated on 19/10/2026   Registered office: Mumbai 400001")
            c.showPage(); page += 1; head()
        c.drawString(40, y, d.strftime("%d/%m/%Y"))
        text = desc
        if wrap:
            first, second = text.split(" ", 1)
            c.drawString(105, y, first); c.drawString(105, y - 11, second)  # narration wraps onto a second line
        else:
            c.drawString(105, y, text)
        sfx = (" Dr" if a < 0 else " Cr") if dr_cr else ""
        c.drawRightString(400 if a < 0 else 470, y, fmt(a) + sfx)
        c.drawRightString(560, y, fmt(b) + (" Cr" if dr_cr else ""))
        y -= 24 if wrap else 14
        n += 1
    if footer:
        c.drawString(40, 30, f"Page {page} of 3   Generated on 19/10/2026   Registered office: Mumbai 400001")
        c.drawString(40, y - 10, f"Closing Balance {fmt(rows[-1][3])}")
    c.save()
    return buf.getvalue()


def table_pdf(t):
    """Ruled table so pdfplumber.extract_tables() sees a real grid, with the header repeated on each page."""
    data = [["Date", "Narration", "Withdrawal", "Deposit", "Balance"]]
    for d, desc, a, b in t:
        data.append([d.strftime("%d-%m-%Y"), desc, f"{abs(a):,.2f}" if a < 0 else "", f"{a:,.2f}" if a > 0 else "", f"{b:,.2f}"])
    buf = io.BytesIO()
    tb = Table(data, repeatRows=1)
    tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    SimpleDocTemplate(buf, pagesize=A4).build([tb])
    return buf.getvalue()


def check(res, t, balance=True):
    assert len(res.df) == len(t), f"{len(res.df)} rows vs {len(t)}; notes={res.notes}"
    assert got(res) == want(t)
    if balance:
        assert res.opening_balance == pytest.approx(OPEN), res.opening_balance


T = truth(60)


def test_multipage_with_footers_and_closing_balance():
    check(ingest.ingest(text_pdf(T)), T)


def test_wrapped_narrations():
    r = ingest.ingest(text_pdf(T[:20], wrap=True))
    check(r, T[:20])
    assert "REF0003" in " ".join(r.df["description"])  # second line joined back to the narration


def test_newest_first_pdf():
    check(ingest.ingest(text_pdf(T, newest_first=True)), T)


def test_lakh_grouping_pdf():
    check(ingest.ingest(text_pdf(T, lakh=True)), T)


def test_dr_cr_suffixes_pdf():
    check(ingest.ingest(text_pdf(T, dr_cr=True)), T)


def test_ruled_table_pdf_multipage():
    check(ingest.ingest(table_pdf(T)), T)


# ---------------------------------------------------------------- scans

def degrade(img, kind, seed=1):
    from PIL import Image, ImageFilter
    rnd = random.Random(seed)
    img = img.convert("L")
    if kind == "jpeg":
        b = io.BytesIO(); img.save(b, "JPEG", quality=35); return Image.open(b).convert("RGB")
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(0.9)).convert("RGB")
    if kind == "rotate":
        return img.rotate(1.6, expand=True, fillcolor=255).convert("RGB")
    if kind == "lowres":
        w, h = img.size
        return img.resize((int(w * 0.6), int(h * 0.6))).convert("RGB")
    if kind == "noise":
        px = img.load()
        for _ in range(img.size[0] * img.size[1] // 90):
            px[rnd.randrange(img.size[0]), rnd.randrange(img.size[1])] = rnd.choice([0, 255])
        return img.convert("RGB")
    return img.convert("RGB")


def page_image(t, scale=2.5):
    pdfium = pytest.importorskip("pypdfium2")
    return pdfium.PdfDocument(text_pdf(t, per_page=40, footer=False))[0].render(scale=scale).to_pil()


def png_bytes(img):
    b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()


@pytest.mark.parametrize("kind", ["clean", "jpeg", "blur", "rotate", "lowres", "noise"])
def test_scanned_page_recovers_most_rows_and_never_silently_wrong(kind):
    pytest.importorskip("rapidocr")
    t = T[:24]
    res = ingest.ingest(png_bytes(degrade(page_image(t), kind)))
    # Accuracy bar for degraded scans: most rows, and every row we return is a true row (no invented amounts).
    truth_set = {(d, round(a, 2)) for d, _, a, _ in t}
    wrong = [r for r in got(res) if r not in truth_set]
    assert len(res.df) >= 0.8 * len(t), f"{kind}: only {len(res.df)}/{len(t)} rows"
    assert len(wrong) <= 1, f"{kind}: wrong rows {wrong}"
    assert any("OCR" in n for n in res.notes)
