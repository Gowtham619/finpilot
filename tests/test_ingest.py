import io
from collections import Counter

import pytest

from finpilot import ingest, pii
from tests.test_api import client  # noqa: F401  (fixture)

reportlab = pytest.importorskip("reportlab")
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

ROWS = [("01/01/2026", "UPI-SWIGGY-9876543210@ybl-402512345678", "450.00", "", "59,550.00"),
        ("02/01/2026", "NEFT-ACME TECHNOLOGIES SALARY", "", "50,000.00", "109,550.00"),
        ("03/01/2026", "UPI-RAHUL SHARMA-rahul@okaxis-Dinner", "1,200.00", "", "108,350.00"),
        ("04/01/2026", "IMPS TO PRIYA NAIR A/C 123456789012", "5,000.00", "", "103,350.00"),
        ("05/01/2026", "POS NETFLIX COM", "649.00", "", "102,701.00"),
        ("06/01/2026", "ATM CASH WITHDRAWAL", "2,000.00", "", "100,701.00"),
        ("07/01/2026", "UBER INDIA", "320.00", "", "100,381.00")]


def make_pdf():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in ["HDFC BANK Account Statement", "Name: Gowtham Amarnath", "Address: 12 MG Road Bengaluru",
                 "A/C No: 50100123456789 IFSC: HDFC0001234 PAN: ABCDE1234F",
                 ]:
        c.drawString(40, y, line)
        y -= 18
    c.drawString(40, y, "Date"); c.drawString(110, y, "Narration")
    c.drawRightString(450, y, "Withdrawal"); c.drawRightString(510, y, "Deposit"); c.drawRightString(575, y, "Balance")
    y -= 18
    for d, n, w, dep, b in ROWS:
        c.drawString(40, y, d); c.drawString(110, y, n[:44])
        c.drawRightString(450, y, w); c.drawRightString(510, y, dep); c.drawRightString(575, y, b)
        y -= 18
    c.save()
    return buf.getvalue()


def check(res):
    assert res.df["amount"].tolist() == [-450, 50000, -1200, -5000, -649, -2000, -320]
    assert res.opening_balance == pytest.approx(60000)
    joined = " ".join(res.df["description"]) + " ".join(res.notes)
    for leak in ("9876543210", "RAHUL", "SHARMA", "PRIYA", "NAIR", "123456789012", "okaxis", "Gowtham", "ABCDE1234F", "HDFC0001234"):
        assert leak not in joined
    assert "NETFLIX" in joined and "SWIGGY" in joined and "ACME TECHNOLOGIES SALARY" in joined


def test_text_pdf_end_to_end():
    check(ingest.ingest(make_pdf()))


def test_scanned_pdf_and_image_via_ocr():
    pdfium = pytest.importorskip("pypdfium2")
    pytest.importorskip("rapidocr")
    img = pdfium.PdfDocument(make_pdf())[0].render(scale=2.5).to_pil()
    png = io.BytesIO(); img.save(png, format="PNG")
    check(ingest.ingest(png.getvalue()))
    # image-only PDF (no text layer) goes through the same OCR path
    pdf = io.BytesIO(); img.convert("RGB").save(pdf, format="PDF")
    check(ingest.ingest(pdf.getvalue()))


def test_csv_descriptions_are_masked_too():
    raw = "date,description,amount\n" + "".join(
        f"2026-01-0{d},UPI-ASHA RAO-asha{d}@ybl-Rent,-{d}00\n" for d in range(1, 7))
    r = ingest.ingest(raw.encode())
    assert not any("ASHA" in x or "@" in x for x in r.df["description"])
    assert any("masked" in n for n in r.notes)


def test_headerless_pasted_text():
    raw = "".join(f"0{d}/02/2026 SWIGGY ORDER {d}  {d}00.00 Dr\n" for d in range(1, 8))
    assert (ingest.ingest(raw.encode()).df["amount"] < 0).all()


def test_pii_unit():
    c = Counter()
    s = pii.mask_description(
        "UPI-RAHUL SHARMA-9876543210@ybl PAN ABCDE1234F a@b.com 1234 5678 9012 Mr Anil Kumar", c)
    for gone in ("RAHUL", "9876543210", "ABCDE1234F", "a@b.com", "5678", "Anil"):
        assert gone not in s
    assert pii.mask_description("UPI-SWIGGY-12", Counter()) == "UPI-SWIGGY-12"
    assert pii.mask_description("POS AMAZON PAY INDIA", Counter()) == "POS AMAZON PAY INDIA"
    assert "XXXX" in pii.mask_text("card 4111 1111 1111 1111") and "1111" in pii.mask_text("card 4111 1111 1111 1111")
    assert pii.mask_free_text("Name: Gowtham\nAddress: 1 MG Road").count("[REDACTED]") == 2


def test_api_accepts_pdf(client):  # noqa: F811
    r = client.post("/api/upload", data={"profile": "pdfp", "opening_balance": ""},
                    files={"file": ("s.pdf", io.BytesIO(make_pdf()), "application/pdf")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["opening_balance"] == pytest.approx(60000) and body["opening_balance_source"] == "statement"
    assert client.get("/api/state/pdfp").json()["has_data"]


def test_garbage_pdf_and_image_give_clear_400(client):  # noqa: F811
    for name, data in (("a.pdf", b"%PDF-1.4 not really"), ("a.png", b"\x89PNG\r\n\x1a\nxxxx")):
        r = client.post("/api/upload", data={"profile": "g", "opening_balance": ""},
                        files={"file": (name, io.BytesIO(data), "application/octet-stream")})
        assert r.status_code == 400
