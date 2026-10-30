"""Generate realistic bank-statement files (bank CSV, text PDF, scanned PDF, PNG) for trying the upload UI.

    .venv/bin/python testdata/make_statements.py      # writes testdata/statements/
Needs reportlab and pypdfium2 (dev only). All data is synthetic; the 'PII' in it is fake on purpose.
"""
import csv
import io
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).parent / "statements"
OPENING = 45000.0


def make_rows(days=120, seed=7, end=date(2026, 10, 19)):
    rnd = random.Random(seed)
    start = end - timedelta(days=days)
    rows = []
    for i in range(days + 1):
        d = start + timedelta(days=i)
        if d.day == 1:
            rows.append((d, "NEFT-ACME TECHNOLOGIES SALARY", 62000.0))
            rows.append((d, "NEFT-RENT-MEERA JOSHI", -18000.0))
        if d.day == 5:
            rows.append((d, "POS NETFLIX COM", -649.0))
            rows.append((d, "UPI-SPOTIFY-spotify@okaxis", -119.0))
        if d.day == 7:
            rows.append((d, "BESCOM ELECTRICITY BILL", -float(rnd.randint(900, 1600))))
        if d.day == 12:
            rows.append((d, "CULT FIT GYM 9876543210", -1800.0))
        if rnd.random() < 0.45:
            rows.append((d, f"UPI-SWIGGY-swiggy{rnd.randint(10,99)}@ybl-{rnd.randint(10**11, 10**12-1)}", -float(rnd.randint(220, 640))))
        if rnd.random() < 0.25:
            rows.append((d, f"UPI-UBER INDIA-ref {rnd.randint(10**9, 10**10-1)}", -float(rnd.randint(120, 420))))
        if rnd.random() < 0.18:
            rows.append((d, "POS BIGBASKET", -float(rnd.randint(700, 2400))))
        if rnd.random() < 0.08:
            rows.append((d, f"UPI-{rnd.choice(['RAHUL SHARMA', 'PRIYA NAIR', 'ARJUN MEHTA'])}-{rnd.randint(7,9)}{rnd.randint(10**8, 10**9-1)}@ybl-Dinner",
                         -float(rnd.randint(300, 2500))))
        if rnd.random() < 0.05:
            rows.append((d, "ATM CASH WITHDRAWAL", -float(rnd.choice([2000, 3000, 5000]))))
    bal, out = OPENING, []
    for d, desc, amt in rows:
        bal += amt
        out.append((d, desc, amt, round(bal, 2)))
    return out


def money(x):
    return f"{abs(x):,.2f}"


def write_csv(rows):
    with open(OUT / "hdfc_style_statement.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["HDFC BANK - Account Statement"])
        w.writerow(["Customer: GOWTHAM AMARNATH", "A/C 50100123456789", "IFSC HDFC0001234", "PAN ABCDE1234F"])
        w.writerow([])
        w.writerow(["Date", "Narration", "Value Dt", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"])
        for d, desc, amt, bal in rows:
            ds = d.strftime("%d/%m/%y")
            w.writerow([ds, desc, ds, money(amt) if amt < 0 else "", money(amt) if amt > 0 else "", f"{bal:,.2f}"])
        w.writerow(["Total", "", "", "", "", ""])
    with open(OUT / "single_amount_dr_cr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Transaction Date", "Description", "Amount (INR)"])
        for d, desc, amt, _ in rows[:80]:
            w.writerow([d.strftime("%d-%b-%Y"), desc, f"{money(amt)} {'Dr' if amt < 0 else 'Cr'}"])


def pdf_bytes(rows):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    def header(y):
        for line in ["HDFC BANK  Account Statement", "Name: GOWTHAM AMARNATH", "Address: 12 MG Road Bengaluru 560001",
                     "A/C No: 50100123456789   IFSC: HDFC0001234   PAN: ABCDE1234F   Ph: 9876543210"]:
            c.drawString(40, y, line)
            y -= 16
        c.drawString(40, y - 6, "Date"); c.drawString(110, y - 6, "Narration")
        c.drawRightString(450, y - 6, "Withdrawal"); c.drawRightString(510, y - 6, "Deposit"); c.drawRightString(575, y - 6, "Balance")
        return y - 26
    y = header(800)
    for d, desc, amt, bal in rows:
        if y < 50:
            c.showPage(); y = header(800)
        c.drawString(40, y, d.strftime("%d/%m/%Y")); c.drawString(110, y, desc[:44])
        c.drawRightString(450 if amt < 0 else 510, y, money(amt)); c.drawRightString(575, y, f"{bal:,.2f}")
        y -= 14
    c.save()
    return buf.getvalue()


def main():
    OUT.mkdir(exist_ok=True)
    rows = make_rows()
    write_csv(rows)
    (OUT / "text_statement.pdf").write_bytes(pdf_bytes(rows))
    import pypdfium2 as pdfium
    small = pdf_bytes(rows[:22])
    img = pdfium.PdfDocument(small)[0].render(scale=2.5).to_pil().convert("RGB")
    img.save(OUT / "scanned_page.png")
    img.save(OUT / "scanned_statement.pdf", format="PDF")  # image-only PDF: forces OCR
    print("wrote", *sorted(p.name for p in OUT.iterdir()), sep="\n  ")
    print(f"opening balance for the plain files: {OPENING:,.0f} (statement files carry it in the balance column)")


if __name__ == "__main__":
    main()
