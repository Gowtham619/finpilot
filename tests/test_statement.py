import io

import pytest

from finpilot import service
from tests.test_api import client  # noqa: F401  (fixture)
from finpilot.statement import parse_statement


def _b(s):
    return s.encode()


def test_plain_format_unchanged():
    r = parse_statement(_b("date,description,amount\n" + "".join(f"2026-01-{d:02d},X,-{d}\n" for d in range(1, 8))))
    assert len(r.df) == 7 and list(r.df.columns) == ["date", "description", "amount"]
    assert r.opening_balance is None


def test_header_aliases_and_amount_formatting():
    raw = "Txn Date,Narration,Amount (INR)\n" + "".join(
        f"{d:02d}/01/2026,UPI-SWIGGY,\"-1,234.50\"\n" for d in range(1, 8))
    r = parse_statement(_b(raw))
    assert len(r.df) == 7
    assert r.df["amount"].iloc[0] == pytest.approx(-1234.5)


def test_debit_credit_columns_with_balance_infers_opening():
    # HDFC-style: Date, Narration, Value Dt, Withdrawal Amt., Deposit Amt., Closing Balance
    rows = [
        ("01/01/26", "SALARY ACME", "", "50,000.00", "60,000.00"),
        ("02/01/26", "UPI-SWIGGY", "500.00", "", "59,500.00"),
        ("03/01/26", "UPI-UBER", "250.00", "", "59,250.00"),
        ("04/01/26", "NEFT-RENT", "20,000.00", "", "39,250.00"),
        ("05/01/26", "ATM", "2,000.00", "", "37,250.00"),
        ("06/01/26", "AMAZON", "1,000.00", "", "36,250.00"),
    ]
    raw = "Date,Narration,Value Dt,Withdrawal Amt.,Deposit Amt.,Closing Balance\n" + "".join(
        f"{d},{n},{d},\"{w}\",\"{c}\",\"{b}\"\n" for d, n, w, c, b in rows)
    r = parse_statement(_b(raw))
    assert r.opening_balance == pytest.approx(10000.0)
    assert r.df["amount"].tolist() == [50000, -500, -250, -20000, -2000, -1000]
    assert r.df["date"].iloc[0].month == 1 and r.df["date"].iloc[0].day == 1


def test_preamble_footer_and_semicolon_delimiter():
    raw = ("Account Statement for A/C 1234\nPeriod: 01 Jan 2026 to 08 Jan 2026\n\n"
           "Date;Description;Debit;Credit\n"
           + "".join(f"0{d}-Jan-2026;SHOP {d};{d * 100};\n" for d in range(1, 8))
           + "Total;;2800;\n")
    r = parse_statement(_b(raw))
    assert len(r.df) == 7 and (r.df["amount"] < 0).all()
    assert r.dropped == 1


def test_dr_cr_suffix_and_parentheses_and_rupee_symbol():
    raw = "Date,Description,Amount\n" + "".join(
        [f"2026-01-0{d},A,₹{d}00.00 Dr\n" for d in range(1, 4)]
        + ["2026-01-04,B,5000 Cr\n", "2026-01-05,C,(250.00)\n", "2026-01-06,D,Rs. 75\n"])
    amts = parse_statement(_b(raw)).df["amount"].tolist()
    assert amts == [-100, -200, -300, 5000, -250, 75]


def test_type_column_with_unsigned_amount():
    raw = "Date,Description,Type,Amount\n" + "".join(
        f"2026-02-0{d},X,{'CREDIT' if d == 1 else 'DEBIT'},{d}00\n" for d in range(1, 7))
    amts = parse_statement(_b(raw)).df["amount"].tolist()
    assert amts[0] == 100 and all(a < 0 for a in amts[1:])


def test_day_first_inferred_and_ambiguous_noted():
    raw = "Date,Description,Amount\n" + "".join(f"{d:02d}/03/2026,X,-1\n" for d in (13, 14, 15, 16, 17, 18))
    assert parse_statement(_b(raw)).df["date"].iloc[0].day == 13
    amb = "Date,Description,Amount\n" + "".join(f"0{d}/04/2026,X,-1\n" for d in range(1, 7))
    r = parse_statement(_b(amb))
    assert r.df["date"].iloc[0].month == 4 and any("day-first" in n for n in r.notes)


def test_month_first_inferred():
    raw = "Date,Description,Amount\n" + "".join(f"03/{d:02d}/2026,X,-1\n" for d in (13, 14, 15, 16, 17, 18))
    assert parse_statement(_b(raw)).df["date"].iloc[0].day == 13


def test_newest_first_statement_with_balance():
    rows = [("04/01/2026", -100, 900), ("03/01/2026", -100, 1000), ("02/01/2026", -100, 1100),
            ("01/01/2026", -100, 1200), ("31/12/2025", -100, 1300), ("30/12/2025", 1500, 1400)]
    raw = "Date,Description,Amount,Balance\n" + "".join(f"{d},X,{a},{b}\n" for d, a, b in rows)
    r = parse_statement(_b(raw))
    assert r.df["date"].is_monotonic_increasing
    assert r.opening_balance == pytest.approx(-100.0)


def test_latin1_and_bom():
    raw = "﻿Date,Description,Amount\n" + "".join(f"2026-01-0{d},CAFÉ,-5\n" for d in range(1, 7))
    assert len(parse_statement(raw.encode("utf-8")).df) == 6
    assert len(parse_statement(raw.lstrip("﻿").encode("latin-1")).df) == 6


def test_unsupported_formats_have_clear_errors():
    with pytest.raises(ValueError, match="PDF"):
        parse_statement(b"%PDF-1.4 blah")
    with pytest.raises(ValueError, match="Excel"):
        parse_statement(b"PK\x03\x04blah")
    with pytest.raises(ValueError):
        parse_statement(b"\x00\x01\x02" * 100)
    with pytest.raises(ValueError, match="date"):
        parse_statement(b"when,what,how_much\n2026-01-01,A,-1\n")


def test_dropped_rows_are_reported():
    raw = "date,description,amount\n" + "".join(f"2026-01-0{d},X,-1\n" for d in range(1, 7)) + "garbage,X,-1\n2026-01-07,X,abc\n"
    r = parse_statement(_b(raw))
    assert len(r.df) == 6 and r.dropped == 2 and any("2 row" in n for n in r.notes)


def test_row_cap():
    with pytest.raises(ValueError, match="too many rows"):
        parse_statement(_b("date,description,amount\n" + "2026-01-01,X,-1\n" * 20), max_rows=10)


def test_service_parse_csv_still_returns_frame():
    df = service.parse_csv(_b("date,description,amount\n" + "".join(f"2026-01-0{d},X,-1\n" for d in range(1, 7))))
    assert len(df) == 6


def test_api_infers_opening_balance_when_blank(client):
    raw = "Date,Narration,Withdrawal Amt.,Deposit Amt.,Closing Balance\n" + "".join(
        f"0{d}/01/2026,X,100,,{1000 - d * 100}\n" for d in range(1, 7))
    r = client.post("/api/upload", data={"profile": "bank1", "opening_balance": ""},
                    files={"file": ("s.csv", io.BytesIO(raw.encode()), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["opening_balance"] == pytest.approx(1000.0) and body["opening_balance_source"] == "statement"


def test_import_preview_then_confirm(client):
    raw = "Date,Narration,Withdrawal Amt.,Deposit Amt.,Closing Balance\n" + "".join(
        f"0{d}/01/2026,UPI-RAHUL SHARMA-9876543210@ybl,100,,{1000 - d * 100}\n" for d in range(1, 7))
    p = client.post("/api/import/preview", files={"file": ("s.csv", io.BytesIO(raw.encode()), "text/csv")})
    assert p.status_code == 200
    j = p.json()
    assert j["rows"] == 6 and j["opening_balance"] == pytest.approx(1000) and j["pii"]
    assert "RAHUL" not in str(j["sample"]) and "9876543210" not in str(j["sample"])
    assert client.get("/api/state/prev1").json()["has_data"] is False  # preview saved nothing
    c = client.post("/api/import/confirm", json={"token": j["token"], "profile": "prev1", "opening_balance": ""})
    assert c.status_code == 200 and c.json()["opening_balance"] == pytest.approx(1000)
    assert client.get("/api/state/prev1").json()["has_data"]
    again = client.post("/api/import/confirm", json={"token": j["token"], "profile": "prev1"})
    assert again.status_code == 410  # token is single use


def test_import_preview_rejects_bad_file(client):
    r = client.post("/api/import/preview", files={"file": ("a.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert r.status_code == 400


def test_failed_start_keeps_the_preview_for_a_retry(client, monkeypatch):
    raw = "date,description,amount\n" + "".join(f"2026-01-0{d},X,-{d}00\n" for d in range(1, 7))
    tok = client.post("/api/import/preview", files={"file": ("s.csv", io.BytesIO(raw.encode()), "text/csv")}).json()["token"]
    monkeypatch.setattr(service, "start_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        client.post("/api/import/confirm", json={"token": tok, "profile": "retry1", "opening_balance": "0"})
    monkeypatch.undo()
    ok = client.post("/api/import/confirm", json={"token": tok, "profile": "retry1", "opening_balance": "0"})
    assert ok.status_code == 200
