"""Synthetic exports in the shapes common Indian banks and wallets produce, each with known ground truth."""
import csv
import io
import random
from datetime import date, timedelta

import pytest

from finpilot.statement import parse_statement

OPEN = 700000.0


def truth(n=40, seed=3):
    rnd = random.Random(seed)
    rows, bal, d = [], OPEN, date(2026, 1, 1)
    for i in range(n):
        d += timedelta(days=rnd.choice([0, 1, 1, 2]))
        credit = i % 9 == 0
        amt = float(rnd.choice([1250, 99900, 45000, 100000])) if credit else -float(rnd.choice([120, 649, 2499, 15000, 12345]) + rnd.choice([0, 0.5, 0.75]))
        bal = round(bal + amt, 2)
        rows.append((d, f"UPI-SHOP {i}-REF{i:04d}", amt, bal))
    return rows


def indian(x):  # 1,23,456.78 lakh grouping
    s = f"{abs(x):.2f}"
    whole, frac = s.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{whole}.{frac}"


def western(x):
    return f"{abs(x):,.2f}"


def out(rows_list, delim=","):
    b = io.StringIO()
    w = csv.writer(b, delimiter=delim)
    for r in rows_list:
        w.writerow(r)
    return b.getvalue().encode()


def sbi(t):
    r = [["Account Name :", "MR TEST USER"], ["Account Number :", "00000012345678901"], ["Branch :", "MG ROAD"], [],
         ["Txn Date", "Value Date", "Description", "Ref No./Cheque No.", "Debit", "Credit", "Balance"]]
    for d, desc, a, b in t:
        ds = f"{d.day} {d.strftime('%b %Y')}"
        r.append([ds, ds, desc, f"TRANSFER {d.day}", indian(a) if a < 0 else "", indian(a) if a > 0 else "", indian(b)])
    return out(r)


def icici(t):
    r = [["Transactions List - Gowtham"], ["Account Number", "123456789012"], [],
         ["S No.", "Value Date", "Transaction Date", "Cheque Number", "Transaction Remarks", "Withdrawal Amount (INR )", "Deposit Amount (INR )", "Balance (INR )"]]
    for i, (d, desc, a, b) in enumerate(t, 1):
        ds = d.strftime("%d/%m/%Y")
        r.append([i, ds, ds, "-", desc, western(a) if a < 0 else "0.00", western(a) if a > 0 else "0.00", western(b)])
    return out(r)


def axis(t):
    r = [["Tran Date", "CHQNO", "PARTICULARS", "DR", "CR", "BAL", "SOL"]]
    for d, desc, a, b in t:
        r.append([d.strftime("%d-%m-%Y"), "", desc, f"{abs(a):.2f}" if a < 0 else "", f"{a:.2f}" if a > 0 else "", f"{b:.2f}", "1"])
    return out(r)


def kotak(t):  # single column with (Dr)/(Cr) suffix, balance with Cr suffix
    r = [["Date", "Narration", "Chq/Ref No", "Withdrawal (Dr)/ Deposit (Cr)", "Balance"]]
    for d, desc, a, b in t:
        r.append([d.strftime("%d-%m-%Y"), desc, "", f"{western(a)}({'Dr' if a < 0 else 'Cr'})", f"{western(b)}(Cr)"])
    return out(r)


def creditcard(t):  # no balance, 'Cr' marks credits, rupee symbol, time in date
    r = [["Date", "Transaction Description", "Amount"]]
    for d, desc, a, b in t:
        r.append([d.strftime("%d/%m/%Y 14:32:11"), desc, f"₹ {western(a)}{' Cr' if a > 0 else ''}"])
    return out(r)


def wallet(t):  # + / - signs, ISO dates
    r = [["Date", "Transaction Details", "Amount"]]
    for d, desc, a, b in t:
        r.append([d.isoformat(), desc, f"{'+' if a > 0 else '-'} ₹{western(a)}"])
    return out(r)


def excel_quoted(t):  # ="..." wrapped cells, semicolon delimiter, trailing-minus negatives
    r = [["Date", "Description", "Amount"]]
    for d, desc, a, b in t:
        r.append([f'="{d.strftime("%d.%m.%Y")}"', desc, f"{western(a)}{'-' if a < 0 else ''}"])
    return out(r, ";")


def summary_block(t):  # opening/closing summary rows before and after the table
    r = [["Statement Summary"], ["Opening Balance", western(OPEN)], ["Closing Balance", western(t[-1][3])], [], ["Statement of Account"],
         ["Date", "Narration", "Withdrawals", "Deposits", "Balance"]]
    for d, desc, a, b in t:
        r.append([d.strftime("%d/%m/%y"), desc, western(a) if a < 0 else "", western(a) if a > 0 else "", western(b)])
    r.append(["", "Closing Balance", "", "", western(t[-1][3])])
    r.append(["", "Total", western(sum(-x[2] for x in t if x[2] < 0)), western(sum(x[2] for x in t if x[2] > 0)), ""])
    return out(r)


def newest_first(t):
    return axis(t[::-1])


LAYOUTS = {"sbi": sbi, "icici": icici, "axis": axis, "kotak": kotak, "creditcard": creditcard, "wallet": wallet,
           "excel_quoted": excel_quoted, "summary_block": summary_block, "newest_first": newest_first}
HAS_BALANCE = {"sbi", "icici", "axis", "kotak", "summary_block", "newest_first"}


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_layout(name):
    t = truth()
    r = parse_statement(LAYOUTS[name](t))
    got = list(zip(r.df["date"].dt.date, r.df["amount"].round(2)))
    want = sorted(((d, round(a, 2)) for d, _, a, _ in t), key=lambda x: x[0])
    assert len(r.df) == len(t), f"{name}: {len(r.df)} rows, expected {len(t)}; notes={r.notes}"
    assert sorted(got) == sorted(want), f"{name}: amounts/dates differ"
    if name in HAS_BALANCE:
        assert r.opening_balance == pytest.approx(OPEN), f"{name}: opening {r.opening_balance}"
    else:
        assert r.opening_balance is None


def test_repeated_header_crlf_trailing_columns():
    t = truth()
    lines = axis(t).decode().splitlines()
    lines.insert(20, lines[0])  # page break repeats the header
    raw = ("\r\n".join(l + ",," for l in lines) + "\r\n").encode()
    r = parse_statement(raw)
    assert len(r.df) == len(t) and r.opening_balance == pytest.approx(OPEN)


def test_tab_delimited_and_slash_iso_dates():
    t = truth()
    rows = [["Date", "Description", "Debit", "Credit", "Balance"]] + [
        [d.strftime("%Y/%m/%d"), desc, f"{abs(a):.2f}" if a < 0 else "", f"{a:.2f}" if a > 0 else "", f"{b:.2f}"] for d, desc, a, b in t]
    r = parse_statement(out(rows, "\t"))
    assert len(r.df) == len(t) and r.df["date"].iloc[0].date() == t[0][0]


def test_weekday_prefixed_dates():
    t = truth(12)
    rows = [["Date", "Narration", "Amount"]] + [[d.strftime("%a, %d %b %Y"), desc, f"{a:.2f}"] for d, desc, a, _ in t]
    r = parse_statement(out(rows))
    assert [x.date() for x in r.df["date"]] == sorted(d for d, *_ in t)


def test_overdrawn_balance_with_dr_suffix():
    d0 = date(2026, 3, 1)
    amts = [-500, -300, 2000, -700, -100, -50]
    bal, rows = -1000.0, [["Date", "Narration", "Amount", "Balance"]]
    for i, a in enumerate(amts):
        bal += a
        rows.append([(d0 + timedelta(days=i)).isoformat(), f"X{i}", f"{a}", f"{abs(bal):,.2f} {'Dr' if bal < 0 else 'Cr'}"])
    r = parse_statement(out(rows))
    assert r.opening_balance == pytest.approx(-1000.0)
