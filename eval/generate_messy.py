"""Second synthetic generator for the out-of-distribution (OOD) robustness check.

Designed separately from eval/generate.py: it shares only the Persona/Truth dataclasses. It does NOT
reuse that module's STREAMS. Same author, same hackathon, so this is a robustness check against a
differently shaped data-generating process, not independent validation.

What is different from the first generator:
- Merchant vocabulary (quick-commerce, UPI P2P, fintech, rail/air travel, EMI lenders, ATM cash) and
  description formats (UPI / NEFT / IMPS / ACH / card POS / e-com with reference numbers, city
  suffixes, mixed case and typos on discretionary merchants).
- Amounts are log-normal (heavy right tail) instead of normal.
- Salary lands on an irregular day (24th-31st) and there is one bonus month.
- Positive credits that are NOT income: refunds (truth category shopping) and friend reimbursements.
- Cash withdrawals and EMI/loan debits (truth category "other"), one annual subscription, one
  subscription price increase.
- Planted targets: two duplicate pairs, two large charges, one spike in `travel` (the first
  generator spikes food_delivery). No payday behaviour is planted, so
  Truth.payday_trigger_category is None (any payday finding is a false positive).
- Merchants unknown to the keyword rules (stress the categorizer) and merchants that look like rule
  keywords but are not (RENT THE RUNWAY, METRO CASH AND CARRY, WONDERLA WATER PARK, SWIGGY INSTAMART
  which is groceries, AMAZON PRIME ANNUAL which is a subscription).

Fixed streams (rent, EMI, subscriptions, salary, utilities) keep a stable merchant string per
persona (reference numbers vary but are dropped by normalisation); discretionary merchants get
random formats, so the same shop can normalise to several different merchant strings, as in real
statements.

Truth fields: categories (aligned with df rows), anomaly_ids (2 large charges), duplicate_ids (the
second row of each of 2 pairs), unused_subs (normalised merchants: the cheaper of two overlapping
video subscriptions, plus the gym if no sports spending exists), spike (month, "travel"),
payday_trigger_category = None. MessyPersona adds `subscription_merchants` (normalised monthly
subscriptions, price-increase one included, the annual one excluded) and `planted` counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eval.generate import Persona, Truth
from finpilot.engines.categorize import normalize_merchant

OOD_PERSONAS = {
    "kavya": {"seed": 11, "salary": 54000, "gym_used": False, "rent": 15500, "emis": 1, "opening": 90000.0},
    "rohan": {"seed": 12, "salary": 61000, "gym_used": True, "rent": 17000, "emis": 2, "opening": 110000.0},
    "isha": {"seed": 13, "salary": 48000, "gym_used": False, "rent": 13000, "emis": 1, "opening": 90000.0},
}

# (name, category, per_month, median amount, sigma of log amount)
DISCRETIONARY = [
    ("ZEPTO", "groceries", 6, 520, 0.5),
    ("BLINKIT", "groceries", 4, 640, 0.5),
    ("RELIANCE FRESH", "groceries", 2, 1500, 0.4),
    ("SPAR SUPERMARKET", "groceries", 1, 1900, 0.4),
    ("SWIGGY INSTAMART", "groceries", 3, 480, 0.5),  # looks like food delivery, is groceries
    ("METRO CASH AND CARRY", "groceries", 0.6, 3200, 0.4),  # rule METRO says transport
    ("BIGBASKET", "groceries", 3, 1400, 0.4),  # mainstream merchants the rules do know
    ("SWIGGY", "food_delivery", 5, 340, 0.4),
    ("ZOMATO", "food_delivery", 4, 360, 0.4),
    ("UBER", "transport", 8, 240, 0.5),
    ("AMAZON", "shopping", 4, 1300, 0.8),
    ("MYNTRA", "shopping", 0.8, 1900, 0.5),
    ("APOLLO PHARMACY", "health", 1, 600, 0.5),
    ("EATSURE", "food_delivery", 3, 360, 0.35),
    ("MAGICPIN", "food_delivery", 2, 300, 0.35),
    ("DOMINOS PIZZA", "food_delivery", 1.5, 520, 0.3),
    ("GASTRO PUB", "dining", 1.5, 1900, 0.45),
    ("CAFE COFFEE DAY", "dining", 2, 380, 0.35),
    ("TRATTORIA", "dining", 1, 1700, 0.4),
    ("NAMMA YATRI", "transport", 7, 170, 0.5),
    ("RAPIDO", "transport", 3, 140, 0.4),
    ("INDIAN OIL FUEL", "transport", 2, 1500, 0.3),
    ("FASTAG RECHARGE", "transport", 0.6, 1000, 0.2),
    ("AJIO", "shopping", 1.5, 1800, 0.55),
    ("MEESHO", "shopping", 1.5, 650, 0.5),
    ("FLIPKART", "shopping", 1, 2400, 0.7),
    ("LIFESTYLE STORES", "shopping", 0.6, 3200, 0.45),
    ("RENT THE RUNWAY", "shopping", 0.25, 3500, 0.3),  # rule RENT says rent
    ("NETMEDS", "health", 1, 700, 0.5),
    ("1MG LABS", "health", 0.6, 950, 0.5),
    ("MANIPAL HOSPITAL", "health", 0.3, 2400, 0.5),
    ("BOOKMYSHOW", "entertainment", 1.2, 780, 0.35),
    ("INOX LEISURE", "entertainment", 0.8, 900, 0.3),
    ("STEAM GAMES", "entertainment", 0.5, 900, 0.6),
    ("WONDERLA WATER PARK", "entertainment", 0.15, 2600, 0.3),  # rule WATER says utilities
    ("IRCTC", "travel", 0.8, 850, 0.5),
    ("MAKEMYTRIP", "travel", 0.35, 2600, 0.5),
    ("OYO ROOMS", "travel", 0.25, 1700, 0.4),
]
SPIKE_STREAMS = [("INDIGO AIRLINES", 4200), ("MAKEMYTRIP", 3600), ("OYO ROOMS", 2800), ("IRCTC", 1500)]
CITIES = ["HYDERABAD", "CHENNAI", "KOCHI", "PUNE", "HYD", "JAIPUR", "INDORE", "BLR"]
BANKS = ["okaxis", "ybl", "paytm", "okicici", "oksbi"]
TYPO_P = 0.12
RATE_SCALE = 0.3  # keeps monthly spend near income (travel stays unscaled so it has a non-zero median)


@dataclass
class MessyPersona(Persona):
    subscription_merchants: set = field(default_factory=set)
    planted: dict = field(default_factory=dict)


def _typo(rng, s):
    if len(s) < 5 or rng.random() > TYPO_P:
        return s
    i = int(rng.integers(1, len(s) - 1))
    if s[i] == " " or s[i - 1] == " ":
        return s
    return s[:i] + s[i + 1:] if rng.random() < 0.5 else s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _case(rng, s):
    r = rng.random()
    return s if r < 0.5 else (s.title() if r < 0.8 else s.lower())


def generate_messy(seed, months=12, end="2026-09", salary=54000, gym_used=False, rent=15500, emis=1,
                    opening=60000.0) -> MessyPersona:
    rng = np.random.default_rng(seed)
    periods = pd.period_range(end=end, periods=months, freq="M")
    rows = []
    ref = lambda n=6: int(rng.integers(10 ** (n - 1), 10 ** n))  # noqa: E731

    def noisy(name, plain=False):
        """Free-form description for a discretionary merchant (varies row to row)."""
        n = name if plain else _typo(rng, name)
        fmt = int(rng.integers(0, 6))
        city = str(rng.choice(CITIES))
        if fmt == 0:
            s = f"UPI/{ref(12)}/{n}/{rng.choice(BANKS)}"
        elif fmt == 1:
            s = f"{n} {city} {ref(6)}"
        elif fmt == 2:
            s = f"POS {ref(6)} {n} {city}"
        elif fmt == 3:
            s = f"IMPS/{ref(10)}/{n}"
        elif fmt == 4:
            s = f"ECOM PUR/{n}/{ref(6)}"
        else:
            s = f"VPS*{n} {city}"
        return s if plain else _case(rng, s)

    stable_fmts = ["ACH D- {n} {r}", "NEFT {n} {r}", "IMPS P2A {n} {r}", "UPI {n} {r}", "{n} AUTOPAY {r}"]

    def stable_fmt():
        return str(rng.choice(stable_fmts))

    landlord = str(rng.choice(["MR SRINIVASAN IYER", "MRS LAKSHMI NAIR", "HOUSE RENT MR GURPREET SINGH"]))
    lenders = ["HDFC BANK LTD EMI", "BAJAJ FINSERV EMI"][:emis]
    streams = {  # name -> [fmt, day]
        "rent": [stable_fmt(), int(rng.integers(1, 6))],
        "tatapower": [stable_fmt(), int(rng.integers(8, 14))],
        "fiber": [stable_fmt(), int(rng.integers(14, 19))],
        "jio": [stable_fmt(), int(rng.integers(18, 24))],
        "prime": [stable_fmt(), int(rng.integers(5, 10))],
        "netflix": [stable_fmt(), int(rng.integers(2, 6))],
        "sonyliv": [stable_fmt(), int(rng.integers(10, 15))],
        "gaana": [stable_fmt(), int(rng.integers(20, 26))],
        "gym": [stable_fmt(), int(rng.integers(1, 5))],
    }
    for l in lenders:
        streams[l] = [stable_fmt(), int(rng.integers(4, 9))]
    job = str(rng.choice(["INFOSYS BPM", "ZOHO CORP", "WIPRO LTD", "FRESHWORKS INC"]))

    spike_p, big1_p, big2_p = periods[-6], periods[-4], periods[-9]
    dup1_p, dup2_p, bonus_p = periods[-2], periods[-8], periods[-5]
    annual_p, hike_from = periods[3], 6
    pending_refunds = []

    def add(date, desc, cat, amt, tag=None):
        rows.append(dict(date=date, description=desc, amount=round(float(amt), 2), category=cat, tag=tag))

    def stable(key, name, date, cat, amt):
        fmt, _ = streams[key]
        add(date, _case(rng, fmt.format(n=name, r=ref(6))), cat, amt, f"sub:{name}" if cat == "subscriptions" else None)

    for mi, p in enumerate(periods):
        dim = p.days_in_month

        def d(day, p=p, dim=dim):
            return p.to_timestamp() + pd.Timedelta(days=int(min(max(day, 1), dim)) - 1) \
                + pd.Timedelta(hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))

        # income: irregular salary day, bonus month, small interest credit
        add(d(int(rng.integers(24, 32))), f"NEFT-CR-HDFC0001234-{job} SALARY-N{ref(6)}", "income",
            salary * float(rng.normal(1.0, 0.01)))
        if p == bonus_p:
            add(d(int(rng.integers(10, 20))), f"NEFT-CR-{job} PERFORMANCE BONUS-N{ref(6)}", "income",
                salary * float(rng.uniform(0.9, 1.3)))
        add(d(30), f"SB INT CREDIT {ref(4)}", "income", float(rng.uniform(40, 160)))

        # fixed outflows
        stable("rent", landlord, d(streams["rent"][1]), "rent", -rent)
        for l in lenders:
            stable(l, l, d(streams[l][1]), "other", -(7800 if l.startswith("HDFC") else 3400))
        stable("tatapower", "TATA POWER", d(streams["tatapower"][1]), "utilities", -max(300, rng.normal(1700, 450)))
        stable("fiber", "ACT FIBERNET", d(streams["fiber"][1]), "utilities", -1049)
        stable("jio", "JIO PREPAID", d(streams["jio"][1]), "utilities", -299)
        stable("prime", "PRIME VIDEO", d(streams["prime"][1] + int(rng.integers(-1, 2))), "subscriptions", -179)
        stable("netflix", "NETFLIX", d(streams["netflix"][1] + int(rng.integers(-1, 2))), "subscriptions", -499)
        stable("sonyliv", "SONYLIV", d(streams["sonyliv"][1]), "subscriptions", -(299 if mi < hike_from else 349))
        stable("gaana", "GAANA", d(streams["gaana"][1]), "subscriptions", -99)
        stable("gym", "ANYTIME FITNESS", d(streams["gym"][1]), "subscriptions", -1699)
        if p == annual_p:
            add(d(int(rng.integers(5, 25))), f"POS AMAZON PRIME ANNUAL {ref(6)}", "subscriptions", -1499)

        # cash
        for _ in range(int(rng.poisson(1.3))):
            amt = float(rng.choice([500, 1000, 2000, 2000, 3000, 5000]))
            add(d(int(rng.integers(1, dim + 1))),
                f"ATM WDL {rng.choice(['SBI', 'HDFC', 'AXIS'])} ATM {rng.choice(['KORAMANGALA', 'INDIRANAGAR', 'T NAGAR', 'BANJARA HILLS'])} {ref(4)}",
                "other", -amt)

        # discretionary
        for name, cat, per_month, med, sig in DISCRETIONARY:
            for day in rng.integers(1, dim + 1, rng.poisson(per_month * (1.0 if cat == "travel" else RATE_SCALE))):
                amt = float(rng.lognormal(np.log(med), sig))
                dt = d(int(day))
                add(dt, noisy(name), cat, -amt)
                if cat == "shopping" and rng.random() < 0.4:
                    pending_refunds.append((dt + pd.Timedelta(days=int(rng.integers(5, 12))), name, amt))
        if gym_used:
            for day in rng.integers(1, dim + 1, rng.poisson(1)):
                add(d(int(day)), noisy("SPORTSKART"), "shopping", -float(rng.lognormal(np.log(2400), 0.4)))
        # friend reimbursements: positive but not income
        for day in rng.integers(1, dim + 1, rng.poisson(1.2)):
            add(d(int(day)), f"UPI/{ref(12)}/{rng.choice(['ARUN', 'DIVYA', 'NAVEEN', 'MEGHA'])} SPLIT/{rng.choice(BANKS)}",
                "dining", float(rng.lognormal(np.log(600), 0.5)))

        if p == spike_p:  # extra bookings: a spike in travel (baseline travel is steady and small)
            for k in range(7):
                name, amt = SPIKE_STREAMS[k % len(SPIKE_STREAMS)]
                add(d(int(rng.integers(1, dim + 1))), noisy(name), "travel", -float(rng.lognormal(np.log(amt), 0.25)))
        if p == big1_p:
            add(d(12), f"POS {ref(6)} CROMA ELECTRONICS HYDERABAD", "shopping", -41500, "big")
        if p == big2_p:
            add(d(19), f"NEFT-{ref(8)}-APOLLO HOSPITALS-{ref(5)}", "health", -36800, "big")
        if p == dup1_p:
            dt = d(9)
            add(dt, f"POS {ref(6)} DECATHLON SPORTS BLR", "shopping", -3299.0, "dup_orig")
            add(dt + pd.Timedelta(days=1), f"POS {ref(6)} DECATHLON SPORTS BLR", "shopping", -3299.0, "dup")
        if p == dup2_p:
            dt = d(22)
            add(dt, f"UPI/{ref(12)}/BOOKMYSHOW/okaxis", "entertainment", -1860.0, "dup_orig")
            add(dt + pd.Timedelta(days=1), f"UPI/{ref(12)}/BOOKMYSHOW/okaxis", "entertainment", -1860.0, "dup")

    for dt, name, amt in pending_refunds:
        if dt <= periods[-1].to_timestamp(how="end"):
            add(dt, f"REFUND CR {name} ORDER {ref(7)}", "shopping", amt, "refund")

    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    df["date"] = df["date"].dt.normalize()
    # Truth is defined by construction, in the engines' normalised-merchant form (a stream formatted
    # "<name> AUTOPAY" normalises to "<NAME> AUTOPAY", so the truth carries that exact string).
    merch = {}
    for tag, g in df[df.tag.fillna("").str.startswith("sub:")].groupby("tag"):
        ms = set(g.description.map(normalize_merchant))
        assert len(ms) == 1, (tag, ms)
        merch[tag[4:]] = ms.pop()
    subs = set(merch.values())
    unused = {merch["PRIME VIDEO"]} | (set() if gym_used else {merch["ANYTIME FITNESS"]})
    anomalies, dups = set(df.index[df.tag == "big"]), set(df.index[df.tag == "dup"])
    truth = Truth(categories=df["category"].copy(), anomaly_ids=anomalies, duplicate_ids=dups,
                  unused_subs=unused, spike=(str(spike_p), "travel"), payday_trigger_category=None)
    return MessyPersona(df[["date", "description", "amount"]], truth, float(opening), subs,
                        {"duplicates": len(dups), "large_charges": len(anomalies),
                         "refunds": int((df.tag == "refund").sum()), "spike": 1})


def load_messy(name) -> MessyPersona:
    return generate_messy(**OOD_PERSONAS[name])
