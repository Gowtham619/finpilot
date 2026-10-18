"""Synthetic personas with known ground truth for evaluation and demo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# (display name, category, kind, params). Rows marked NOT-IN-RULES are deliberately
# absent from the keyword rules so the LLM fallback has real work to do.
STREAMS = [
    ("SUNRISE APARTMENTS RENT", "rent", "monthly", {"day": 5, "amt": 18000}),
    ("BESCOM ELECTRICITY", "utilities", "monthly", {"day": 12, "amt": 1500, "sd": 250}),
    ("AIRTEL BROADBAND", "utilities", "monthly", {"day": 15, "amt": 999}),
    ("NETFLIX", "subscriptions", "monthly", {"day": 3, "amt": 649}),
    ("HOTSTAR", "subscriptions", "monthly", {"day": 8, "amt": 299}),
    ("SPOTIFY", "subscriptions", "monthly", {"day": 20, "amt": 119}),
    ("CULT FIT GYM", "subscriptions", "monthly", {"day": 10, "amt": 1500}),
    ("BIGBASKET", "groceries", "poisson", {"per_month": 5, "amt": 1600, "sd": 400}),
    ("DMART", "groceries", "poisson", {"per_month": 3, "amt": 1100, "sd": 300}),
    ("SWIGGY", "food_delivery", "poisson", {"per_month": 4, "amt": 380, "sd": 90}),
    ("ZOMATO", "food_delivery", "poisson", {"per_month": 3, "amt": 340, "sd": 90}),
    ("TOAST RESTAURANT", "dining", "poisson", {"per_month": 2, "amt": 1400, "sd": 300}),
    ("BLUE TOKAI COFFEE", "dining", "poisson", {"per_month": 4, "amt": 320, "sd": 60}),  # NOT-IN-RULES
    ("UBER", "transport", "poisson", {"per_month": 8, "amt": 260, "sd": 80}),
    ("AMAZON", "shopping", "poisson", {"per_month": 3, "amt": 1500, "sd": 700}),
    ("NYKAA", "shopping", "poisson", {"per_month": 1, "amt": 1800, "sd": 600}),  # NOT-IN-RULES
    ("PVR CINEMAS", "entertainment", "poisson", {"per_month": 1.5, "amt": 800, "sd": 150}),  # NOT-IN-RULES
    ("APOLLO PHARMACY", "health", "poisson", {"per_month": 1, "amt": 650, "sd": 250}),
    ("DR SHAH CLINIC", "health", "poisson", {"per_month": 0.4, "amt": 1200, "sd": 300}),  # NOT-IN-RULES
    ("REDBUS", "travel", "poisson", {"per_month": 0.5, "amt": 900, "sd": 200}),  # NOT-IN-RULES
]

PERSONAS = {
    "priya": {"seed": 1, "salary": 62000, "gym_used": False},
    "arjun": {"seed": 2, "salary": 58000, "gym_used": False},
    "meera": {"seed": 3, "salary": 70000, "gym_used": True},
}


@dataclass
class Truth:
    categories: pd.Series
    anomaly_ids: set
    duplicate_ids: set
    unused_subs: set
    spike: tuple
    payday_trigger_category: str


@dataclass
class Persona:
    df: pd.DataFrame
    truth: Truth
    opening_balance: float


def generate(seed=0, months=12, end="2026-09", salary=62000, gym_used=False) -> Persona:
    rng = np.random.default_rng(seed)
    periods = pd.period_range(end=end, periods=months, freq="M")
    rows = []

    def desc(name):
        ref = int(rng.integers(100000, 999999))
        return str(rng.choice([f"UPI-{name}-{ref}", f"POS {name} BANGALORE", name, f"{name} PAYMENT {ref}"]))

    def add(date, name, cat, amt, tag=None):
        rows.append(dict(date=date, description=desc(name), amount=round(float(amt), 2),
                         category=cat, tag=tag))

    spike_p, dup_p, big_p = periods[-3], periods[-4], periods[-5]
    streams = list(STREAMS)
    if gym_used:
        streams.append(("DECATHLON", "shopping", "poisson", {"per_month": 1, "amt": 2500, "sd": 500}))

    for p in periods:
        dim = p.days_in_month

        def d(day, p=p, dim=dim):
            return p.to_timestamp() + pd.Timedelta(days=min(day, dim) - 1)

        add(d(1), "ACME CORP SALARY", "income", salary)
        for name, cat, kind, prm in streams:
            if kind == "monthly":
                add(d(prm["day"]), name, cat, -max(1.0, rng.normal(prm["amt"], prm.get("sd", 0))))
            else:
                mult = 3.0 if (p == spike_p and cat == "food_delivery") else 1.0
                for day in rng.integers(1, dim + 1, rng.poisson(prm["per_month"] * mult)):
                    add(d(int(day)), name, cat, -max(50.0, rng.normal(prm["amt"], prm["sd"])))
        for day in rng.integers(1, 4, 3):  # payday behavior trigger
            add(d(int(day)), "SWIGGY", "food_delivery", -max(50.0, rng.normal(380, 90)), "payday")
        if p == big_p:
            add(d(17), "GLOBALTECH TRADING LTD", "shopping", -24000, "big")
        if p == dup_p:
            add(d(21), "AMAZON", "shopping", -2199.0, "dup_orig")
            add(d(22), "AMAZON", "shopping", -2199.0, "dup")

    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    unused = {"HOTSTAR"} if gym_used else {"HOTSTAR", "CULT FIT GYM"}
    truth = Truth(
        categories=df["category"].copy(),
        anomaly_ids=set(df.index[df.tag == "big"]),
        duplicate_ids=set(df.index[df.tag == "dup"]),
        unused_subs=unused,
        spike=(str(spike_p), "food_delivery"),
        payday_trigger_category="food_delivery",
    )
    return Persona(df[["date", "description", "amount"]], truth, 40000.0)


def load_persona(name) -> Persona:
    return generate(**PERSONAS[name])
