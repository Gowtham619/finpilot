"""Rolling-origin forecast check on 6 datasets (3 eval personas + testdata good_01..03... good_04).

Not pre-registered, small n: a post-hoc diagnostic used to find weaknesses, not a validation.
For each dataset, origins every 15 days back from the end; train on data up to the origin, forecast 30 days, compare
with what happened. err = |median end balance - actual| / monthly spend; bias = (median - actual) / monthly spend
(negative = too pessimistic); cov = share of days inside p10-p90 (nominal 0.80).

    .venv/bin/python -m eval.rolling_origin
"""
import numpy as np  # noqa: F401
import pandas as pd

from finpilot.engines import forecast as F
from finpilot.agents.categorizer import categorize_df
from finpilot.engines.categorize import normalize_merchant
from finpilot.service import parse_csv
from eval.generate import generate, PERSONAS
import glob
import os

def frames():
    out = {}
    for n in PERSONAS:
        p = generate(**PERSONAS[n]); df = p.df.copy()
        df["merchant"] = df["description"].map(normalize_merchant); df["category"] = p.truth.categories
        out[n] = (df, 0.0)
    for f in sorted(glob.glob("testdata/good_0[1-4]*.csv")):
        df = parse_csv(open(f, "rb").read()); df = categorize_df(df, None) if "category" not in df else df
        if "merchant" not in df: df["merchant"] = df["description"].map(normalize_merchant)
        out[os.path.basename(f)[:9]] = (df, {"good_01_": 60000, "good_02_": 8000, "good_03_": 80000, "good_04_": 40000}[os.path.basename(f)[:8]])
    return out

def run(model_fn, H=30, step=15, n=1000):
    res = []
    for name, (df, opening) in frames().items():
        df = df.sort_values("date"); last = df["date"].max().normalize()
        for back in range(120, 30, -step):
            cut = last - pd.Timedelta(days=back)
            train = df[df["date"] <= cut]; test = df[(df["date"] > cut) & (df["date"] <= cut + pd.Timedelta(days=H))]
            if (train["date"].max() - train["date"].min()).days < 80 or test["date"].nunique() < H * 0.7: continue
            bal0 = F.current_balance(train, opening)
            r = model_fn(train, bal0, H, n)
            days = pd.date_range(cut + pd.Timedelta(days=1), periods=H)
            act = bal0 + test.groupby(test["date"].dt.normalize())["amount"].sum().reindex(days).fillna(0).cumsum().to_numpy()
            spend = -train[train.amount < 0].amount.sum() / max((train.date.max() - train.date.min()).days / 30.4, 1)
            p10, p50, p90 = r.p10, r.p50, r.p90
            res.append(dict(ds=name, cut=cut.date(), err=abs(act[-1] - p50[-1]) / spend, bias=(p50[-1] - act[-1]) / spend,
                            cov=float(((act >= p10) & (act <= p90)).mean()), width=(p90[-1] - p10[-1]) / spend))
    return pd.DataFrame(res)



def _no_inflow(m):
    return F.Model(m.recurring, m.cats, m.daily, m.weekdays, m.last_date)


def before(train, bal0, H, n):
    """The engine before irregular income was modelled."""
    return F.monte_carlo(_no_inflow(F.fit(train)), bal0, H, n)


def with_income(train, bal0, H, n):
    return F.monte_carlo(F.fit(train), bal0, H, n)


def shipped(train, bal0, H, n):
    """Irregular income + the typical-range pool the forecaster now uses (flagged one-offs excluded)."""
    return F.monte_carlo(F.fit(train, robust_pool=True), bal0, H, n)


if __name__ == "__main__":
    for name, fn in (("before", before), ("with_income", with_income), ("shipped", shipped)):
        d = run(fn)
        print(f"\n{name}\n" + d.groupby("ds")[["err", "bias", "cov"]].mean().round(3).T.to_string())
        print("mean:", d[["err", "bias", "cov"]].mean().round(3).to_dict())
