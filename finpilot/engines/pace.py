import pandas as pd

from .analyze import detect_recurring


def pace_alerts(df, today, min_over=0.3, min_abs=500.0):
    """Month-to-date spend per category vs the median of the same day-span in prior months."""
    today = pd.Timestamp(today).normalize()
    month = today.to_period("M")
    rec = {r["merchant"] for r in detect_recurring(df)}
    spend = df[(df["amount"] < 0) & (~df["merchant"].isin(rec))].copy()
    spend["m"] = spend["date"].dt.to_period("M")
    spend["day"] = spend["date"].dt.day
    spend["s"] = -spend["amount"]
    prior_months = sorted(m for m in spend["m"].unique() if m < month)
    if len(prior_months) < 3:
        return []
    within = spend[spend["day"] <= today.day]
    out = []
    for cat, g in within.groupby("category"):
        by_m = g.groupby("m")["s"].sum()
        prior = by_m.reindex(prior_months[-6:], fill_value=0.0)
        mtd = float(by_m.get(month, 0.0))
        expected = float(prior.median())
        if mtd > expected * (1 + min_over) and mtd - expected >= min_abs:
            out.append({"id": f"pace:{cat}:{month}", "type": "pace_alert", "category": cat,
                        "mtd": round(mtd, 2), "expected": round(expected, 2),
                        "over_pct": round((mtd / expected - 1) * 100, 1) if expected else None,
                        "day": int(today.day)})
    return out
