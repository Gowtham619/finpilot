from ..engines.analyze import analyze_all
from .common import fmt_date, fmt_month, grounded_or, inr, narrate, plural

WEIGHT = {"duplicate_charge": 0, "large_charge": 1, "category_spike": 2, "unused_subscription": 3,
          "behavior_trigger": 4, "subscription": 5}

SYSTEM = ("You review financial findings. Return JSON {\"overview\": str (2 sentences, no new numbers), "
          "\"priority\": [finding ids ordered by importance]}.")


def finding_text(f):
    t = f["type"]
    if t == "unused_subscription":
        return f"{f['merchant']} costs {inr(f['monthly_cost'])}/month and looks unused ({f['detail']})."
    if t == "category_spike":
        return (f"You spent {inr(f['value'])} on {f['category'].replace('_', ' ')} in {fmt_month(f['month'])}, "
                f"{f['pct_change']:.0f}% more than your usual {inr(f['baseline'])}.")
    if t == "duplicate_charge":
        return f"{inr(f['amount'])} was charged twice at {f['merchant']} within 48 hours (second charge on {fmt_date(f['date'])})."
    if t == "large_charge":
        return f"{inr(f['amount'])} at {f['merchant']} on {fmt_date(f['date'])} is far above your normal spend."
    if t == "behavior_trigger":
        return (f"{f['share_pct']:.0f}% of your {f['category'].replace('_', ' ')} spending happens in the "
                f"{f['window_days']} days after payday, {f['ratio']:.1f} times your normal rate.")
    return f"{f['merchant']}: {inr(f['monthly_cost'])}/month."


def run(df, llm=None):
    findings = analyze_all(df)
    for f in findings:
        f["summary"] = finding_text(f)
    findings.sort(key=lambda f: WEIGHT.get(f["type"], 9))
    overview = (f"{plural(len(findings), 'thing')} worth a look. Most important: {findings[0]['summary']}"
                if findings else "Nothing unusual found in this statement.")
    res = narrate(llm, SYSTEM, [{"id": f["id"], "summary": f["summary"]} for f in findings], None)
    if isinstance(res, dict):
        ids = {f["id"] for f in findings}
        pri = res.get("priority")
        order = []
        if isinstance(pri, list):
            for i in pri:
                if isinstance(i, str) and i in ids and i not in order:
                    order.append(i)
        if order:
            rank = {i: n for n, i in enumerate(order)}
            findings.sort(key=lambda f: rank.get(f["id"], len(rank)))
        overview = grounded_or(res.get("overview"), findings, overview)
    return {"findings": findings, "overview": overview}
