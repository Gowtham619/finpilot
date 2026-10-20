from __future__ import annotations

import json
import uuid

from ..engines.verify import recompute
from ..llm import LLMUnavailable


def inr(x):
    return f"₹{x:,.0f}"


def _claim(kind, df, ctx=None, **kw):
    v = recompute({"kind": kind, **kw}, df, ctx)
    return None if v is None else {"kind": kind, **kw, "value": round(v, 2)}


def _opening(df, forecast, ctx):
    """Opening balance for the weather claims: explicit ctx, else back out of the forecast balance."""
    if isinstance(ctx, dict) and ctx.get("opening") is not None:
        return ctx
    bal = (forecast or {}).get("balance")
    if isinstance(bal, (int, float)) and not isinstance(bal, bool):
        return {"opening": float(bal) - float(df["amount"].sum())}
    return None


def build_candidates(findings, df, forecast=None, goals=None, weather=None, ctx=None):
    """`weather` is a weather.report(); with it (and an opening balance, from `ctx` or the forecast
    balance) forecast_risk and storm_warning carry recomputable claims."""
    out = []
    ctx = _opening(df, forecast, ctx) if weather else None

    def add(trigger, action, fids, evidence, claims, monthly=0.0, one_off=0.0):
        out.append({"cid": f"c{len(out)}", "trigger_type": trigger, "action": action,
                    "finding_ids": fids, "evidence": evidence,
                    "claims": [c for c in claims if c], "est_monthly_savings": round(monthly, 2),
                    "est_one_off": round(one_off, 2)})

    for f in findings:
        t = f["type"]
        if t == "unused_subscription":
            add(t, f"Cancel {f['merchant']}", [f["id"]], f,
                [_claim("monthly_cost", df, merchant=f["merchant"]),
                 _claim("annual_cost", df, merchant=f["merchant"])], monthly=f["monthly_cost"])
        elif t == "category_spike":
            f = dict(f, excess=round(f["value"] - f["baseline"], 2))
            add(t, f"Cut back on {f['category']}", [f["id"]], f,
                [_claim("month_total", df, category=f["category"], month=f["month"]),
                 _claim("pct_change", df, category=f["category"], month=f["month"])],
                monthly=max(0.0, f["value"] - f["baseline"]))
        elif t == "duplicate_charge":
            add(t, f"Dispute duplicate charge at {f['merchant']}", [f["id"]], f,
                [_claim("txn_amount", df, txn_id=f["txn_id"])], one_off=f["amount"])
        elif t == "large_charge":
            add(t, f"Review large charge at {f['merchant']}", [f["id"]], f,
                [_claim("txn_amount", df, txn_id=f["txn_id"])])
        elif t == "behavior_trigger":
            add(t, f"Add friction to {f['category']} right after payday", [f["id"]], f, [])
    if forecast and forecast["summary"]["p_overdraft"] >= 0.10:
        s = forecast["summary"]
        ev = {**s, "horizon": forecast["horizon"]}
        claims = []
        if weather and ctx:
            h = weather["headline"]
            ev.update(horizon=weather["horizon"], buffer=weather["buffer"], p_any_breach=h["p_any_breach"], worst_day=h["worst_day"],
                      worst_p=h["worst_p"], n_storm_days=h["n_storm_days"])
            if weather["storms"]:
                ev.update(storm_start=weather["storms"][0]["start"], storm_end=weather["storms"][0]["end"])
            claims = [_claim("breach_probability", df, ctx, horizon=weather["horizon"])]
            if not claims[0]:
                claims = []
        add("forecast_risk", "Protect cash before the projected low point", ["forecast"], ev, claims)
    top = weather["storms"][0] if weather and weather.get("storms") else None
    if top and top["peak_p"] >= 0.25 and ctx:
        ev = {"storm": dict(top), "buffer": weather["buffer"], "horizon": weather["horizon"]}
        peak = _claim("storm_peak", df, ctx, date=top["peak_date"])
        claims = [peak]
        if peak and weather.get("umbrellas"):
            u = weather["umbrellas"][0]
            after = _claim("storm_peak_after_cut", df, ctx, date=top["peak_date"], category=u["category"],
                           cut_pct=u["cut_pct"])
            if after and after["value"] < peak["value"]:
                ev["umbrella"] = dict(u)
                claims.append(after)
        if peak:
            add("storm_warning", f"Storm warning {top['start']} to {top['end']}", ["weather"], ev, claims)
    for g in (goals or []):
        if not isinstance(g, dict) or "error" in g:
            continue
        if g.get("p_met", 1) < 0.5:
            add("goal_behind", f"Goal '{g['name']}' is falling behind", [f"goal:{g['name']}"], g, [])
    out.sort(key=lambda c: -(c["est_monthly_savings"] + c["est_one_off"]))
    for i, c in enumerate(out):
        c["cid"] = f"c{i}"
    return out


def _cv(rec, kind, default):
    """Claim value for `kind` if the rec carries that claim, so templates state exactly what is verified."""
    for c in rec.get("claims") or []:
        if isinstance(c, dict) and c.get("kind") == kind and isinstance(c.get("value"), (int, float)) \
                and not isinstance(c.get("value"), bool):
            return c["value"]
    return default


def template_text(rec):
    e, t = rec["evidence"], rec["trigger_type"]
    if t == "unused_subscription":
        return (f"Cancel {e['merchant']}: it costs {inr(_cv(rec, 'monthly_cost', e['monthly_cost']))}/month "
                f"({inr(_cv(rec, 'annual_cost', e['annual_cost']))}/year) and looks unused — {e['detail']}.")
    if t == "category_spike":
        cat = e['category'].replace('_', ' ')
        return (f"In {e['month']} {cat} spend was {inr(_cv(rec, 'month_total', e['value']))}, "
                f"{inr(e['excess'])} over your usual {inr(e['baseline'])} "
                f"({_cv(rec, 'pct_change', e['pct_change']):.0f}% above). "
                f"Bringing it back to normal would have saved about {inr(e['excess'])} that month.")
    if t == "duplicate_charge":
        return (f"{e['merchant']} charged {inr(_cv(rec, 'txn_amount', e['amount']))} twice within two days "
                f"(second charge {e['date']}). Dispute it.")
    if t == "large_charge":
        return (f"A {inr(_cv(rec, 'txn_amount', e['amount']))} charge at {e['merchant']} on {e['date']} "
                f"is far above your normal spend. Confirm it is yours.")
    if t == "behavior_trigger":
        return (f"{e['share_pct']:.0f}% of your {e['category'].replace('_', ' ')} spending lands in the "
                f"{e['window_days']} days after payday ({e['ratio']:.1f}x the normal rate). Delay those orders a few days.")
    if t == "storm_warning":
        st = e["storm"]
        txt = (f"Storm warning: {st['start']} to {st['end']}, peak {_cv(rec, 'storm_peak', st['peak_p']*100):.0f}% "
               f"chance of dropping below your safety buffer ({inr(e['buffer'])}).")
        u = e.get("umbrella")
        if u and any(c.get("kind") == "storm_peak_after_cut" for c in rec.get("claims") or []):
            txt += (f" Cutting {u['category'].replace('_', ' ')} by {u['cut_pct']}% lowers the peak to "
                    f"{_cv(rec, 'storm_peak_after_cut', 0):.0f}%.")
        return txt
    if t == "forecast_risk" and "buffer" in e:
        return (f"At the current pace there is a {e['p_overdraft']*100:.0f}% chance your balance goes negative "
                f"within {e['horizon']} days, first around {e['risk_day']}. Across all {e['horizon']} days "
                f"{_cv(rec, 'breach_probability', e['p_any_breach']*100):.0f}% of simulated futures dip below your "
                f"safety buffer ({inr(e['buffer'])}). Hold discretionary spending until then.")
    if t == "forecast_risk":
        return (f"At the current pace there is a {e['p_overdraft']*100:.0f}% chance your balance goes negative "
                f"within {e['horizon']} days, first around {e['risk_day']}. Hold discretionary spending until then.")
    if t == "goal_behind":
        return f"You have a {e['p_met']*100:.0f}% chance of meeting '{e['name']}' by its deadline. Increase contributions or move the date."
    return rec["action"]


SYSTEM = ("You are a financial advisor. For each candidate write at most two sentences of specific advice "
          "using ONLY numbers present in its evidence or claims. Write full numbers (₹1500, not 1.5k). "
          "Return JSON {\"recommendations\":[{\"cid\":str,\"text\":str,\"claims\":[...]}]} where claims "
          "copies the candidate's claim objects, with 'value' set to the number you state in the text.")


_ID_KEYS = ("kind", "merchant", "category", "month", "txn_id", "date", "horizon", "cut_pct")


def _identity(claim):
    return tuple(str(claim.get(k)) for k in _ID_KEYS)


def _merge_claims(engine_claims, llm_claims):
    """The engine decides WHAT is verified; the LLM may only restate the value of a matching claim."""
    merged = [dict(c) for c in engine_claims]
    if not isinstance(llm_claims, list):
        return merged
    for lc in llm_claims:
        if not isinstance(lc, dict):
            continue
        v = lc.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
            continue
        for c in merged:
            if _identity(c) == _identity(lc):
                c["value"] = v
                break
    return merged


def advise(candidates, llm=None, model_name=""):
    by_cid = {}
    if llm is not None and candidates:
        try:
            res = llm.json(SYSTEM, json.dumps(
                [{k: c[k] for k in ("cid", "action", "evidence", "claims")} for c in candidates], default=str))
            by_cid = {r["cid"]: r for r in res.get("recommendations", []) if "cid" in r and "text" in r}
        except (LLMUnavailable, ValueError, AttributeError, TypeError):
            by_cid = {}
    recs = []
    for c in candidates:
        r = dict(c, id=f"rec-{uuid.uuid4().hex[:8]}", verified=False, verifier_notes="",
                 model=model_name if c["cid"] in by_cid else "template")
        if c["cid"] in by_cid:
            r["text"] = str(by_cid[c["cid"]]["text"])
            r["claims"] = _merge_claims(c["claims"], by_cid[c["cid"]].get("claims"))
        else:
            r["text"] = template_text(r)
        r["verification"] = "recomputed" if r["claims"] else "evidence_only"
        recs.append(r)
    return recs
