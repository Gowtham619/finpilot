"""Self-evaluation. Run: python -m eval.run_eval [--no-llm]. Writes eval/results.json and eval/results.md."""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd

from eval import metrics as M
from eval.generate import PERSONAS, load_persona
from eval.generate_messy import OOD_PERSONAS, load_messy
from finpilot import graph
from finpilot.agents.categorizer import categorize_df
from finpilot.engines import analyze as A
from finpilot.engines import forecast as F
from finpilot.engines.categorize import normalize_merchant
from finpilot.llm import LLM, LLMUnavailable
from finpilot.trace import Trace

OUT = pathlib.Path(__file__).parent


def _with_truth(persona):
    df = persona.df.copy()
    df["merchant"] = df["description"].map(normalize_merchant)
    df["category"] = persona.truth.categories
    return df


def eval_categorization(persona, llm):
    truth = persona.truth.categories
    out = {}
    for name, use in (("rules_only", None), ("hybrid", llm)):
        if name == "hybrid" and llm is None:
            out[name] = None
            continue
        cat = categorize_df(persona.df, use)
        pred = cat["category"]
        out[name] = {"accuracy": round(M.accuracy(truth, pred), 4), "macro_f1": round(M.macro_f1(truth, pred), 4),
                     "coverage": round(float((pred != "uncategorized").mean()), 4)}
        if name == "hybrid":  # rows the LLM actually labelled (0 means hybrid silently equals rules-only)
            out[name]["n_llm_labeled"] = int((cat["category_source"] == "llm").sum())
    return out


ORIGINAL_SUBSCRIPTIONS = {"NETFLIX", "HOTSTAR", "SPOTIFY", "CULT FIT GYM"}


def eval_detection(persona, subscription_truth=ORIGINAL_SUBSCRIPTIONS):
    """subscription_truth / payday_trigger_category=None are parameterised only so the same code scores
    the second generator; defaults reproduce the original scoring exactly."""
    df, t = _with_truth(persona), persona.truth
    rec = {r["merchant"] for r in A.detect_recurring(df)}
    subs = A.detect_subscriptions(df)
    return {
        "large_charge": M.precision_recall({f["txn_id"] for f in A.txn_outliers(df, rec)}, t.anomaly_ids),
        "duplicate": M.precision_recall({f["txn_id"] for f in A.duplicate_charges(df)}, t.duplicate_ids),
        "category_spike": M.precision_recall({(f["month"], f["category"]) for f in A.category_spikes(df)}, {t.spike}),
        "subscriptions": M.precision_recall({s["merchant"] for s in subs},
                                            set(subscription_truth)),
        "unused_subscriptions": M.precision_recall({u["merchant"] for u in A.unused_subscriptions(df, subs)}, t.unused_subs),
        "payday_trigger": M.precision_recall({f["category"] for f in A.behavior_triggers(df, rec)},
                                             {t.payday_trigger_category} if t.payday_trigger_category else set()),
    }


QUANTILES = (0.1, 0.5, 0.9)
BASELINES = ("flat_balance", "seasonal_naive", "trailing_mean")


def _align(model, cutoff):
    """Safeguard: force the forecast to start on the cutoff day. fit(train) already yields
    last_date == cutoff - 1 day whenever the train data ends on that day; this only matters
    if the final day before the cutoff has no transaction."""
    model.last_date = cutoff - pd.Timedelta(days=1)


def _mape(pred, actual):
    return float(np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1.0)))


def _pinball(fc_by_q, actual):
    return float(np.mean([np.mean(np.maximum(q * (actual - f), (q - 1) * (actual - f)))
                          for q, f in fc_by_q.items()]))


def baseline_paths(bal0, prev_flows, n_days, trailing_mean):
    """Baseline balance paths over the n_days held-out days.
    flat_balance: stays at bal0. seasonal_naive: last month's daily net flows replayed
    (zero-padded/truncated to n_days). trailing_mean: bal0 + trailing mean daily net flow * day."""
    prev = np.zeros(n_days)
    k = min(n_days, len(prev_flows))
    prev[:k] = np.asarray(prev_flows, dtype=float)[:k]
    return {"flat_balance": np.full(n_days, float(bal0)),
            "seasonal_naive": bal0 + np.cumsum(prev),
            "trailing_mean": bal0 + trailing_mean * np.arange(1, n_days + 1)}


def _daily_net(df, start, end):
    days = pd.date_range(start, end)
    sub = df[(df["date"] >= start) & (df["date"] <= end + pd.Timedelta(hours=23, minutes=59))]
    return sub.groupby(sub["date"].dt.normalize())["amount"].sum().reindex(days, fill_value=0.0).to_numpy()


def forecast_origins(df, origins=3):
    """Monthly forecast origins shared by eval_forecast and the LLM-only ablation: the first day
    of each of the last `origins` calendar months of data."""
    last_month = df["date"].max().to_period("M")
    return [(last_month - k).to_timestamp() for k in range(origins)]


def eval_forecast(persona, origins=3):
    df = _with_truth(persona)
    months = df["date"].dt.to_period("M").nunique()
    monthly_spend = float(-df.loc[df["amount"] < 0, "amount"].sum() / months)
    names = ("model",) + BASELINES
    acc = {n: {"ape": [], "mape": [], "mae": [], "pin": [], "cov": []} for n in names}
    for cutoff in forecast_origins(df, origins):
        end = (cutoff + pd.offsets.MonthEnd(0)).normalize()
        train = df[df["date"] < cutoff]
        model, bal0 = F.fit(train), F.current_balance(train, persona.opening_balance)
        _align(model, cutoff)
        days = pd.date_range(cutoff, end)
        actual = bal0 + np.cumsum(_daily_net(df, cutoff, end))
        true_net = actual[-1] - bal0
        r = F.monte_carlo(model, bal0, len(days), 1000)
        prev_start = cutoff - pd.offsets.MonthBegin(1)
        prev = _daily_net(train, prev_start, cutoff - pd.Timedelta(days=1))
        trailing = train[train["date"] >= cutoff - pd.Timedelta(days=30)]["amount"].sum() / 30
        paths = baseline_paths(bal0, prev, len(days), trailing)
        fcs = {"model": (r.p50, {0.1: r.p10, 0.5: r.p50, 0.9: r.p90})}
        for n, p in paths.items():
            fcs[n] = (p, {q: p for q in QUANTILES})
        for n, (pt, qs) in fcs.items():
            acc[n]["ape"].append(abs((pt[-1] - bal0) - true_net) / max(abs(true_net), 1.0))
            acc[n]["mape"].append(_mape(pt, actual))
            acc[n]["mae"].append(abs(pt[-1] - actual[-1]) / monthly_spend)
            acc[n]["pin"].append(_pinball(qs, actual) / monthly_spend)
        acc["model"]["cov"].append(float(np.mean((actual >= r.p10) & (actual <= r.p90))))
    out = {}
    for n, v in acc.items():
        out[n] = {"net_flow_ape": round(float(np.mean(v["ape"])), 4), "balance_mape": round(float(np.mean(v["mape"])), 4),
                  "mae_vs_monthly_spend": round(float(np.mean(v["mae"])), 4),
                  "pinball_vs_monthly_spend": round(float(np.mean(v["pin"])), 4),
                  "band_coverage": round(float(np.mean(v["cov"])), 4) if v["cov"] else None}
    out["_meta"] = {"origins": origins}
    return out


def eval_calibration(personas, horizon=30, drop=0.10):
    """Event: balance falls more than `drop` below its starting level within `horizon` days."""
    probs, outcomes = [], []
    for p in personas:
        df = _with_truth(p)
        for cutoff in pd.date_range(df["date"].min() + pd.Timedelta(days=100),
                                    df["date"].max() - pd.Timedelta(days=horizon), freq="7D"):
            train = df[df["date"] < cutoff]
            model, bal0 = F.fit(train), F.current_balance(train, p.opening_balance)
            _align(model, cutoff)
            r = F.monte_carlo(model, bal0, horizon, 500)
            probs.append(float((r.paths.min(axis=1) < bal0 * (1 - drop)).mean()))
            path = bal0 + np.cumsum(_daily_net(df, cutoff, cutoff + pd.Timedelta(days=horizon - 1)))
            outcomes.append(int(path.min() < bal0 * (1 - drop)))
    base = float(np.mean(outcomes))
    base_brier = base * (1 - base)
    br = M.brier(probs, outcomes)
    idx = np.minimum((np.array(probs) * 5).astype(int), 4)
    bins = []
    for b in range(5):
        m = idx == b
        bins.append({"range": f"{b / 5:.1f}-{(b + 1) / 5:.1f}", "count": int(m.sum()),
                     "mean_pred": round(float(np.array(probs)[m].mean()), 4) if m.any() else None,
                     "observed": round(float(np.array(outcomes)[m].mean()), 4) if m.any() else None})
    return {"n": len(probs), "mean_pred": round(float(np.mean(probs)), 4), "observed": round(base, 4),
            "brier": round(br, 4), "base_brier": round(base_brier, 4),
            "brier_skill": round(1 - br / base_brier, 4) if base_brier > 0 else None, "bins": bins}


def eval_ood(personas):
    """Second generator, engines and thresholds unchanged, same metric code paths as the main sections."""
    cats = [eval_categorization(p, None)["rules_only"] for p in personas.values()]
    dets = [eval_detection(p, p.subscription_merchants) for p in personas.values()]
    fcs = [eval_forecast(p) for p in personas.values()]
    n_orig = fcs[0]["_meta"]["origins"]
    for x in fcs:
        x.pop("_meta")
    planted = {k: sorted({p.planted[k] for p in personas.values()}) for k in ("duplicates", "large_charges", "spike")}
    planted = {k: v[0] if len(v) == 1 else v for k, v in planted.items()}
    planted["unused_subs"] = [len(p.truth.unused_subs) for p in personas.values()]
    planted["refunds"] = [p.planted["refunds"] for p in personas.values()]
    return {"personas": list(personas), "categorization": {"rules_only": _avg(cats)}, "detection": _avg(dets),
            "forecast": _avg(fcs),
            "forecast_meta": {"n_personas": len(personas), "n_origins_per_persona": n_orig,
                              "total_months": len(personas) * n_orig},
            "planted": planted}


def ood_notes(res):
    """Methodology bullets for the OOD block, built from the numbers actually measured."""
    o, d = res["ood"], res["ood"]["detection"]
    oc = o["categorization"]["rules_only"]
    notes = [
        "Out-of-distribution check: a second generator (eval/generate_messy.py) with different merchants, noisy UPI/NEFT/IMPS/card "
        "description formats, refunds, cash, EMIs, an annual subscription, a price increase and a travel spike was scored with the "
        "engines and thresholds unchanged. The same author wrote both generators and knows the engines, so this is a robustness "
        "check, not independent validation; it is not evidence about real bank data.",
        f"OOD behaviour: rules-only categorization accuracy fell from {res['categorization']['rules_only']['accuracy']['mean']} to "
        f"{oc['accuracy']['mean']} (coverage {oc['coverage']['mean']}); detection and forecast were scored on oracle categories, so the "
        "categorizer drop does not feed into them. The oracle-category scoring flatters the OOD detection and forecast numbers.",
        "OOD refunds are positive credits whose truth category is shopping; the rules label every positive amount as income, so "
        "refunds are always miscategorized by the rules-only categorizer. Cash withdrawals and EMIs have truth category 'other', "
        "which the rules never emit (they return uncategorized).",
        f"OOD payday_trigger has no planted target (truth set empty), so precision {_fmt(d['payday_trigger']['precision'])} means "
        "any finding is a false positive. OOD subscription truth is the monthly subscriptions only; the annual one is not expected.",
        "OOD generator provenance: spend level and opening balances were set so balances stay plausible, and the share of "
        "mainstream merchants the rules know (Swiggy, Zomato, Uber, Amazon, BigBasket) was raised once after the first draft scored "
        "rules-only accuracy below the 0.3 sanity floor. Only categorization accuracy was looked at when doing so. After a first "
        "scoring run two truth-labelling bugs in the generator were fixed because they scored correct detections as misses (subscription "
        "truth now uses the engine's normalised merchant strings; a same-day duplicate pair whose 'second row' was ambiguous now sits on "
        "consecutive days). No engine, threshold or planted target was changed to move a result.",
    ]
    return notes


def eval_grounding(persona, llm):
    tr = Trace()
    g = graph.build_graph(llm, tr)
    out = g.invoke({"run_id": "eval", "profile": "eval", "request": {"type": "pipeline"}, "df": persona.df.copy(),
                    "opening": persona.opening_balance, "corrections": {}, "goals": []})
    evs = tr.since("eval")
    llm_ids = set()
    for e in evs:
        if e["agent"] == "advisor" and e["kind"] == "result":
            llm_ids |= set((e.get("data") or {}).get("llm_written", []))
    # first-pass verdicts for LLM-written recs only: a deterministic template passing its own audit
    # says nothing about grounding
    kinds = [e["kind"] for e in evs if e["agent"] == "verifier" and e["kind"] in ("ok", "caught")
             and (e.get("data") or {}).get("rec") in llm_ids]
    ok, caught = kinds.count("ok"), kinds.count("caught")
    n = ok + caught
    return {"recs": n, "first_pass_ok": ok, "first_pass_caught": caught,
            "first_pass_grounding_rate": round(ok / n, 4) if n else None,
            "template_recs": len(out["recs"]) + len(out.get("dropped") or []) - len(llm_ids),
            "final_verified": sum(r["verified"] for r in out["recs"])}


def _avg(dicts):
    """Aggregate per-persona results: every numeric leaf becomes {mean, min, max, values}."""
    if not dicts or dicts[0] is None:
        return None
    if isinstance(dicts[0], dict):
        return {k: _avg([d[k] for d in dicts]) for k in dicts[0]}
    vals = [float(x) for x in dicts if x is not None]
    if not vals:
        return None
    return {"mean": round(float(np.mean(vals)), 4), "min": round(min(vals), 4), "max": round(max(vals), 4),
            "values": [round(v, 4) for v in vals]}


def _fmt(x):
    if x is None:
        return "n/a"
    if not isinstance(x, dict):
        return str(x)
    return f"{x['mean']} ({x['min']}\u2013{x['max']})"


METHODOLOGY = [
    "Detection, forecast and calibration engines are scored on oracle ground-truth categories to isolate them from the categorizer, so end-to-end performance is bounded by the categorization accuracy.",
    "All data is synthetic, produced by one generator that also planted the detection targets.",
    "The personas share that generator, so results are not independent samples; min-max shows spread across only three personas.",
    "category_spike precision is capped at 1/3 by construction (max_findings=3, one planted spike per persona); it is reported as precision@3 plus recall.",
    "Macro-F1 averages over labels present in the ground truth only.",
    "precision_recall returns 1.0 for an empty prediction set by convention.",
    "Calibration origins overlap (7-day stride, 30-day windows) and sit mid-month rather than month-aligned, so the effective n is far smaller than n.",
    "A flat-balance forecast scores exactly 1.0 on net-flow APE, so that metric is reported only for completeness.",
    "Pinball loss and month-end MAE are divided by the persona's mean monthly spend; baselines are point forecasts used as all three quantiles.",
    "Advice backtest: the headline is cancel savings, the deterministic post-cutoff spend at merchants the system recommended cancelling. It assumes the user cancelled every flagged subscription on the cutoff day.",
    "Backtest excess is a separate, noisy upper bound: for each flagged category spike, post-cutoff monthly spend above the pre-cutoff median baseline (prorated by days present for partial months, net of cancelled merchants in that category to avoid double counting). It nets nothing off for months below baseline, so it only ever adds, and it is not a forecast of behaviour.",
    "Each persona has only 1-2 cancellations, so cancel precision and recall (reported as k/n counts) are anecdotal.",
    "The LLM-only baselines receive a compressed monthly merchant table, not raw transactions; this is a design choice.",
    "The LLM-only advice is audited with no engine evidence, so any derived number in its text that is not in its claims counts as ungrounded; this inflates the gap relative to the hybrid, whose recommendations carry engine evidence.",
    "The LLM-only forecast uses the same 3 monthly origins per persona as the model (monthly net-flow history before the origin -> that month's net flow). A net-flow APE of 1.0 equals the flat-balance forecast.",
    "The ablation is a single run at temperature 0 over 3 synthetic personas (only when run with an LLM); differences are indicative, not statistically tested.",
]


def render_ood(res):
    o = res.get("ood")
    if not o:
        return []
    c, d, f, m, pl = o["categorization"]["rules_only"], o["detection"], o["forecast"], o["forecast_meta"], o["planted"]
    rows = ["", "## Out-of-distribution check (second generator, engines unchanged)",
            f"Personas: {', '.join(o['personas'])}. Same author wrote both generators: a robustness check, not independent validation. "
            f"Planted per persona: {pl['duplicates']} duplicates, {pl['large_charges']} large charges, {pl['spike']} category spike (travel), "
            f"unused subscriptions {pl['unused_subs']}, refunds {pl['refunds']}. Cells show mean (min\u2013max) across personas.", "",
            "| Categorization (rules-only) | Accuracy | Macro-F1 | Coverage |", "|---|---|---|---|",
            f"| rules_only | {_fmt(c['accuracy'])} | {_fmt(c['macro_f1'])} | {_fmt(c['coverage'])} |", "",
            "| Detection | Precision | Recall |", "|---|---|---|"]
    rows += [f"| {k} | {_fmt(v['precision'])} | {_fmt(v['recall'])} |" for k, v in d.items()]
    rows += ["", f"| Forecast ({m['n_personas']} personas x {m['n_origins_per_persona']} origins = {m['total_months']} months) | Balance MAPE | Month-end MAE / monthly spend | Pinball / monthly spend | p10-p90 coverage (nominal 0.80) |",
             "|---|---|---|---|---|"]
    for n in ("model", "flat_balance", "seasonal_naive", "trailing_mean"):
        v = f[n]
        rows.append(f"| {n} | {_fmt(v['balance_mape'])} | {_fmt(v['mae_vs_monthly_spend'])} | {_fmt(v['pinball_vs_monthly_spend'])} "
                    f"| {_fmt(v['band_coverage']) if v['band_coverage'] else '-'} |")
    return rows


def render_md(res):
    c, f, d, cal = res["categorization"], res["forecast"], res["detection"], res["calibration"]
    meta = res["forecast_meta"]
    rows = ["# Evaluation results", "", f"Personas: {', '.join(res['personas'])} (synthetic, known ground truth)", "",
            "Cells show mean (min\u2013max) across personas.", "",
            "## Categorization", "| Method | Accuracy | Macro-F1 | Coverage |", "|---|---|---|---|"]
    for name in ("rules_only", "hybrid"):
        m = c.get(name)
        rows.append(f"| {name} | " + (" | ".join(_fmt(m[k]) for k in ("accuracy", "macro_f1", "coverage")) if m else "n/a | n/a | n/a") + " |")
    n_llm = (c.get("hybrid") or {}).get("n_llm_labeled")
    if n_llm is not None:
        rows += ["", f"Hybrid rows labelled by the LLM per persona: {_fmt(n_llm)}."]
    rows += ["", "## Detection (precision / recall)", "| Task | Precision | Recall |", "|---|---|---|"]
    rows += [f"| {k} | {_fmt(v['precision'])} | {_fmt(v['recall'])} |" for k, v in d.items()]
    rows += ["", f"## Forecast (held-out months; {meta['n_personas']} personas x {meta['n_origins_per_persona']} origins = {meta['total_months']} months)",
             "| Method | Balance MAPE | Month-end MAE / monthly spend | Pinball / monthly spend | Net-flow APE | p10-p90 coverage (nominal 0.80) |",
             "|---|---|---|---|---|---|"]
    labels = {"model": "model", "flat_balance": "flat_balance", "seasonal_naive": "seasonal_naive",
              "trailing_mean": "trailing-30d-mean (spikes on salary)"}
    for n, lab in labels.items():
        v = f[n]
        rows.append(f"| {lab} | {_fmt(v['balance_mape'])} | {_fmt(v['mae_vs_monthly_spend'])} | {_fmt(v['pinball_vs_monthly_spend'])} "
                    f"| {_fmt(v['net_flow_ape'])} | {_fmt(v['band_coverage']) if v['band_coverage'] else '-'} |")
    rows += ["", f"Coverage is over n={meta['total_months']} held-out months (daily points within each). A flat-balance forecast scores exactly 1.0 on net-flow APE, so it is reported only for completeness.",
             "", "## Calibration",
             f"Event: balance falls >10% below its starting level within 30 days. n={cal['n']} (overlapping origins), mean predicted {cal['mean_pred']}, "
             f"observed {cal['observed']}, Brier {cal['brier']}, constant-base-rate Brier {cal['base_brier']}, Brier skill score {_fmt(cal['brier_skill'])}.",
             "", "| Predicted bin | Count | Mean predicted | Observed |", "|---|---|---|---|"]
    rows += [f"| {b['range']} | {b['count']} | {b['mean_pred'] if b['mean_pred'] is not None else '-'} | "
             f"{b['observed'] if b['observed'] is not None else '-'} |" for b in cal["bins"]]
    rows += render_ood(res)
    ab = res.get("ablation")
    g0 = res.get("grounding")
    rows += ["", "## Ablation: LLM-only vs rules-only vs hybrid"]
    if ab:
        ac = ab["categorization"]
        la = ab["advice"]["llm_only"]
        fa = ab["forecast"]["llm_only"]
        if not g0:
            caught = "n/a (--no-llm)"
        elif not g0["recs"]:
            caught = "n/a: no LLM-written recs"
        else:
            caught = (f"{round(g0['first_pass_caught'] / g0['recs'], 4)} of {g0['recs']} recs caught (priya only); "
                      f"final output 0 flagged by construction (only recs passing audit are kept)")
        rows += ["| Task | LLM-only | Rules-only | Hybrid (FinPilot) |", "|---|---|---|---|",
                 f"| Categorization accuracy (n personas: {ac['hybrid']['n_used']}) | {_fmt(ac['llm_only']['accuracy'])} | {_fmt(ac['rules_only']['accuracy'])} | {_fmt(ac['hybrid']['accuracy'])} |",
                 f"| Forecast net-flow APE ({fa.get('n_used')} origins; 1.0 = balance stays flat) | {_fmt(fa['net_flow_ape'])} | flat-balance: {_fmt(f['flat_balance']['net_flow_ape'])} | model: {_fmt(f['model']['net_flow_ape'])} |",
                 f"| LLM-only: share of recs with ungrounded/incorrect numbers ({la['recs']} recs) | {_fmt(la['hallucination_rate'])} | - | - |",
                 f"| FinPilot hybrid: first-pass share caught by the Verifier | - | - | {caught} |"]
    else:
        rows.append("n/a (--no-llm)")
    bt = res.get("backtest")
    if bt:
        rows += ["", "## Advice backtest (last 90 days; assumes every recommendation was followed)",
                 "Headline is cancel savings (deterministic). Excess is an estimated excess over median (noisy upper bound), shown separately.", "",
                 "| Persona | Cancel savings (headline) | % of spend | Estimated excess over median (noisy upper bound) | Cancelled (correct/total) | Unused in truth | Recall |",
                 "|---|---|---|---|---|---|---|"]
        rows += [f"| {n} | \u20b9{b['cancel_savings']:,.0f} | {b['cancel_pct_of_spend'] * 100:.1f}% | \u20b9{b['excess_savings']:,.0f} "
                 f"({b['excess_pct_of_spend'] * 100:.1f}%) | {b['n_correct']}/{b['n_cancelled']} | {b['n_unused_truth']} | "
                 f"{b['n_correct']}/{b['n_unused_truth']} |" for n, b in bt.items()]
    rows += ["", "## Real-data check"]
    rd = res.get("real_data")
    if isinstance(rd, dict):
        rows.append("REAL data (not synthetic), from eval/real_results.json:")
        rows += [f"- {k}: {v}" for k, v in rd.items()]
    else:
        rows.append(rd or "not run: no dataset supplied (see eval/real_data.py)")
    g = res.get("grounding")
    if not g:
        gline = "n/a (--no-llm)"
    elif not g["recs"]:
        gline = f"n/a: no LLM-written recs (all were templates), final verified {g['final_verified']}"
    else:
        gline = (f"first-pass grounding rate {g['first_pass_grounding_rate']} ({g['first_pass_caught']} caught of "
                 f"{g['recs']} LLM-written recs), final verified {g['final_verified']}")
    rows += ["", "## Grounding", gline,
             "", "## Methodology and limitations"]
    rows += [f"- {m}" for m in res["methodology"]]
    return "\n".join(rows)


def preflight(llm):
    """Fail loudly instead of silently recording rules-only numbers as 'hybrid'."""
    try:
        res = llm.json('Return {"ok":true}', "ping")
    except LLMUnavailable as e:
        raise SystemExit(f"LLM preflight failed: {e}. Set a valid MESH_API_KEY (and FINPILOT_MODEL), "
                         f"or run `python -m eval.run_eval --no-llm`.")
    if not isinstance(res, dict):
        raise SystemExit("LLM preflight failed: the model did not return a JSON object.")
    return res


def main(argv=None, llm=None):
    argv = argv if argv is not None else sys.argv[1:]
    use_llm = "--no-llm" not in argv
    if use_llm:
        llm = llm or LLM()
        preflight(llm)
    else:
        llm = None
    personas = {n: load_persona(n) for n in PERSONAS}
    res = {"personas": list(personas), "model": getattr(llm, "model", None)}
    cats = [eval_categorization(p, llm) for p in personas.values()]
    res["categorization"] = {"rules_only": _avg([c["rules_only"] for c in cats]),
                             "hybrid": _avg([c["hybrid"] for c in cats]) if use_llm else None}
    res["detection"] = _avg([eval_detection(p) for p in personas.values()])
    fcs = [eval_forecast(p) for p in personas.values()]
    n_orig = fcs[0]["_meta"]["origins"]
    for x in fcs:
        x.pop("_meta")
    res["forecast"] = _avg(fcs)
    res["forecast_meta"] = {"n_personas": len(personas), "n_origins_per_persona": n_orig,
                            "total_months": len(personas) * n_orig}
    res["calibration"] = eval_calibration(list(personas.values()))
    res["methodology"] = list(METHODOLOGY)
    res["ood"] = eval_ood({n: load_messy(n) for n in OOD_PERSONAS})
    res["methodology"] += ood_notes(res)
    res["grounding"] = eval_grounding(personas["priya"], llm) if use_llm else None
    from eval import backtest  # lazy: backtest/ablation import from this module
    res["backtest"] = {n: backtest.run(p) for n, p in personas.items()}
    res["ablation"] = None
    rr = OUT / "real_results.json"
    res["real_data"] = json.loads(rr.read_text()) if rr.exists() else "not run: no dataset supplied (see eval/real_data.py)"
    if use_llm:
        from eval import ablation
        res["ablation"] = ablation.run(personas, llm)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    (OUT / "results.md").write_text(render_md(res))
    print(render_md(res))
    return res


if __name__ == "__main__":
    main()
