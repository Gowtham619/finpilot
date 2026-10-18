from __future__ import annotations

import json
import math
import re

import pandas as pd

from ..config import CATEGORIES
from ..engines import forecast as F
from ..engines import simulate as S
from ..engines.analyze import detect_subscriptions
from .common import fmt_bal, grounded_or, inr

_MONTH = re.compile(r"\d{4}-(0[1-9]|1[0-2])")
MAX_AMOUNT = 1e12


class BadArgs(ValueError):
    pass


def _month(v):
    if not isinstance(v, str) or not _MONTH.fullmatch(v):
        raise BadArgs("month must look like YYYY-MM")
    return v


def _category(v):
    if v not in CATEGORIES:
        raise BadArgs("unknown category")
    return v


def _int(lo, hi):
    def f(v):
        if isinstance(v, bool):
            raise BadArgs("expected a whole number")
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise BadArgs("expected a whole number")
        if not math.isfinite(x):
            raise BadArgs("expected a whole number")
        return int(min(max(x, lo), hi))
    return f


def _amount(v):
    if isinstance(v, bool):
        raise BadArgs("amount must be a number")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise BadArgs("amount must be a number")
    if not math.isfinite(x) or x < 0 or x > MAX_AMOUNT:
        raise BadArgs("amount must be a reasonable non-negative number")
    return x


def _date(v):
    try:
        t = pd.Timestamp(v)
    except (TypeError, ValueError):
        raise BadArgs("date must look like YYYY-MM-DD")
    if pd.isna(t):
        raise BadArgs("date must look like YYYY-MM-DD")
    return str(t.date())


# tool -> {arg name: coercer}; required args have no default in the tool
_SPECS = {
    "month_total": {"category": _category, "month": _month},
    "top_merchants": {"n": _int(1, 20), "month": _month},
    "list_subscriptions": {},
    "forecast": {"horizon": _int(1, 365)},
    "afford": {"amount": _amount, "date": _date},
    "goal_status": {},
}
_REQUIRED = {"month_total": {"category", "month"}, "afford": {"amount"}}


def _clean_args(name, args):
    spec = _SPECS[name]
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise BadArgs("arguments must be an object")
    out = {}
    for k, v in args.items():
        if k not in spec:
            raise BadArgs("unexpected argument")
        if v is None:
            continue
        out[k] = spec[k](v)
    if _REQUIRED.get(name, set()) - set(out):
        raise BadArgs("missing required argument")
    return out


def _tools(df, model, balance, goals):
    def month_total(category, month):
        m = df[(df["category"] == category) & (df["amount"] < 0) & (df["date"].dt.to_period("M").astype(str) == month)]
        return {"category": category, "month": month, "total": round(float(-m["amount"].sum()), 2), "n": int(len(m))}

    def top_merchants(n=5, month=None):
        d = df[df["amount"] < 0]
        if month:
            d = d[d["date"].dt.to_period("M").astype(str) == month]
        s = (-d.groupby("merchant")["amount"].sum()).sort_values(ascending=False).head(int(n))
        return {"month": month, "merchants": [{"merchant": k, "total": round(float(v), 2)} for k, v in s.items()]}

    def list_subscriptions():
        subs = detect_subscriptions(df)
        return {"subscriptions": [{"merchant": s["merchant"], "monthly_cost": s["monthly_cost"]} for s in subs],
                "total_monthly": round(sum(s["monthly_cost"] for s in subs), 2)}

    def forecast(horizon=30):
        h = int(horizon)
        return F.monte_carlo(model, balance, h).summary() | {"horizon": h, "balance": round(balance, 2)}

    def afford(amount, date=None):
        on = date or str((model.last_date + pd.Timedelta(days=1)).date())
        try:
            r = S.simulate_purchase(model, balance, float(amount), on)
        except ValueError as e:  # date outside the forecast window
            return {"error": str(e)}
        return {"amount": float(amount), "date": on, "overdraft_risk_before": r["baseline"]["p_overdraft"],
                "overdraft_risk_after": r["with_purchase"]["p_overdraft"],
                "median_balance_after": r["with_purchase"]["end_p50"]}

    def goal_status():
        out = []
        for g in goals:
            name = g.get("name", "?") if isinstance(g, dict) else "?"
            try:
                st = F.goal_status(model, balance, float(g["target"]), pd.Timestamp(g["deadline"]))
                out.append({"name": g["name"], "target": g["target"], "deadline": str(g["deadline"]), **st})
            except Exception as e:
                out.append({"name": name, "error": f"invalid goal: {type(e).__name__}"})
        return {"goals": out}

    return {"month_total": month_total, "top_merchants": top_merchants, "list_subscriptions": list_subscriptions,
            "forecast": forecast, "afford": afford, "goal_status": goal_status}


def _run_tool(tools, name, args):
    """Validate args, then run. Invalid args fall back to defaults only for tools whose args are all optional;
    any other failure becomes an {"error": ...} result. Never raises."""
    try:
        clean = _clean_args(name, args)
    except BadArgs as e:
        if _REQUIRED.get(name):
            return {"error": f"I need valid arguments for {name.replace('_', ' ')}: {e}."}
        clean = {}
    except Exception:
        clean = {}
    try:
        return tools[name](**clean)
    except Exception as e:
        return {"error": f"{name.replace('_', ' ')} failed ({type(e).__name__})"}


TOOL_SYSTEM = ("Choose a tool to answer the user's money question. Tools: month_total(category, month 'YYYY-MM'), "
               "top_merchants(n, month), list_subscriptions(), forecast(horizon), afford(amount, date), goal_status(). "
               "Categories: rent, groceries, dining, food_delivery, transport, utilities, subscriptions, shopping, "
               "entertainment, health, travel. Return JSON {\"tool\": name, \"args\": {...}}.")
ANSWER_SYSTEM = ("Answer the question using ONLY the tool results. Be concise, use full numbers like ₹1500. "
                 "Return JSON {\"text\": str}.")


def _heuristic(q):
    ql = q.lower()
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", q) if x.replace(",", "").replace(".", "")]
    if "afford" in ql or "buy" in ql:
        return "afford", {"amount": nums[0] if nums else 0}
    if "subscri" in ql:
        return "list_subscriptions", {}
    if "goal" in ql:
        return "goal_status", {}
    if "top" in ql or "merchant" in ql or "where" in ql:
        return "top_merchants", {}
    return "forecast", {"horizon": 30}


def _fallback_text(name, res):
    if "error" in res:
        return f"I couldn't work that out: {res['error']}"
    if name == "list_subscriptions":
        if not res["subscriptions"]:
            return "I found no recurring subscriptions in your transactions."
        subs = ", ".join(f"{s['merchant']} ({inr(s['monthly_cost'])})" for s in res["subscriptions"])
        return f"You pay for {len(res['subscriptions'])} subscriptions: {subs}. Total {inr(res['total_monthly'])}/month."
    if name == "afford":
        return (f"Buying {inr(res['amount'])} on {res['date']} moves your overdraft risk from "
                f"{res['overdraft_risk_before']*100:.0f}% to {res['overdraft_risk_after']*100:.0f}%, "
                f"leaving a median balance of {fmt_bal(res['median_balance_after'])} at the end of the window.")
    if name == "month_total":
        return f"You spent {inr(res['total'])} on {res['category'].replace('_', ' ')} in {res['month']}."
    if name == "top_merchants":
        if not res["merchants"]:
            return "I found no spending for that period."
        return "Top merchants: " + ", ".join(f"{m['merchant']} ({inr(m['total'])})" for m in res["merchants"]) + "."
    if name == "forecast":
        return (f"In {res['horizon']} days your median balance is {fmt_bal(res['end_p50'])} "
                f"with a {res['p_overdraft']*100:.0f}% overdraft risk.")
    if name == "goal_status":
        if not res["goals"]:
            return "You have no goals set yet."
        parts = []
        for g in res["goals"]:
            if "p_met" in g:
                parts.append(f"{g['name']}: {g['p_met']*100:.0f}% chance of reaching {inr(g['target'])} by {g['deadline']}")
            else:
                parts.append(f"{g['name']}: could not be evaluated")
        return "Goals - " + "; ".join(parts) + "."
    return "Done."


def answer(question, df, model, balance, goals, llm=None):
    tools = _tools(df, model, balance, goals)
    name, args = None, {}
    if llm is not None:
        try:
            pick = llm.json(TOOL_SYSTEM, question)
            if isinstance(pick, dict) and isinstance(pick.get("tool"), str) and pick["tool"] in tools:
                name, args = pick["tool"], pick.get("args")
        except Exception:
            pass
    if name is None:
        name, args = _heuristic(str(question))
    result = _run_tool(tools, name, args)
    fallback = _fallback_text(name, result)
    text = fallback
    if llm is not None:
        try:
            out = llm.json(ANSWER_SYSTEM, json.dumps({"question": question, "tool": name, "result": result}, default=str))
            if isinstance(out, dict) and "error" not in result:
                text = grounded_or(out.get("text"), result, fallback)
        except Exception:
            pass
    shown = args if isinstance(args, dict) else {}
    return {"text": text, "tool_calls": [{"name": name, "args": shown, "result": result}]}
