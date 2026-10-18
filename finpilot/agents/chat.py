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
    return _run_tool_args(tools, name, args)[0]


def _run_tool_args(tools, name, args):
    """Returns (result, cleaned_args). Validate args, then run. Invalid args fall back to defaults only for tools whose args are all optional;
    any other failure becomes an {"error": ...} result. Never raises."""
    try:
        clean = _clean_args(name, args)
    except BadArgs as e:
        if _REQUIRED.get(name):
            return {"error": f"I need valid arguments for {name.replace('_', ' ')}: {e}."}, {}
        clean = {}
    except Exception:
        clean = {}
    try:
        return tools[name](**clean), clean
    except Exception as e:
        return {"error": f"{name.replace('_', ' ')} failed ({type(e).__name__})"}, clean


TOOL_SYSTEM = ("Choose a tool to answer the user's money question. Tools: month_total(category, month 'YYYY-MM'), "
               "top_merchants(n, month), list_subscriptions(), forecast(horizon), afford(amount, date), goal_status(). "
               "Categories: rent, groceries, dining, food_delivery, transport, utilities, subscriptions, shopping, "
               "entertainment, health, travel. Return JSON {\"tool\": name, \"args\": {...}}.")
ANSWER_SYSTEM = ("Answer the question using ONLY the tool results. Be concise, use full numbers like ₹1500. "
                 "Return JSON {\"text\": str}.")


# no-LLM routing vocabulary: phrase -> category (longest phrases are tried first)
CATEGORY_WORDS = {
    "food delivery": "food_delivery", "food_delivery": "food_delivery", "delivery": "food_delivery",
    "swiggy": "food_delivery", "zomato": "food_delivery",
    "groceries": "groceries", "grocery": "groceries", "bigbasket": "groceries", "dmart": "groceries",
    "dining": "dining", "eating out": "dining", "restaurants": "dining", "restaurant": "dining",
    "cafe": "dining", "rent": "rent",
    "transport": "transport", "uber": "transport", "ola": "transport", "cab": "transport",
    "cabs": "transport", "taxi": "transport", "metro": "transport",
    "shopping": "shopping", "amazon": "shopping", "flipkart": "shopping",
    "subscriptions": "subscriptions", "subscription": "subscriptions",
    "utilities": "utilities", "utility": "utilities", "electricity": "utilities", "broadband": "utilities",
    "health": "health", "pharmacy": "health", "medical": "health", "doctor": "health",
    "entertainment": "entertainment", "movies": "entertainment", "cinema": "entertainment",
    "travel": "travel", "flights": "travel",
}
_CAT_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(CATEGORY_WORDS, key=len, reverse=True))
                     + r")\b", re.I)
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
           "oct": 10, "nov": 11, "dec": 12}
_MONTH_NAME_RE = re.compile(r"\b(?P<pre>in\s+)?(?P<m>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
                            r"june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
                            r"dec(?:ember)?)\b(?:,?\s+(?P<y>(?:19|20)\d{2}))?", re.I)
_ISO_MONTH_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})-(0[1-9]|1[0-2])(?!\d)")
_SPEND_WORDS = re.compile(r"\b(spen[dt]|spending|how much|total)\b", re.I)


def _last_date(df):
    try:
        return pd.Timestamp(df["date"].max())
    except Exception:
        return pd.Timestamp.today()


def _find_month(q, df):
    m = _ISO_MONTH_RE.search(q)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    for m in _MONTH_NAME_RE.finditer(q):
        name = m.group("m").lower()
        if name == "may" and not (m.group("pre") or m.group("y")):
            continue  # "may I ..." is not a month
        mon = _MONTHS[name[:3]]
        if m.group("y"):
            return f"{m.group('y')}-{mon:02d}"
        end = _last_date(df)  # most recent such month that the data reaches
        year = end.year if mon <= end.month else end.year - 1
        return f"{year}-{mon:02d}"
    return None


def _latest_full_month(df):
    end = _last_date(df).normalize()
    p = end.to_period("M")
    return str(p if end == (end + pd.offsets.MonthEnd(0)).normalize() else p - 1)


def _amounts(q):
    """Numbers in the question, ignoring YYYY-MM months and bare 4-digit years (1990-2100)."""
    q = _ISO_MONTH_RE.sub(" ", q)
    out = []
    for tok in re.findall(r"\d[\d,]*\.?\d*", q):
        raw = tok.rstrip(".,").replace(",", "")
        if not raw.replace(".", ""):
            continue
        if re.fullmatch(r"\d{4}", tok.rstrip(".,")) and 1990 <= int(raw) <= 2100:
            continue
        out.append(float(raw))
    return out


def _heuristic(q, df=None):
    ql = q.lower()
    nums = _amounts(q)
    if "afford" in ql or "buy" in ql:
        return "afford", {"amount": nums[0] if nums else 0}
    cm = _CAT_RE.search(q)
    cat = CATEGORY_WORDS[cm.group(1).lower()] if cm else None
    month = _find_month(q, df)
    if cat and (month or _SPEND_WORDS.search(q)):
        return "month_total", {"category": cat, "month": month or _latest_full_month(df)}
    if "subscri" in ql:
        return "list_subscriptions", {}
    if "goal" in ql:
        return "goal_status", {}
    if "top" in ql or "merchant" in ql or "where" in ql:
        return "top_merchants", ({"month": month} if month else {})
    if cat:
        return "month_total", {"category": cat, "month": month or _latest_full_month(df)}
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
        name, args = _heuristic(str(question), df)
    result, args = _run_tool_args(tools, name, args)
    try:
        fallback = _fallback_text(name, result)
    except Exception:
        fallback = "I couldn't put that into words, sorry."
    text = fallback
    if llm is not None:
        try:
            out = llm.json(ANSWER_SYSTEM, json.dumps({"question": question, "tool": name, "result": result}, default=str))
            if isinstance(out, dict) and "error" not in result:
                text = grounded_or(out.get("text"), result, fallback)
        except Exception:
            pass
    return {"text": text, "tool_calls": [{"name": name, "args": args, "result": result}]}
