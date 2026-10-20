from __future__ import annotations

import io
import json
import math
import os
import re
import threading
import uuid

import pandas as pd

from . import db
from .engines.analyze import monthly_aggregates
from .graph import build_graph
from .llm import get_llm
from .trace import trace


PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_ROWS = 50000


def check_profile(profile):
    if not isinstance(profile, str) or not PROFILE_RE.match(profile):
        raise ValueError("profile must be 1-40 characters of letters, digits, '_' or '-'")
    return profile


def _clean(obj):
    return json.loads(json.dumps(obj, default=str))


def _sync():
    return os.getenv("FINPILOT_SYNC") == "1"


def parse_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"could not read CSV: {e}")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ("date", "description", "amount") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must have columns date, description, amount (missing: {', '.join(missing)})")
    if len(df) > MAX_ROWS:
        raise ValueError(f"CSV has too many rows (max {MAX_ROWS})")
    df = df[["date", "description", "amount"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])
    if len(df) < 5:
        raise ValueError("need at least 5 valid rows")
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def _persist(profile):
    def save(state):
        with db.session() as s:
            db.save_transactions(s, profile, state["df"], state["opening"])
            db.save_findings(s, profile, _clean(state["findings"]))
            db.save_recs(s, profile, _clean(state["recs"]))
            db.save_forecast(s, profile, _clean({"forecast": state["forecast"], "overview": state["overview"],
                                                   "weather": state.get("weather")}))
            by_name = {g["name"]: g["pk"] for g in db.list_goals(s, profile)}
            for g in state["forecast"]["goals"]:
                if "p_met" not in g:  # error entry {"name", "error"}
                    continue
                if g["name"] in by_name:
                    db.upsert_goal_snapshot(s, by_name[g["name"]], _clean(
                        {"on": str(state["df"]["date"].max().date()), "p_met": g["p_met"],
                         "shortfall_p50": g["shortfall_p50"]}))
    return save


def _run(profile, raw_df, opening, run_id):
    try:
        with db.session() as s:
            corr, goals = db.corrections(s, profile), db.list_goals(s, profile)
        g = build_graph(get_llm(), trace, _persist(profile))
        g.invoke({"run_id": run_id, "profile": profile, "request": {"type": "pipeline"}, "df": raw_df,
                  "opening": float(opening), "corrections": corr, "goals": goals})
    except Exception as e:  # surface in the trace instead of dying silently
        trace.emit(run_id, "coordinator", "error", f"{type(e).__name__}: {e}")
    finally:
        trace.finish(run_id)
        try:
            with db.session() as s:
                db.save_trace(s, run_id)
        except Exception:  # must not mask the original error
            pass


def start_pipeline(profile, raw_df, opening):
    check_profile(profile)
    run_id = uuid.uuid4().hex[:10]
    trace.emit(run_id, "coordinator", "start", f"new data for '{profile}': {len(raw_df)} transactions")
    if _sync():
        _run(profile, raw_df, opening, run_id)
    else:
        threading.Thread(target=_run, args=(profile, raw_df, opening, run_id), daemon=True).start()
    return run_id


def load_persona(name):
    from eval.generate import PERSONAS, load_persona as lp
    if name not in PERSONAS:
        raise ValueError("unknown persona")
    check_profile(name)
    p = lp(name)
    return name, start_pipeline(name, p.df, p.opening_balance)


def autoseed(name="priya"):
    with db.session() as s:
        has = db.load_df(s, name)[0] is not None
    if not has:
        load_persona(name)


def _num(v, what):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number")
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{what} must be a finite number")
    return x


def _norm_date(v, what="date"):
    try:
        t = pd.Timestamp(v)
    except Exception:
        raise ValueError(f"{what} is not a valid date: {v!r}")
    if pd.isna(t):
        raise ValueError(f"{what} is not a valid date: {v!r}")
    return str(t.date())


def _flow_ok(opt_name, amount, date, months, start, end):
    if not 0 <= months <= 60:
        raise ValueError(f"{opt_name}: financed_months must be between 0 and 60")
    if amount > 0 and not (start <= pd.Timestamp(date) <= end):
        raise ValueError(f"{opt_name}: purchase date {date} is outside the forecast window "
                         f"{start.date()}..{end.date()}")


def _cuts(v, what):
    """{category: fraction in [0, 1]}; anything else is a ValueError (HTTP 400)."""
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"{what} cuts must be an object mapping a category to a fraction between 0 and 1")
    from .config import CATEGORIES
    out = {}
    for k, x in v.items():
        if k not in CATEGORIES:
            raise ValueError(f"{what} cuts: unknown category {k!r}")
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError(f"{what} cuts: {k} must be a number between 0 and 1")
        out[k] = float(x)
    return out


def _months(o):
    try:
        return int(o.get("financed_months") or 0)
    except (TypeError, ValueError):
        raise ValueError("financed_months must be a whole number")


MAX_GOAL_TARGET = 1e12


def validate_goal(name, target, deadline, last_date=None):
    """Returns (name, target, deadline_str) or raises ValueError. The deadline must lie between one year
    before and ten years after the data's last date (today when there is no data yet)."""
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
        raise ValueError("goal name must be 1-60 characters")
    if isinstance(target, bool):
        raise ValueError("target must be a number")
    target = _num(target, "target")
    if not 0 < target <= MAX_GOAL_TARGET:
        raise ValueError(f"target must be greater than 0 and at most {MAX_GOAL_TARGET:,.0f}")
    try:
        d = pd.Timestamp(deadline)
    except Exception:
        raise ValueError(f"deadline is not a valid date: {deadline!r}")
    if pd.isna(d):
        raise ValueError(f"deadline is not a valid date: {deadline!r}")
    d = d.normalize()
    ref = pd.Timestamp(last_date).normalize() if last_date is not None else pd.Timestamp.today().normalize()
    lo, hi = ref - pd.DateOffset(years=1), ref + pd.DateOffset(years=10)
    if not lo <= d <= hi:
        raise ValueError(f"deadline must be between {lo.date()} and {hi.date()}")
    return name.strip(), target, str(d.date())


def add_goal(profile, body):
    check_profile(profile)
    if not isinstance(body, dict):
        raise ValueError("goal needs name, target (number) and deadline")
    with db.session() as s:
        df, _ = db.load_df(s, profile)
        last = df["date"].max() if df is not None else None
        name, target, deadline = validate_goal(body.get("name"), body.get("target"), body.get("deadline"), last)
        db.add_goal(s, profile, name, target, deadline)


def request(profile, req):
    rtype = req.get("type")
    if rtype not in ("afford", "duel", "chat"):
        raise ValueError(f"unknown request type: {rtype!r}")
    with db.session() as s:
        df, opening = db.load_df(s, profile)
        goals = db.list_goals(s, profile)
    if df is None:
        raise LookupError(f"no data for profile '{profile}'")
    d0 = str((df["date"].max() + pd.Timedelta(days=1)).date())
    if rtype == "afford":
        req["amount"] = _num(req.get("amount"), "amount")
        if req["amount"] < 0:
            raise ValueError("amount must not be negative")
        req["date"] = _norm_date(req["date"]) if req.get("date") else d0
        req["financed_months"] = _months(req)
        if not 0 <= req["financed_months"] <= 60:
            raise ValueError("financed_months must be between 0 and 60")
    elif rtype == "duel":
        for k in ("a", "b"):
            o = req.get(k)
            if not isinstance(o, dict):
                raise ValueError(f"option '{k}' is required")
            o["name"] = str(o.get("name") or f"Option {k.upper()}")
            o["amount"] = _num(o.get("amount"), f"option {k} amount")
            if o["amount"] < 0:
                raise ValueError(f"option {k} amount must not be negative")
            o["date"] = _norm_date(o["date"], f"option {k} date") if o.get("date") else d0
            o["financed_months"] = _months(o)
            o["cuts"] = _cuts(o.get("cuts"), f"option {k}")
            try:
                raw_h = req.get("horizon", 90)
                if isinstance(raw_h, (float, bool)) and not (isinstance(raw_h, float) and raw_h.is_integer()):
                    raise ValueError("horizon must be a whole number")
                hz = int(raw_h)
            except (TypeError, ValueError):
                raise ValueError("horizon must be a whole number")
            if not 1 <= hz <= 365:
                raise ValueError("horizon must be between 1 and 365")
            start = pd.Timestamp(d0)
            _flow_ok(f"option {k} ({o['name']})", o["amount"], o["date"], o["financed_months"],
                     start, start + pd.Timedelta(days=hz - 1))
    run_id = uuid.uuid4().hex[:10]
    trace.emit(run_id, "coordinator", "start", f"{rtype} request")
    try:
        out = build_graph(get_llm(), trace).invoke(
            {"run_id": run_id, "profile": profile, "request": req, "df": df, "opening": opening,
             "corrections": {}, "goals": goals})
    except Exception as e:
        trace.emit(run_id, "coordinator", "error", f"{type(e).__name__}: {e}")
        raise
    finally:
        trace.finish(run_id)
    return run_id, _clean(out["answer"])


def state(profile):
    with db.session() as s:
        df, opening = db.load_df(s, profile)
        if df is None:
            return {"has_data": False}
        fc = db.load_forecast(s, profile) or {}
        out = {"has_data": True, "profile": profile, "findings": db.load_findings(s, profile),
               "recs": db.load_recs(s, profile), "goals": db.list_goals(s, profile),
               "forecast": fc.get("forecast"), "overview": fc.get("overview", ""),
               "balance": (fc.get("forecast") or {}).get("balance")}
    agg = monthly_aggregates(df).round(0)
    out["monthly"] = {"months": list(agg.index), "series": {c: agg[c].tolist() for c in agg.columns}}
    return _clean(out)
