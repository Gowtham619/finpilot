import json
import pandas as pd
import pytest
from eval.generate import generate
from finpilot.agents import chat
from finpilot.engines import forecast as F
from finpilot.engines.categorize import normalize_merchant
from finpilot.engines.verify import ungrounded_numbers
from finpilot.llm import LLMUnavailable
from tests.conftest import FakeLLM


@pytest.fixture(scope="module")
def world():
    p = generate(seed=0)
    df = p.df.copy()
    df["merchant"] = df.description.map(normalize_merchant)
    df["category"] = p.truth.categories
    return p, df, F.fit(df), F.current_balance(df, p.opening_balance)


GOALS = [{"name": "Laptop", "target": 150000, "deadline": "2027-03-01"}]


def test_heuristic_routing_without_llm(world):
    _, df, m, bal = world
    out = chat.answer("what subscriptions do I pay for?", df, m, bal, [], None)
    assert out["tool_calls"][0]["name"] == "list_subscriptions"
    assert "NETFLIX" in out["text"]
    out = chat.answer("can I afford 50000?", df, m, bal, [], None)
    assert out["tool_calls"][0]["name"] == "afford"


def test_llm_picks_tool_and_answer_is_grounded(world):
    p, df, m, bal = world
    month, cat = p.truth.spike

    def handler(system, user):
        if "choose a tool" in system.lower():
            return {"tool": "month_total", "args": {"category": cat, "month": month}}
        return {"text": "You spent ₹99999 that month."}  # ungrounded -> must be replaced

    out = chat.answer("food delivery in July?", df, m, bal, [], FakeLLM(handler))
    assert out["tool_calls"][0]["name"] == "month_total"
    assert "99999" not in out["text"]
    assert out["tool_calls"][0]["result"]["total"] > 5000


def test_bad_tool_name_degrades(world):
    _, df, m, bal = world
    llm = FakeLLM(lambda s, u: {"tool": "drop_tables", "args": {}})
    out = chat.answer("hello", df, m, bal, [], llm)
    assert out["text"]


def _tc(out):
    return out["tool_calls"][0]


@pytest.mark.parametrize("name,args", [
    ("list_subscriptions", {}),
    ("top_merchants", {"n": 5}),
    ("top_merchants", {"n": 3, "month": "2026-07"}),
    ("forecast", {"horizon": 30}),
    ("forecast", {"horizon": 365}),
    ("afford", {"amount": 50000}),
    ("afford", {"amount": 5000000}),
    ("goal_status", {}),
])
def test_fallback_text_is_grounded_for_every_tool(world, name, args):
    p, df, m, bal = world
    tools = chat._tools(df, m, bal, GOALS)
    res = chat._run_tool(tools, name, args)
    text = chat._fallback_text(name, res)
    assert text and ungrounded_numbers(text, [], res) == [], text


def test_month_total_fallback_grounded(world):
    p, df, m, bal = world
    month, cat = p.truth.spike
    tools = chat._tools(df, m, bal, GOALS)
    for args in ({"category": cat, "month": month}, {"category": cat, "month": "1999-01"}):
        res = chat._run_tool(tools, "month_total", args)
        assert ungrounded_numbers(chat._fallback_text("month_total", res), [], res) == []


def test_negative_balances_are_worded_not_signed(world):
    _, df, m, bal = world
    out = chat.answer("what will my balance be?", df, m, -80000.0, [], None)
    res = _tc(out)["result"]
    assert res["end_p50"] < 0
    assert "overdrawn" in out["text"] and "-" not in out["text"].replace("-0", "")
    assert ungrounded_numbers(out["text"], [], res) == []
    out = chat.answer("can I afford 50000?", df, m, -80000.0, [], None)
    assert ungrounded_numbers(out["text"], [], _tc(out)["result"]) == []


def test_afford_outside_window_is_error_result(world):
    _, df, m, bal = world
    llm = FakeLLM(lambda s, u: {"tool": "afford", "args": {"amount": 5000, "date": "2020-01-01"}}
                  if "choose a tool" in s.lower() else {"text": "Sure, ₹123456."})
    out = chat.answer("can I afford 5000 in 2020?", df, m, bal, [], llm)
    res = _tc(out)["result"]
    assert "error" in res and "outside" in res["error"]
    assert "outside" in out["text"] and "123456" not in out["text"]
    assert ungrounded_numbers(out["text"], [], res) == []


def test_unknown_category_and_month_do_not_crash(world):
    _, df, m, bal = world
    for args in ({"category": "nope", "month": "2026-07"}, {"category": "dining", "month": "garbage"},
                 {"category": "dining"}, {}):
        llm = FakeLLM(lambda s, u, a=args: {"tool": "month_total", "args": a} if "choose a tool" in s.lower()
                      else {"text": "x"})
        out = chat.answer("how much?", df, m, bal, [], llm)
        assert out["text"] and "error" in _tc(out)["result"]
    out = chat.answer("x", df, m, bal, [], FakeLLM(
        lambda s, u: {"tool": "month_total", "args": {"category": "dining", "month": "1999-01"}}
        if "choose a tool" in s.lower() else {"text": "x"}))
    assert _tc(out)["result"]["total"] == 0


@pytest.mark.parametrize("reply", [None, [], "string", 5, {}, {"tool": 5}, {"tool": ["a"]},
                                   {"tool": "forecast", "args": "x"}, {"tool": "forecast", "args": [1]},
                                   {"tool": "afford", "args": {"amount": "abc"}},
                                   {"tool": "afford", "args": {"amount": float("inf")}},
                                   {"tool": "afford", "args": {"amount": -5}},
                                   {"tool": "afford", "args": {"amount": True}},
                                   {"tool": "afford", "args": {"amount": 5, "date": "garbage"}},
                                   {"tool": "forecast", "args": {"horizon": "abc"}},
                                   {"tool": "top_merchants", "args": {"n": None, "month": 5}},
                                   {"tool": "list_subscriptions", "args": {"evil": 1}}])
def test_malformed_llm_json_never_crashes(world, reply):
    _, df, m, bal = world
    out = chat.answer("hello there", df, m, bal, GOALS, FakeLLM(lambda s, u: reply))
    assert isinstance(out["text"], str) and out["text"]
    assert isinstance(out["tool_calls"], list)
    json.dumps(out, default=str)


def test_llm_answer_malformed_falls_back(world):
    _, df, m, bal = world
    for ans in (None, [], "str", {"text": 5}, {"nope": 1}):
        llm = FakeLLM(lambda s, u, a=ans: {"tool": "list_subscriptions", "args": {}}
                      if "choose a tool" in s.lower() else a)
        out = chat.answer("subs?", df, m, bal, [], llm)
        assert "NETFLIX" in out["text"]


def test_llm_exception_degrades(world):
    _, df, m, bal = world

    def boom(s, u):
        raise LLMUnavailable("down")
    out = chat.answer("what subscriptions?", df, m, bal, [], FakeLLM(boom))
    assert "NETFLIX" in out["text"]
    out = chat.answer("what subscriptions?", df, m, bal, [], FakeLLM(lambda s, u: (_ for _ in ()).throw(RuntimeError("x"))))
    assert "NETFLIX" in out["text"]


def test_args_are_clamped(world):
    _, df, m, bal = world
    tools = chat._tools(df, m, bal, [])
    assert chat._run_tool(tools, "forecast", {"horizon": 10**9})["horizon"] == 365
    assert chat._run_tool(tools, "forecast", {"horizon": -4})["horizon"] == 1
    assert chat._run_tool(tools, "forecast", {"horizon": "30"})["horizon"] == 30
    assert len(chat._run_tool(tools, "top_merchants", {"n": 10**9})["merchants"]) <= 20
    assert len(chat._run_tool(tools, "top_merchants", {"n": 0})["merchants"]) == 1


def test_invalid_args_retry_with_defaults_only_for_default_tools(world):
    _, df, m, bal = world
    tools = chat._tools(df, m, bal, [])
    r = chat._run_tool(tools, "forecast", {"horizon": "abc", "evil": 1})
    assert r["horizon"] == 30
    assert "error" in chat._run_tool(tools, "afford", {"evil": 1})
    assert "error" in chat._run_tool(tools, "month_total", {"category": {"a": 1}, "month": "2026-07"})


def test_goal_status_tool_handles_bad_goals(world):
    _, df, m, bal = world
    goals = GOALS + [{"name": "bad"}, "junk", {"name": "d", "target": "x", "deadline": "nope"}]
    out = chat.answer("how are my goals?", df, m, bal, goals, None)
    res = _tc(out)["result"]
    assert _tc(out)["name"] == "goal_status" and "Laptop" in out["text"]
    assert ungrounded_numbers(out["text"], [], res) == []
    out = chat.answer("how are my goals?", df, m, bal, [], None)
    assert "no goals" in out["text"].lower()


def test_tool_calls_echo_cleaned_args(world):
    _, df, m, bal = world
    llm = FakeLLM(lambda s, u: {"tool": "forecast", "args": {"horizon": "99999", "evil": 1}}
                  if "choose a tool" in s.lower() else {"text": "x"})
    out = chat.answer("hi", df, m, bal, [], llm)
    assert _tc(out)["args"] == {} and _tc(out)["result"]["horizon"] == 30
    llm = FakeLLM(lambda s, u: {"tool": "forecast", "args": {"horizon": "99999"}}
                  if "choose a tool" in s.lower() else {"text": "x"})
    out = chat.answer("hi", df, m, bal, [], llm)
    assert _tc(out)["args"] == {"horizon": 365}


def test_fallback_text_guard(world, monkeypatch):
    _, df, m, bal = world
    monkeypatch.setattr(chat, "_fallback_text", lambda n, r: (_ for _ in ()).throw(KeyError("x")))
    assert chat.answer("hi", df, m, bal, [], None)["text"]


# ---- final review fix I8: no-LLM routing for category + month questions ----
@pytest.mark.parametrize("q,cat,month", [
    ("How much did I spend on food delivery in July?", "food_delivery", "2026-07"),
    ("How much did I spend on food delivery in 2026-07?", "food_delivery", "2026-07"),
    ("Swiggy and Zomato in Jan 2026", "food_delivery", "2026-01"),
    ("groceries in march", "groceries", "2026-03"),
    ("What did uber cost me in September 2026?", "transport", "2026-09"),
    ("How much on shopping?", "shopping", "2026-09"),  # no month: latest full month in the data
    ("dining in may", "dining", "2026-05"),
])
def test_heuristic_category_month(world, q, cat, month):
    _, df, m, bal = world
    out = chat.answer(q, df, m, bal, [], None)
    call = out["tool_calls"][0]
    assert call["name"] == "month_total" and call["args"] == {"category": cat, "month": month}, call
    exp = round(float(-df[(df["category"] == cat) & (df["amount"] < 0)
                          & (df["date"].dt.to_period("M").astype(str) == month)]["amount"].sum()), 2)
    assert call["result"]["total"] == exp and month in out["text"]


def test_heuristic_ignores_years_as_amounts(world):
    _, df, m, bal = world
    out = chat.answer("Can I afford it in 2027?", df, m, bal, [], None)
    assert out["tool_calls"][0]["name"] == "afford" and out["tool_calls"][0]["args"]["amount"] == 0
    out = chat.answer("Can I afford 45,000 in 2026-11?", df, m, bal, [], None)
    assert out["tool_calls"][0]["args"]["amount"] == 45000


def test_heuristic_keeps_existing_routes(world):
    _, df, m, bal = world
    assert chat._heuristic("what subscriptions do I pay for?", df)[0] == "list_subscriptions"
    assert chat._heuristic("how are my goals?", df)[0] == "goal_status"
    assert chat._heuristic("top merchants in July", df) == ("top_merchants", {"month": "2026-07"})
    assert chat._heuristic("may I see my balance next month?", df)[0] == "forecast"


def test_chat_answer_with_false_risk_is_rejected(world):
    _, df, m, bal = world

    def handler(system, user):
        if "choose a tool" in system.lower():
            return {"tool": "forecast", "args": {"horizon": 30}}
        return {"text": "Your overdraft risk is 90% this month, borrow from a payday lender."}
    out = chat.answer("what's my risk?", df, m, bal, [], FakeLLM(handler))
    if out["tool_calls"][0]["result"]["p_overdraft"] != 0.9:
        assert "90%" not in out["text"]
