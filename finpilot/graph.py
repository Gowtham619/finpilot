from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import advisor, analyzer, categorizer, forecaster, meteorologist, simulator, verifier
from .engines import weather as weather_engine
from .engines.forecast import current_balance
from .trace import trace as default_trace


class State(TypedDict, total=False):
    run_id: str
    profile: str
    request: dict
    df: Any
    opening: float
    corrections: dict
    goals: list
    findings: list
    overview: str
    forecast: dict
    weather: dict
    model: Any
    balance: float
    recs: list
    dropped: list
    answer: dict


def route(state):
    return state["request"]["type"]


def _public(fc):
    return {k: v for k, v in fc.items() if k != "model"}


def build_graph(llm=None, tr=default_trace, persist=None):
    def emit(state, agent, kind, summary, data=None):
        tr.emit(state["run_id"], agent, kind, summary, data)

    def n_categorize(state):
        emit(state, "categorizer", "start", f"labeling {len(state['df'])} transactions")
        df = categorizer.categorize_df(state["df"], llm, state.get("corrections"))
        src = df["category_source"].value_counts().to_dict()
        emit(state, "categorizer", "result",
             f"{src.get('rule', 0)} by rule, {src.get('llm', 0)} by LLM, "
             f"{src.get('user', 0)} by your corrections, {src.get('none', 0)} uncategorized", src)
        return {"df": df}

    def n_analyze(state):
        emit(state, "analyzer", "tool", "running anomaly, subscription and behavior engines")
        out = analyzer.run(state["df"], llm)
        emit(state, "analyzer", "result", out["overview"], {"n": len(out["findings"])})
        return {"findings": out["findings"], "overview": out["overview"]}

    def n_forecast(state):
        emit(state, "forecaster", "tool", "fitting model and running 1,000 Monte Carlo paths")
        fc = forecaster.run(state["df"], state["opening"], state.get("goals") or [], llm)
        emit(state, "forecaster", "result", fc["narrative"], fc["summary"])
        try:  # same unrounded balance the Verifier rebuilds from opening + transactions
            wx = weather_engine.report(fc["model"], current_balance(state["df"], state["opening"]))
        except Exception as e:  # weather is additive: degrade, never crash the pipeline
            wx = None
            emit(state, "forecaster", "tool", f"weather unavailable: {e}")
        return {"forecast": _public(fc), "model": fc["model"], "balance": fc["balance"], "weather": wx}

    def n_advise(state):
        cands = advisor.build_candidates(state["findings"], state["df"], state["forecast"],
                                         state["forecast"]["goals"], weather=state.get("weather"),
                                         ctx={"opening": state["opening"]})
        emit(state, "advisor", "tool", f"{len(cands)} candidate actions from findings and forecast")
        recs = advisor.advise(cands, llm, getattr(llm, "model", "") if llm else "")
        n_llm = sum(r.get("model") != "template" for r in recs)
        emit(state, "advisor", "result", f"drafted {len(recs)} recommendations ({n_llm} written by the LLM)",
             {"llm_written": [r["id"] for r in recs if r.get("model") != "template"]})
        return {"recs": recs}

    def n_verify(state):
        n_re = sum(1 for r in state["recs"] if r.get("claims"))
        emit(state, "verifier", "start",
             f"checking every number in {len(state['recs'])} recommendations: {n_re} with claims recomputed "
             f"from raw transactions, {len(state['recs']) - n_re} against the engine's own evidence only")
        kept, dropped = verifier.verify_all(
            state["recs"], state["df"], llm,
            emit=lambda kind, summary, data=None: emit(state, "verifier", kind, summary, data),
            ctx={"opening": state["opening"]})
        k_re = sum(1 for r in kept if r.get("verification") == "recomputed")
        emit(state, "verifier", "result",
             f"{len(kept)} verified ({k_re} recomputed from transactions, {len(kept) - k_re} checked against "
             f"engine evidence only), {len(dropped)} dropped")
        return {"recs": kept, "dropped": dropped}

    def n_persist(state):
        if persist:
            persist(state)
        emit(state, "coordinator", "result", "saved results")
        return {}

    def n_prepare(state):
        if state.get("model") is None:
            fc = forecaster.run(state["df"], state["opening"], state.get("goals") or [], None)
            return {"model": fc["model"], "balance": fc["balance"], "forecast": _public(fc)}
        return {}

    def n_afford(state):
        emit(state, "simulator", "tool", "simulating the purchase against 1,000 futures")
        # ValueError (date outside forecast window) intentionally propagates
        ans = simulator.afford(state["model"], state["balance"], state["request"], llm)
        emit(state, "simulator", "result", ans["narrative"])
        return {"answer": ans}

    def n_duel(state):
        emit(state, "simulator", "tool", "running both options on identical random futures")
        ans = simulator.run_duel(state["model"], state["balance"], state["request"], llm)
        emit(state, "simulator", "result", ans["reason"])
        return {"answer": ans}

    def n_weather(state):
        emit(state, "meteorologist", "start", "reading the 60-day money weather")
        emit(state, "meteorologist", "tool", "simulating 1,000 futures for the risk calendar, autopsy and umbrellas")
        bal = current_balance(state["df"], state["opening"])
        # ValueError (bad params) intentionally propagates
        ans = meteorologist.run(state["model"], bal, state["request"].get("params") or {}, llm)
        emit(state, "meteorologist", "result", ans["headline_text"], ans["headline"])
        return {"answer": ans}

    def n_chat(state):
        from .agents import chat  # built in Task 11; ImportError until then
        emit(state, "chat", "tool", "answering from your data")
        ans = chat.answer(state["request"]["question"], state["df"], state["model"], state["balance"],
                          state.get("goals") or [], llm)
        emit(state, "chat", "result", ans["text"])
        return {"answer": ans}

    g = StateGraph(State)
    for name, fn in [("categorize", n_categorize), ("analyze", n_analyze), ("forecast", n_forecast),
                     ("advise", n_advise), ("verify", n_verify), ("persist", n_persist),
                     ("prepare", n_prepare), ("afford", n_afford), ("duel", n_duel), ("weather", n_weather), ("chat", n_chat)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, route, {"pipeline": "categorize", "afford": "prepare",
                                           "duel": "prepare", "weather": "prepare",
                                           "chat": "prepare"})
    g.add_conditional_edges("prepare", route, {"afford": "afford", "duel": "duel", "weather": "weather",
                                            "chat": "chat"})
    for a, b in [("categorize", "analyze"), ("analyze", "forecast"), ("forecast", "advise"),
                 ("advise", "verify"), ("verify", "persist"), ("persist", END),
                 ("afford", END), ("duel", END), ("weather", END), ("chat", END)]:
        g.add_edge(a, b)
    return g.compile()
