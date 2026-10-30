import re

from finpilot.agents import analyzer, meteorologist as M
from finpilot.agents.common import fmt_date, fmt_month, inr, plural
from finpilot.engines.verify import text_numbers


def nums(t):
    return [n["value"] for n in text_numbers(t)]


def test_named_month_dates_are_not_claims_but_amounts_still_are():
    assert nums("charged ₹2,199 on 22 Jun 2026 and again Nov 30 in Mar 2026") == [2199.0]
    assert nums("on 8th March, 3 Sept 2025") == []
    assert nums("lost 50000 Mar 2026") == [50000.0]  # a big number next to a month is not a date


def test_rupees_use_indian_grouping_like_the_ui():
    assert inr(108729) == "₹1,08,729" and inr(13212) == "₹13,212" and inr(12345678) == "₹1,23,45,678" and inr(0) == "₹0"
    assert inr(-1500) == "-₹1,500"


def test_plain_helpers():
    assert plural(1, "row") == "1 row" and plural(3, "row") == "3 rows" and plural(2, "stretch", "stretches") == "2 stretches"
    assert fmt_date("2026-03-08") == "8 Mar 2026" and fmt_month("2026-04") == "Apr 2026" and fmt_date("d") == "d"


def test_finding_text_reads_like_a_person_wrote_it():
    spike = analyzer.finding_text({"type": "category_spike", "category": "food_delivery", "value": 12040, "month": "2026-04",
                                   "pct_change": 180.4, "baseline": 4300})
    assert spike == "You spent ₹12,040 on food delivery in Apr 2026, 180% more than your usual ₹4,300."
    big = analyzer.finding_text({"type": "large_charge", "amount": 148600, "merchant": "CROMA", "date": "2026-03-08"})
    assert "₹1,48,600" in big and "8 Mar 2026" in big and "2026-03" not in big


def test_weather_headline_has_no_placeholder_plurals():
    rep = {"headline": {"p_any_breach": .59, "worst_day": "2026-11-30", "worst_p": .48, "n_storm_days": 1},
           "buffer": 7435.0, "horizon": 60, "storms": [{}], "umbrellas": [],
           "autopsy": {"n_failing": 0, "n_paths": 1000, "causes": [], "bills": [], "insufficient_failing": False,
                       "first_breach": {"date": None}}}
    t = M.fallback_texts(rep)["headline_text"]
    assert "(s)" not in t and "reached on 1 day, in 1 separate stretch." in t and "Nov 30" in t
    rep["headline"]["n_storm_days"], rep["storms"] = 9, [{}, {}]
    assert "reached on 9 days, in 2 separate stretches." in M.fallback_texts(rep)["headline_text"]
    assert not re.search(r"\(s\)", M.fallback_texts({**rep, "storms": []})["headline_text"])
