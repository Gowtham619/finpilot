"""risk_claim_consistent: number-free contradictions of computed risk (best-effort wording filter)."""
import pytest
from finpilot.agents.common import pct_text, pct_values, risk_claim_consistent, risk_level


@pytest.mark.parametrize("p,level", [(0, "none"), (0.009, "none"), (0.01, "low"), (0.05, "low"),
                                     (0.051, "some"), (0.49, "some"), (0.5, "high"), (1.0, "high")])
def test_risk_level(p, level):
    assert risk_level(p) == level


def test_pct_text_matches_ui_pct():
    assert [pct_text(x) for x in (0, 0.003, 0.0049, 0.005, 0.5, 1)] == ["0%", "<1%", "<1%", "1%", "50%", "100%"]
    assert pct_values(0.003) == [0.3, 1.0] and pct_values(0.234) == [23.4, 23.0]


@pytest.mark.parametrize("text", [
    "You will definitely not go overdrawn; zero chance.",
    "There is no chance you go overdrawn.",
    "You cannot go overdrawn.",
    "You will not be overdrawn.",
    "Skies look fine; no breach expected.",
    "Go ahead, this purchase will not hurt you at all.",
    "It is risk-free.",
])
def test_no_risk_wording_rejected_when_risk_is_real(text):
    assert not risk_claim_consistent(text, risk_level(0.02))
    assert not risk_claim_consistent(text, risk_level(0.3))


@pytest.mark.parametrize("text", ["Nothing to worry about, all good.", "This looks fine.", "Go ahead."])
def test_soft_no_risk_allowed_only_when_low(text):
    assert risk_claim_consistent(text, "low")
    assert not risk_claim_consistent(text, "some") and not risk_claim_consistent(text, "high")


@pytest.mark.parametrize("text", ["You will definitely be overdrawn.", "Overdraft is certain.",
                                  "You will be overdrawn by the end of the month."])
def test_certainty_only_when_high(text):
    assert not risk_claim_consistent(text, "some")
    assert not risk_claim_consistent(text, "none")
    assert risk_claim_consistent(text, "high")


def test_quantity_words_checked_against_failing_share():
    assert not risk_claim_consistent("Half of futures fail.", "some", [0.2])
    assert risk_claim_consistent("Half of futures fail.", "some", [0.5])
    assert not risk_claim_consistent("Most simulated futures dip below the buffer.", "some", [0.2])
    assert risk_claim_consistent("Most futures stay above the buffer.", "some", [0.2])
    assert not risk_claim_consistent("All paths stay above the buffer.", "some", [0.2])
    assert risk_claim_consistent("None of the futures dip below the buffer.", "none", [0.0])
    assert not risk_claim_consistent("None of the futures dip below the buffer.", "some", [0.3])
    assert risk_claim_consistent("Every simulated future includes the purchase.", "high", [0.77])  # no polarity


def test_plain_factual_prose_passes():
    assert risk_claim_consistent("Overdraft risk rises from 2% to 30%. Verdict: risky.", "some", [0.3])
    assert risk_claim_consistent("Chance of going overdrawn: 0%.", "none", [0.0])
    assert not risk_claim_consistent(None, "none")
