import json
import pytest
from eval.generate import generate
from finpilot.agents import advisor as AD
from finpilot.agents import verifier as VF
from finpilot.engines.analyze import analyze_all
from finpilot.engines.categorize import normalize_merchant
from finpilot.engines.verify import check_claim
from finpilot.llm import LLMUnavailable
from tests.conftest import FakeLLM


@pytest.fixture(scope="module")
def world():
    p = generate(seed=0)
    df = p.df.copy()
    df["merchant"] = df.description.map(normalize_merchant)
    df["category"] = p.truth.categories
    return p, df, analyze_all(df)


def test_candidates_are_grounded_and_cite_findings(world):
    _, df, findings = world
    cands = AD.build_candidates(findings, df)
    ids = {f["id"] for f in findings}
    assert cands
    for c in cands:
        assert c["finding_ids"] and set(c["finding_ids"]) <= ids
        assert all(check_claim(cl, df)["ok"] for cl in c["claims"])
    hot = next(c for c in cands if c["trigger_type"] == "unused_subscription" and "HOTSTAR" in c["action"])
    assert hot["est_monthly_savings"] == pytest.approx(299, abs=1)


def test_advise_without_llm_uses_templates(world):
    _, df, findings = world
    recs = AD.advise(AD.build_candidates(findings, df), None)
    assert recs and all(r["text"] for r in recs)
    kept, dropped = VF.verify_all(recs, df)
    assert len(kept) == len(recs) and not dropped and all(r["verified"] for r in kept)


def test_advise_llm_failure_falls_back(world):
    _, df, findings = world
    def boom(s, u):
        raise LLMUnavailable("down")
    recs = AD.advise(AD.build_candidates(findings, df), FakeLLM(boom))
    assert recs and all(r["text"] for r in recs)


def test_verifier_catches_and_corrects_hallucinated_number(world):
    _, df, findings = world
    cands = [c for c in AD.build_candidates(findings, df) if "HOTSTAR" in c["action"]]

    def advisor_handler(system, user):
        cid = cands[0]["cid"]
        bad = json.loads(json.dumps(cands[0]["claims"]))
        for c in bad:
            c["value"] = 999
        return {"recommendations": [{"cid": cid, "text": "Cancel HOTSTAR to save ₹999 a month.", "claims": bad}]}

    recs = AD.advise(cands, FakeLLM(advisor_handler))
    assert "999" in recs[0]["text"]

    events = []
    def rewrite(system, user):
        return {"text": "Cancel HOTSTAR to save ₹299 a month."}
    kept, dropped = VF.verify_all(recs, df, FakeLLM(rewrite), emit=lambda k, s, d: events.append(k))
    assert not dropped and kept[0]["verified"]
    assert "299" in kept[0]["text"] and "999" not in kept[0]["text"]
    assert "caught" in events and "corrected" in events
    assert "mismatch" in kept[0]["verifier_notes"]


def test_verifier_falls_back_to_template_when_rewrite_still_wrong(world):
    _, df, findings = world
    cands = [c for c in AD.build_candidates(findings, df) if "HOTSTAR" in c["action"]]
    recs = AD.advise(cands, None)
    recs[0]["text"] = "Cancel HOTSTAR to save ₹999 a month."
    kept, dropped = VF.verify_all(recs, df, FakeLLM(lambda s, u: {"text": "Still ₹888 wrong."}))
    assert kept and "999" not in kept[0]["text"] and "888" not in kept[0]["text"]


def test_unverifiable_claim_is_dropped(world):
    _, df, findings = world
    cands = AD.build_candidates(findings, df)
    recs = AD.advise(cands[:1], None)
    recs[0]["claims"] = [{"kind": "monthly_cost", "merchant": "GHOST", "value": 100}]
    kept, dropped = VF.verify_all(recs, df)
    assert not kept and len(dropped) == 1


def test_ungrounded_number_in_text_is_caught(world):
    _, df, findings = world
    cands = [c for c in AD.build_candidates(findings, df) if "HOTSTAR" in c["action"]]
    recs = AD.advise(cands, None)
    recs[0]["text"] = "Cancel HOTSTAR (₹299/month) and also save ₹4,321 elsewhere."
    issues = VF.audit(recs[0], df)
    assert [i["type"] for i in issues] == ["ungrounded_number"]
    assert issues[0]["value"] == 4321.0
    kept, dropped = VF.verify_all(recs, df)
    assert kept and "4,321" not in kept[0]["text"] and not dropped


def test_verify_all_rewrite_unavailable_falls_back_to_template(world):
    _, df, findings = world
    cands = [c for c in AD.build_candidates(findings, df) if "HOTSTAR" in c["action"]]
    recs = AD.advise(cands, None)
    recs[0]["text"] = "Cancel HOTSTAR to save ₹999 a month."
    def boom(s, u):
        raise LLMUnavailable("down")
    events = []
    kept, dropped = VF.verify_all(recs, df, FakeLLM(boom), emit=lambda k, s, d: events.append(k))
    assert kept and not dropped and "999" not in kept[0]["text"]
    assert "template" in kept[0]["verifier_notes"] and "corrected" in events


def test_unknown_claim_kind_rec_is_dropped(world):
    _, df, findings = world
    recs = AD.advise(AD.build_candidates(findings, df)[:1], None)
    recs[0]["claims"] = [{"kind": "mystery", "value": 5}]
    issues = VF.audit(recs[0], df)
    assert any(i["type"] == "unverifiable" for i in issues)
    kept, dropped = VF.verify_all(recs, df)
    assert not kept and len(dropped) == 1


def _rec(trigger, evidence, action="x", claims=()):
    return {"cid": "c0", "trigger_type": trigger, "action": action, "finding_ids": ["f"],
            "evidence": evidence, "claims": list(claims), "est_monthly_savings": 0.0,
            "est_one_off": 0.0, "id": "rec-t", "verified": False, "verifier_notes": "", "model": "template"}


def test_every_template_passes_its_own_audit(world):
    _, df, findings = world
    cands = AD.build_candidates(findings, df, forecast={
        "summary": {"p_overdraft": 0.37, "floor_p50": -1234.5, "end_p10": 2000.0, "end_p50": 9000.0,
                    "end_p90": 15000.0, "risk_day": "2026-09-14"}, "horizon": 60},
        goals=[{"name": "Emergency fund", "p_met": 0.23}])
    types = {c["trigger_type"] for c in cands}
    assert {"unused_subscription", "category_spike", "duplicate_charge", "large_charge",
            "behavior_trigger", "forecast_risk", "goal_behind"} <= types
    seen = set()
    for c in cands:
        rec = AD.advise([c], None)[0]
        seen.add(c["trigger_type"])
        assert VF.audit(rec, df) == [], (c["trigger_type"], rec["text"])
    assert len(seen) >= 7


MALFORMED = [{}, "junk", None, {"kind": "monthly_cost", "merchant": "HOTSTAR", "value": "₹5"},
             {"kind": "monthly_cost", "value": 5}, {"kind": "txn_amount", "txn_id": None, "value": 24000},
             {"kind": "monthly_cost", "merchant": "HOTSTAR"}]


@pytest.mark.parametrize("bad", MALFORMED)
def test_malformed_claims_through_verify_all_do_not_raise(world, bad):
    _, df, findings = world
    cands = [c for c in AD.build_candidates(findings, df) if "HOTSTAR" in c["action"]]
    recs = AD.advise(cands, None)
    recs[0]["claims"] = [bad]
    events = []
    kept, dropped = VF.verify_all(recs, df, None, emit=lambda k, s, d: events.append(k))
    assert len(kept) + len(dropped) == 1
    assert "caught" in events
    if kept:  # identifiable claim with a bad value is corrected from the recomputed number
        assert VF.audit(kept[0], df) == [] and kept[0]["claims"][0]["value"] == pytest.approx(299, abs=1)
        assert "corrected" in events
    else:
        assert "dropped" in events


def test_llm_cannot_choose_what_is_verified(world):
    p, df, findings = world
    cands = [c for c in AD.build_candidates(findings, df) if "HOTSTAR" in c["action"]]
    orig = json.loads(json.dumps(cands[0]["claims"]))
    tid = next(iter(p.truth.anomaly_ids))

    def handler(s, u):
        return {"recommendations": [{"cid": cands[0]["cid"],
                                     "text": "Cancel HOTSTAR to save ₹24000 a month.",
                                     "claims": [{"kind": "txn_amount", "txn_id": tid, "value": 24000}]}]}
    recs = AD.advise(cands, FakeLLM(handler))
    assert recs[0]["claims"] == orig
    assert {c["kind"] for c in recs[0]["claims"]} == {"monthly_cost", "annual_cost"}
    assert VF.audit(recs[0], df)  # fake 24000 flagged
    kept, dropped = VF.verify_all(recs, df)
    assert kept and "24000" not in kept[0]["text"] and kept[0]["model"] == "template"
    assert "template" in kept[0]["verifier_notes"]


def test_llm_value_restated_only_for_matching_identity(world):
    _, df, findings = world
    cands = [c for c in AD.build_candidates(findings, df) if "HOTSTAR" in c["action"]]
    def handler(s, u):
        return {"recommendations": [{"cid": cands[0]["cid"], "text": "Cancel HOTSTAR (₹299/month).",
                                     "claims": [{"kind": "monthly_cost", "merchant": "NETFLIX", "value": 1},
                                                {"kind": "monthly_cost", "merchant": "HOTSTAR", "value": 777}]}]}
    recs = AD.advise(cands, FakeLLM(handler))
    by = {c["kind"]: c["value"] for c in recs[0]["claims"]}
    assert by["monthly_cost"] == 777 and by["annual_cost"] == cands[0]["claims"][1]["value"]
    assert len(recs[0]["claims"]) == len(cands[0]["claims"])


def test_evidence_only_recs_are_labelled_honestly(world):
    _, df, findings = world
    cands = AD.build_candidates(findings, df, goals=[{"name": "Fund", "p_met": 0.2}])
    recs = AD.advise(cands, None)
    kept, _ = VF.verify_all(recs, df)
    ev = [r for r in kept if not r["claims"]]
    rc = [r for r in kept if r["claims"]]
    assert ev and rc
    for r in ev:
        assert r["verification"] == "evidence_only"
        assert r["verifier_notes"] == "no recomputable claims; numbers checked against engine evidence only"
    for r in rc:
        assert r["verification"] == "recomputed" and "recomputed" in r["verifier_notes"]


@pytest.mark.parametrize("rw", [{"text": 5}, {"text": ""}, {"text": None}, {"text": ["a"]}, ["x"]])
def test_non_string_rewrite_falls_back_to_template(world, rw):
    _, df, findings = world
    cands = [c for c in AD.build_candidates(findings, df) if "HOTSTAR" in c["action"]]
    recs = AD.advise(cands, None)
    recs[0]["text"] = "Cancel HOTSTAR to save ₹999 a month."
    kept, dropped = VF.verify_all(recs, df, FakeLLM(lambda s, u: rw))
    assert kept and not dropped and "999" not in kept[0]["text"]
    assert kept[0]["model"] == "template"


def test_currency_escape_in_rec_text_is_flagged(world):
    _, df, findings = world
    cands = [c for c in AD.build_candidates(findings, df) if "HOTSTAR" in c["action"]]
    recs = AD.advise(cands, None)
    recs[0]["text"] = "Cancel HOTSTAR (₹299/month) and save ₹1.5k more."
    assert [i["type"] for i in VF.audit(recs[0], df)] == ["ungrounded_number"]
