from __future__ import annotations

import copy
import json

from ..engines.verify import check_claim, claims_missing_from_text, ungrounded_numbers, weather_text_issues
from ..llm import LLMUnavailable
from .advisor import template_text

REWRITE_SYSTEM = ("Rewrite the advice using ONLY the corrected claim values and the evidence. Keep it to "
                  "two sentences. Return JSON {\"text\": str}.")


EVIDENCE_NOTE = "no recomputable claims; numbers checked against engine evidence only"


def _checkable_text(rec):
    """The goal's own name (e.g. "Bike 150cc") is quoted verbatim and is not a numeric claim."""
    text = str(rec.get("text", ""))
    name = rec.get("evidence", {}).get("name") if isinstance(rec.get("evidence"), dict) else None
    if isinstance(name, str) and name.strip():
        text = text.replace(name, " ")
    return text


def audit(rec, df, ctx=None):
    issues = []
    text = _checkable_text(rec)
    storm = rec["evidence"].get("storm") if isinstance(rec.get("evidence"), dict) else None
    for c in rec["claims"]:
        r = check_claim(c, df, ctx)
        if isinstance(c, dict) and c.get("kind") in ("storm_peak", "storm_peak_after_cut") \
                and isinstance(storm, dict) and storm.get("start") and storm.get("end"):
            if not (str(storm["start"]) <= str(c.get("date")) <= str(storm["end"])):
                r = {"ok": False, "actual": None}  # claim is about a day outside the storm being reported
        if r["actual"] is None:
            issues.append({"type": "unverifiable", "claim": c})
        elif not r["ok"]:
            issues.append({"type": "mismatch", "claim": c, "actual": round(r["actual"], 2)})
    for n in ungrounded_numbers(text, rec["claims"], rec["evidence"]):
        issues.append({"type": "ungrounded_number", "value": n})
    for c in claims_missing_from_text(text, rec["claims"]):
        issues.append({"type": "claim_not_in_text", "claim": c})
    for typ, c in weather_text_issues(text, rec["claims"]):
        if not any(i["type"] == "claim_not_in_text" and i["claim"] is c for i in issues):
            issues.append({"type": typ, "claim": c})
    return issues


def _describe_one(i):
    if i["type"] == "mismatch":
        return f"mismatch: claimed {i['claim'].get('value')} vs actual {i['actual']}"
    if i["type"] in ("claim_not_in_text", "claim_not_percent", "claim_order"):
        return f"{i['type']}: {i['claim'].get('kind')} {i['claim'].get('value')}"
    return f"{i['type']}: {i.get('claim') or i.get('value')}"


def _describe(issues):
    return "; ".join(_describe_one(i) for i in issues)


def verify_all(recs, df, llm=None, emit=None, ctx=None):
    emit = emit or (lambda kind, summary, data=None: None)
    kept, dropped = [], []
    for rec in recs:
        rec["verification"] = "recomputed" if rec["claims"] else "evidence_only"
        issues = audit(rec, df, ctx)
        if not issues:
            if rec["claims"]:
                rec.update(verification="recomputed", verified=True,
                           verifier_notes="all claims recomputed from transactions")
                emit("ok", f"verified (recomputed): {rec['action']}", {"rec": rec["id"]})
            else:
                rec.update(verification="evidence_only", verified=True, verifier_notes=EVIDENCE_NOTE)
                emit("ok", f"verified (evidence only): {rec['action']}", {"rec": rec["id"]})
            kept.append(rec)
            continue
        emit("caught", f"{rec['action']}: {_describe(issues)}", {"rec": rec["id"], "issues": issues})
        if any(i["type"] == "unverifiable" for i in issues):
            rec.update(verified=False, verifier_notes="dropped: " + _describe(issues))
            emit("dropped", f"dropped: {rec['action']} (claim cannot be recomputed)", {"rec": rec["id"]})
            dropped.append(rec)
            continue
        notes = "mismatch caught: " + _describe(issues)
        fixed = copy.deepcopy(rec["claims"])
        for i in issues:
            if i["type"] == "mismatch":
                for c in fixed:
                    if c == i["claim"]:
                        c["value"] = i["actual"]
        rewritten = None
        if llm is not None:
            try:
                rewritten = llm.json(REWRITE_SYSTEM, json.dumps(
                    {"evidence": rec["evidence"], "claims": fixed, "bad_text": rec["text"],
                     "issues": [str(i) for i in issues]}, default=str)).get("text")
            except (LLMUnavailable, ValueError, AttributeError, TypeError):
                rewritten = None
        if not isinstance(rewritten, str) or not rewritten.strip():
            rewritten = None
        attempts = ([(rewritten, "LLM rewrite")] if rewritten else []) + [(None, "template")]
        for candidate_text, how in attempts:
            rec["claims"] = fixed
            rec["text"] = candidate_text if candidate_text else template_text(rec)
            if not candidate_text:
                rec["model"] = "template"
            if not audit(rec, df, ctx):
                rec.update(verified=True, verifier_notes=f"{notes}; corrected via {how}"
                           + ("; text is deterministic template" if how == "template" else ""))
                emit("corrected", f"corrected via {how}: {rec['action']}", {"rec": rec["id"]})
                kept.append(rec)
                break
        else:
            rec.update(verified=False, verifier_notes=notes + "; could not be grounded")
            emit("dropped", f"dropped: {rec['action']}", {"rec": rec["id"]})
            dropped.append(rec)
    return kept, dropped
