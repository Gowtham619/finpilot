from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

from . import db
from .agents.common import inr

log = logging.getLogger("finpilot.digest")


# "One action" ranking: recurring savings first, past one-off spikes and habits last; ties by savings.
ACTION_PRIORITY = {"unused_subscription": 0, "duplicate_charge": 1, "forecast_risk": 2, "large_charge": 3,
                   "goal_behind": 4, "category_spike": 5, "behavior_trigger": 6}


def pick_action(recs):
    return min(recs, key=lambda r: (ACTION_PRIORITY.get(r.get("trigger_type"), 9),
                                    -(r.get("est_monthly_savings", 0) + r.get("est_one_off", 0))), default=None)


def verification_phrase(rec):
    """How the recommendation's numbers were checked, in words (never more than what happened)."""
    return ("recomputed from your transactions" if rec.get("verification") == "recomputed"
            else "checked against engine evidence")


def weather_line(wx):
    """One deterministic line from the stored Money Weather report (None if there is none)."""
    if not isinstance(wx, dict) or "headline" not in wx:
        return None
    from .agents.meteorologist import fallback_texts
    try:
        line = "Money Weather: " + fallback_texts(wx)["headline_text"]
        if wx.get("storms"):
            st = wx["storms"][0]  # sorted by peak risk
            line += (f" Worst storm window: {st['start']} to {st['end']} "
                     f"(peak {st['peak_p']*100:.0f}% chance of dropping below the safety buffer).")
        return line
    except (KeyError, TypeError, ValueError):
        return None


def build_digest(profile):
    with db.session() as s:
        recs = [r for r in db.load_recs(s, profile) if r.get("verified")]
        stored = db.load_forecast(s, profile) or {}
    fc, wx = stored.get("forecast"), stored.get("weather")
    if not recs and not fc:
        return f"No data yet for {profile}."
    insight = (fc or {}).get("narrative") or "No forecast yet."
    best = pick_action(recs)
    lines = [f"FinPilot weekly digest for {profile}", "", f"Key insight: {insight}"]
    wl = weather_line(wx)
    if wl:
        lines.append(wl)
    lines.append("")
    lines.append(f"One action: {best['text']}" if best else "One action: nothing urgent this week.")
    if best:
        lines.append(f"Why: triggered by {', '.join(best.get('finding_ids', []))} ({verification_phrase(best)}).")
    for g in (fc or {}).get("goals", []):
        if "p_met" not in g:  # error entry {"name", "error"}
            continue
        lines.append(f"Goal '{g['name']}': {g['p_met']*100:.0f}% chance of hitting {inr(g['target'])} by {g['deadline']}.")
    return "\n".join(lines)


def _send(subject, body):
    """Returns (sent, error_type). error_type is None when not configured or on success."""
    host, to = os.getenv("SMTP_HOST"), os.getenv("DIGEST_TO")
    if not host or not to:
        return False, None  # not configured: quiet
    try:
        port = int(os.getenv("SMTP_PORT") or 587)
        msg = EmailMessage()
        msg["Subject"], msg["To"], msg["From"] = subject, to, os.getenv("SMTP_USER", "finpilot@localhost")
        msg.set_content(body)
        with smtplib.SMTP(host, port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", ""))
            smtp.send_message(msg)
        return True, None
    except Exception as e:
        # log type and message only; never credentials
        pw = os.getenv("SMTP_PASS") or None
        detail = str(e).replace(pw, "***") if pw else str(e)
        log.warning("digest email failed: %s: %s", type(e).__name__, detail)
        return False, type(e).__name__


def send_email(subject, body):
    return _send(subject, body)[0]


def deliver(profile):
    text = build_digest(profile)
    sent, err = _send(f"FinPilot digest: {profile}", text)
    channel = "email" if sent else "ui"
    with db.session() as s:
        db.save_digest(s, profile, channel, text, sent)
    out = {"channel": channel, "sent": sent, "content": text}
    if err:
        out["error"] = err
    return out
