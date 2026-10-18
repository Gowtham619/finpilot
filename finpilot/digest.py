from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from . import db
from .agents.common import inr


def build_digest(profile):
    with db.session() as s:
        recs = [r for r in db.load_recs(s, profile) if r.get("verified")]
        fc = (db.load_forecast(s, profile) or {}).get("forecast")
    if not recs and not fc:
        return f"No data yet for {profile}."
    insight = fc["narrative"] if fc else "No forecast yet."
    best = max(recs, key=lambda r: r.get("est_monthly_savings", 0) + r.get("est_one_off", 0), default=None)
    lines = [f"FinPilot weekly digest for {profile}", "", f"Key insight: {insight}", ""]
    lines.append(f"One action: {best['text']}" if best else "One action: nothing urgent this week.")
    if best:
        lines.append(f"Why: triggered by {', '.join(best.get('finding_ids', []))} (verified against your transactions).")
    for g in (fc or {}).get("goals", []):
        if "p_met" not in g:  # error entry {"name", "error"}
            continue
        lines.append(f"Goal '{g['name']}': {g['p_met']*100:.0f}% chance of hitting {inr(g['target'])} by {g['deadline']}.")
    return "\n".join(lines)


def send_email(subject, body):
    host = os.getenv("SMTP_HOST")
    to = os.getenv("DIGEST_TO")
    if not host or not to:
        return False
    msg = EmailMessage()
    msg["Subject"], msg["To"], msg["From"] = subject, to, os.getenv("SMTP_USER", "finpilot@localhost")
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, 587) as smtp:
            smtp.starttls()
            smtp.login(os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", ""))
            smtp.send_message(msg)
        return True
    except Exception:
        return False


def deliver(profile):
    text = build_digest(profile)
    sent = send_email(f"FinPilot digest: {profile}", text)
    channel = "email" if sent else "ui"
    with db.session() as s:
        db.save_digest(s, profile, channel, text, sent)
    return {"channel": channel, "sent": sent, "content": text}
