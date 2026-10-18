import logging

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from . import db, digest
from .engines import pace

log = logging.getLogger(__name__)


def _today():
    return pd.Timestamp.today()


def weekly_digest_job():
    with db.session() as s:
        profiles = db.list_profiles(s)
    for p in profiles:
        try:
            digest.deliver(p)
        except Exception:  # one bad profile must not stop the rest
            log.exception("weekly digest failed for %s", p)


def pace_check_job():
    with db.session() as s:
        profiles = db.list_profiles(s)
    for p in profiles:
        try:
            with db.session() as s:
                df, _ = db.load_df(s, p)
            if df is None:
                continue
            today = _today()
            alerts = pace.pace_alerts(df, today)
            if alerts:
                a = max(alerts, key=lambda x: x["mtd"] - x["expected"])
                aid = a.get("id") or f"pace:{a['category']}:{pd.Timestamp(today).to_period('M')}"
                over = float(a["mtd"] - a["expected"])
                cat = a["category"].replace("_", " ")
                if a.get("new_spend") or a.get("over_pct") is None:
                    body = f"Pace alert: new spending in {cat} this month: ₹{a['mtd']:,.0f} by day {a['day']}."
                else:
                    body = (f"Pace alert: {cat} spend is {a['over_pct']}% above your usual by day {a['day']} "
                            f"(₹{a['mtd']:,.0f} vs ₹{a['expected']:,.0f}).")
                with db.session() as s:
                    if not db.pace_alert_due(s, p, aid, over):
                        continue  # same alert id within 24h, or the overage has not grown by >25%
                sent = digest.send_email(f"FinPilot pace alert: {p}", body)
                with db.session() as s:
                    db.save_digest(s, p, "email" if sent else "ui", body, sent)
                    db.record_pace_alert(s, p, aid, over)
        except Exception:
            log.exception("pace check failed for %s", p)


def build_scheduler():
    """Cron times use the server's local timezone. Missed runs are coalesced and allowed 1h of grace."""
    sched = BackgroundScheduler()
    kw = {"misfire_grace_time": 3600, "coalesce": True}
    sched.add_job(weekly_digest_job, "cron", day_of_week="mon", hour=9, id="weekly_digest", **kw)
    sched.add_job(pace_check_job, "cron", hour=18, id="pace_check", **kw)
    return sched
