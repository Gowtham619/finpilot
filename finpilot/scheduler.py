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
            alerts = pace.pace_alerts(df, _today())
            if alerts:
                a = alerts[0]
                body = (f"Pace alert: {a['category']} spend is {a['over_pct']}% above your usual by day {a['day']} "
                        f"(₹{a['mtd']:,.0f} vs ₹{a['expected']:,.0f}).")
                sent = digest.send_email(f"FinPilot pace alert: {p}", body)
                with db.session() as s:
                    db.save_digest(s, p, "email" if sent else "ui", body, sent)
        except Exception:
            log.exception("pace check failed for %s", p)


def build_scheduler():
    sched = BackgroundScheduler()
    sched.add_job(weekly_digest_job, "cron", day_of_week="mon", hour=9, id="weekly_digest")
    sched.add_job(pace_check_job, "cron", hour=18, id="pace_check")
    return sched
