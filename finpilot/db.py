from __future__ import annotations

import datetime as dt
import json
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String, Text, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from . import config


class Base(DeclarativeBase):
    pass


class Profile(Base):
    __tablename__ = "profiles"
    name: Mapped[str] = mapped_column(String, primary_key=True)
    opening_balance: Mapped[float] = mapped_column(Float, default=0.0)


class Txn(Base):
    __tablename__ = "transactions"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile: Mapped[str] = mapped_column(String, index=True)
    tid: Mapped[int] = mapped_column(Integer)
    date: Mapped[dt.datetime] = mapped_column(DateTime)
    description: Mapped[str] = mapped_column(Text)
    amount: Mapped[float] = mapped_column(Float)
    merchant: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    category_source: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)


class Blob(Base):
    """JSON documents keyed by profile + kind (findings, recs, forecast)."""
    __tablename__ = "blobs"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String, index=True)
    data: Mapped[dict] = mapped_column(JSON)


class Goal(Base):
    __tablename__ = "goals"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String)
    target: Mapped[float] = mapped_column(Float)
    deadline: Mapped[dt.date] = mapped_column(Date)
    history: Mapped[list] = mapped_column(JSON, default=list)


class Correction(Base):
    __tablename__ = "corrections"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile: Mapped[str] = mapped_column(String, index=True)
    merchant: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)


class Digest(Base):
    __tablename__ = "digests"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile: Mapped[str] = mapped_column(String, index=True)
    channel: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    sent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))


class TraceEvent(Base):
    __tablename__ = "trace_events"
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    seq: Mapped[int] = mapped_column(Integer)
    agent: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)
    summary: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSON)


_engine = None
_Session = None


def configure(url=None):
    global _engine, _Session
    url = url or config.DB_URL
    kw = {"json_serializer": lambda o: json.dumps(o, default=str), "connect_args": {"check_same_thread": False}}
    if url in ("sqlite://", "sqlite:///:memory:"):
        kw["poolclass"] = StaticPool
    _engine = create_engine(url, **kw)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)


@contextmanager
def session():
    if _Session is None:
        configure()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init():
    if _Session is None:
        configure()


def save_transactions(s, profile, df, opening):
    s.execute(delete(Txn).where(Txn.profile == profile))
    p = s.get(Profile, profile) or Profile(name=profile)
    p.opening_balance = float(opening)
    s.add(p)
    s.add_all([Txn(profile=profile, tid=int(i), date=r.date.to_pydatetime(), description=r.description,
                   amount=float(r.amount), merchant=r.merchant, category=r.category,
                   category_source=r.category_source, confidence=float(r.confidence))
               for i, r in df.iterrows()])


def load_df(s, profile):
    rows = s.scalars(select(Txn).where(Txn.profile == profile).order_by(Txn.tid)).all()
    p = s.get(Profile, profile)
    if not rows or p is None:
        return None, None
    df = pd.DataFrame([{"date": r.date, "description": r.description, "amount": r.amount,
                        "merchant": r.merchant, "category": r.category,
                        "category_source": r.category_source, "confidence": r.confidence} for r in rows],
                      index=[r.tid for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df, p.opening_balance


def _put(s, profile, kind, data):
    s.execute(delete(Blob).where(Blob.profile == profile, Blob.kind == kind))
    s.add(Blob(profile=profile, kind=kind, data={"v": data}))


def _get(s, profile, kind, default):
    b = s.scalars(select(Blob).where(Blob.profile == profile, Blob.kind == kind)).first()
    return b.data["v"] if b else default


def save_findings(s, profile, findings): _put(s, profile, "findings", findings)
def load_findings(s, profile): return _get(s, profile, "findings", [])
def save_recs(s, profile, recs): _put(s, profile, "recs", recs)
def load_recs(s, profile): return _get(s, profile, "recs", [])
def save_forecast(s, profile, fc): _put(s, profile, "forecast", fc)
def load_forecast(s, profile): return _get(s, profile, "forecast", None)


def add_goal(s, profile, name, target, deadline):
    s.add(Goal(profile=profile, name=name, target=float(target),
               deadline=pd.Timestamp(deadline).date(), history=[]))


def list_goals(s, profile):
    return [{"pk": g.pk, "name": g.name, "target": g.target, "deadline": str(g.deadline), "history": g.history}
            for g in s.scalars(select(Goal).where(Goal.profile == profile)).all()]


def append_goal_snapshot(s, pk, snapshot):
    g = s.get(Goal, pk)
    g.history = list(g.history or []) + [snapshot]


def add_correction(s, profile, merchant, category):
    s.execute(delete(Correction).where(Correction.profile == profile, Correction.merchant == merchant))
    s.add(Correction(profile=profile, merchant=merchant, category=category))


def corrections(s, profile):
    return {c.merchant: c.category for c in s.scalars(select(Correction).where(Correction.profile == profile))}


def save_digest(s, profile, channel, content, sent):
    s.add(Digest(profile=profile, channel=channel, content=content, sent=sent))


def list_digests(s, profile):
    rows = s.scalars(select(Digest).where(Digest.profile == profile).order_by(Digest.pk.desc())).all()
    return [{"channel": d.channel, "content": d.content, "sent": d.sent, "created_at": str(d.created_at)} for d in rows]


def save_trace(s, run_id):
    from .trace import trace
    for e in trace.since(run_id):
        s.add(TraceEvent(run_id=run_id, seq=e["seq"], agent=e["agent"], kind=e["kind"],
                         summary=e["summary"], data=e["data"]))
