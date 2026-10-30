from __future__ import annotations

import asyncio
import math
import json
import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from . import db, service
from .trace import trace

HERE = pathlib.Path(__file__).parent
MAX_UPLOAD = 5 * 1024 * 1024


def valid_profile(profile: str) -> str:
    try:
        return service.check_profile(profile)
    except ValueError as e:
        raise HTTPException(400, str(e))


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        db.init()
        if os.getenv("FINPILOT_AUTOSEED") == "1":
            service.autoseed("priya")
        sched = None
        if os.getenv("FINPILOT_SCHEDULER") == "1":
            from .scheduler import build_scheduler
            sched = build_scheduler()
            sched.start()
        yield
        if sched:
            sched.shutdown(wait=False)

    app = FastAPI(title="FinPilot", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(HERE / "templates" / "index.html")

    @app.get("/api/personas")
    def personas():
        from eval.generate import PERSONAS
        return list(PERSONAS)

    @app.post("/api/load/{persona}")
    def load(persona: str):
        from eval.generate import PERSONAS
        if persona not in PERSONAS:
            raise HTTPException(404, "unknown persona")
        valid_profile(persona)
        profile, run_id = service.load_persona(persona)
        return {"profile": profile, "run_id": run_id}

    async def _read_upload(file):
        chunks, size = [], 0
        while chunk := await file.read(64 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "file too large (max 5 MB)")
            chunks.append(chunk)
        return b"".join(chunks)

    @app.post("/api/upload")
    async def upload(profile: str = Form(...), opening_balance: str = Form(""), file: UploadFile = File(...)):
        valid_profile(profile)
        opening = None
        if opening_balance.strip():
            try:
                opening = float(opening_balance)
            except ValueError:
                raise HTTPException(400, "opening_balance must be a number")
            if not math.isfinite(opening):
                raise HTTPException(400, "opening_balance must be a finite number")
        raw = await _read_upload(file)
        try:
            parsed = await asyncio.to_thread(service.parse_statement, raw)
        except ValueError as e:
            raise HTTPException(400, str(e))
        source = "user"
        if opening is None:
            opening, source = (parsed.opening_balance, "statement") if parsed.opening_balance is not None else (0.0, "default")
        notes = list(parsed.notes)
        if source == "default":
            notes.append("no opening balance given or found in the file; assumed 0")
        return {"profile": profile, "run_id": service.start_pipeline(profile, parsed.df, opening),
                "opening_balance": opening, "opening_balance_source": source, "notes": notes}

    @app.post("/api/import/preview")
    async def import_preview(file: UploadFile = File(...)):
        raw = await _read_upload(file)
        try:
            parsed = await asyncio.to_thread(service.parse_statement, raw)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"token": service.stash_import(parsed), **service.preview_summary(parsed)}

    @app.post("/api/import/confirm")
    async def import_confirm(body: dict):
        profile = valid_profile(str(body.get("profile", "")))
        raw_ob = body.get("opening_balance")
        try:
            opening = float(raw_ob) if raw_ob not in (None, "") else None
        except (TypeError, ValueError):
            raise HTTPException(400, "opening_balance must be a number")
        if opening is not None and not math.isfinite(opening):
            raise HTTPException(400, "opening_balance must be a finite number")
        token = str(body.get("token", ""))
        try:
            parsed = service.take_import(token)
        except ValueError as e:
            raise HTTPException(410, str(e))
        if opening is None:
            opening = parsed.opening_balance if parsed.opening_balance is not None else 0.0
        try:
            run_id = service.start_pipeline(profile, parsed.df, opening)
        except Exception:
            service.restore_import(token, parsed)  # let the user retry without re-uploading
            raise
        return {"profile": profile, "run_id": run_id, "opening_balance": opening}

    @app.get("/api/state/{profile}")
    def state(profile: str = Depends(valid_profile)):
        return service.state(profile)

    @app.get("/trace/{run_id}")
    def trace_json(run_id: str):
        return trace.since(run_id)

    @app.get("/trace/{run_id}/stream")
    async def trace_stream(run_id: str):
        async def gen():
            n = 0
            waited = 0.0
            while True:
                evs = trace.since(run_id, n)
                for e in evs:
                    n += 1
                    yield {"event": "trace", "data": json.dumps(e, default=str)}
                if trace.is_done(run_id) and not trace.since(run_id, n):
                    break
                if n == 0 and not evs:  # unknown run: do not wait forever
                    waited += 0.25
                    if waited >= 2.0:
                        break
                await asyncio.sleep(0.25)
        return EventSourceResponse(gen())

    def _ask_raw(profile, req):
        try:
            return service.request(profile, req)
        except KeyError as e:  # malformed request body (KeyError is also a LookupError)
            raise HTTPException(400, f"missing field {e}")
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    def _ask(profile, req):
        run_id, answer = _ask_raw(profile, req)
        return {"run_id": run_id, "answer": answer}

    @app.post("/api/afford/{profile}")
    def afford(body: dict, profile: str = Depends(valid_profile)):
        return _ask(profile, {**body, "type": "afford"})

    @app.post("/api/duel/{profile}")
    def duel(body: dict, profile: str = Depends(valid_profile)):
        return _ask(profile, {**body, "type": "duel"})

    @app.post("/api/chat/{profile}")
    def chat(body: dict, profile: str = Depends(valid_profile)):
        return _ask(profile, {"type": "chat", "question": str(body.get("question", ""))})

    @app.get("/api/weather/{profile}")
    def weather(profile: str = Depends(valid_profile), lifestyle: str = "1.0", horizon: str = "60",
                buffer: str | None = None, scenario: str = "none", amount: str | None = None,
                cuts: str | None = None):
        params = {"lifestyle": lifestyle, "horizon": horizon, "buffer": buffer, "scenario": scenario,
                  "amount": amount}
        if cuts:
            try:
                params["cuts"] = json.loads(cuts)
            except ValueError:
                raise HTTPException(400, "cuts must be a JSON object such as {\"dining\": 0.2}")
        run_id, answer = _ask_raw(profile, {"type": "weather", "params": params})
        return {**answer, "run_id": run_id}

    @app.get("/api/goals/{profile}")
    def goals_get(profile: str = Depends(valid_profile)):
        with db.session() as s:
            return db.list_goals(s, profile)

    @app.post("/api/goals/{profile}")
    def goals_add(body: dict, profile: str = Depends(valid_profile)):
        try:
            service.add_goal(profile, body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/digest/{profile}")
    def digest_now(profile: str = Depends(valid_profile)):
        from . import digest
        with db.session() as s:
            df, _ = db.load_df(s, profile)
        if df is None:
            raise HTTPException(404, "no data")
        return digest.deliver(profile)

    @app.get("/api/digests/{profile}")
    def digests_list(profile: str = Depends(valid_profile)):
        with db.session() as s:
            return db.list_digests(s, profile)[:20]

    @app.post("/api/rerun/{profile}")
    def rerun(profile: str = Depends(valid_profile)):
        with db.session() as s:
            df, opening = db.load_df(s, profile)
        if df is None:
            raise HTTPException(404, "no data")
        return {"profile": profile, "run_id": service.start_pipeline(profile, df[["date", "description", "amount"]], opening)}

    @app.post("/api/correct/{profile}")
    def correct(body: dict, profile: str = Depends(valid_profile)):
        from .config import CATEGORIES
        from .engines.categorize import normalize_merchant
        if body.get("category") not in CATEGORIES:
            raise HTTPException(400, "unknown category")
        raw_m = body.get("merchant")
        merchant = normalize_merchant(raw_m) if isinstance(raw_m, str) else ""
        if not merchant:
            raise HTTPException(400, "merchant is required (e.g. 'BLUE TOKAI COFFEE' or a statement description)")
        with db.session() as s:
            df, opening = db.load_df(s, profile)
            if df is None:
                raise HTTPException(404, "no data")
            db.add_correction(s, profile, merchant, body["category"])
        raw = df[["date", "description", "amount"]]
        return {"profile": profile, "run_id": service.start_pipeline(profile, raw, opening)}

    return app


app = create_app()
