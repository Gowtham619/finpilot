"""Upload every testdata CSV through the real API (no LLM key needed) and print what the app finds.

Run:  .venv/bin/python testdata/verify_testdata.py
Uses an in-memory database and the deterministic no-LLM path, so it is free and repeatable.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ["FINPILOT_SYNC"] = "1"
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from finpilot import api, db  # noqa: E402
from finpilot.llm import LLMUnavailable, set_llm  # noqa: E402

HERE = pathlib.Path(__file__).parent


class NoLLM:
    model = ""

    def json(self, system, user):
        raise LLMUnavailable("no key in verify run")

    def complete(self, system, user, json_mode=False):
        raise LLMUnavailable("no key in verify run")


def upload(client, path, profile, opening):
    with open(path, "rb") as fh:
        return client.post("/api/upload", data={"profile": profile, "opening_balance": str(opening)},
                           files={"file": (path.name, fh, "text/csv")})


def main():
    db.configure("sqlite://")
    set_llm(NoLLM())
    client = TestClient(api.create_app())
    manifest = json.loads((HERE / "manifest.json").read_text())
    print("=== GOOD FILES ===")
    for name, meta in manifest.items():
        path = HERE / name
        r = upload(client, path, meta["profile"], meta["opening_balance"])
        if r.status_code != 200:
            print(f"{name}: UPLOAD FAILED {r.status_code} {r.text[:120]}")
            continue
        st = client.get(f"/api/state/{meta['profile']}").json()
        types = {}
        for f in st["findings"]:
            types[f["type"]] = types.get(f["type"], 0) + 1
        print(f"\n{name}  (profile={meta['profile']}, opening={meta['opening_balance']}, rows={meta.get('rows')})")
        print(f"  balance ₹{st['balance']:,.0f} | findings {types} | recs {len(st['recs'])}")
        for rec in st["recs"][:4]:
            print(f"   - [{rec['trigger_type']}] {rec['text'][:110]}")
        for sc in ("none", "salary_skip", "purchase", "emergency"):
            w = client.get(f"/api/weather/{meta['profile']}?scenario={sc}").json()
            h = w["headline"]
            print(f"  weather[{sc:11}] p_any={h['p_any_breach']:.2f} worst={h['worst_day']} "
                  f"({h['worst_p']:.2f}) storm_days={h['n_storm_days']} | {w.get('scenario_note') or ''}")
        digest = client.post(f"/api/digest/{meta['profile']}").json()
        print(f"  digest action: {digest['content'].split('One action:')[-1].strip()[:100]}")
    print("\n=== BAD FILES (expect a clear rejection, or a safe load for injection/extreme) ===")
    for path in sorted(HERE.glob("bad_*.csv")):
        r = upload(client, path, "badtest", 0)
        detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text[:80]
        print(f"{path.name:28} -> HTTP {r.status_code}  {str(detail)[:90]}")


if __name__ == "__main__":
    main()
