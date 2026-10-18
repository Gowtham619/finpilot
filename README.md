# FinPilot

A multi-agent personal-finance copilot (Agent Arena 2026, Track 02). You upload a bank-statement CSV (or load a demo persona); a LangGraph coordinator runs a team of agents that categorize transactions, find anomalies and wasteful subscriptions, forecast your balance with uncertainty, simulate purchases, and give advice in which every number is re-checked against your raw transactions before it is shown.

Deployed demo: see "Before you submit" at the end (not deployed yet).

## What it does

1. **Ingest.** Upload a CSV (`date, description, amount`, plus an opening balance) or click **Load persona** to load one of three synthetic personas (priya, arjun, meera).
2. **Categorize.** Deterministic rules label what they can; the LLM labels the rest; your corrections override both.
3. **Analyze.** Engines find large charges, duplicate charges, category spikes, recurring subscriptions, likely-unused subscriptions and payday-driven spending.
4. **Forecast.** A weekday-profile model plus 1,000 weekday-matched bootstrap Monte Carlo paths give p10/p50/p90 balance bands, overdraft probability and goal ETAs.
5. **Advise.** The Advisor turns findings into specific actions (e.g. cancel a subscription, dispute a duplicate charge), each carrying finding ids, evidence and numeric claims.
6. **Verify.** The Verifier recomputes the claims from raw transactions and checks the numbers in the text. Mismatches are corrected or the recommendation is dropped. Only then is it displayed.
7. **Ask.** "Can I afford it?" simulates a purchase (optionally financed) against the Monte Carlo futures; "Decision duel" runs two options on identical random futures; Chat answers questions from read-only query tools; Goals track a target balance by a deadline.
8. **Nudges.** A scheduler sends a weekly digest and a daily mid-month pace check (email only if SMTP is configured; otherwise stored for the UI).

A live trace panel streams every agent step (SSE) as it happens.

## Architecture

Principle: **engines compute, the LLM interprets, the Verifier checks.** All arithmetic lives in deterministic, unit-tested engines (`finpilot/engines/`). The LLM only labels unmatched merchants, writes narrative and drafts advice text. Every agent has a no-LLM fallback path (templates), so the app works without a key.

| Agent | Job | Engines / tools | LLM role |
|---|---|---|---|
| Categorizer | Label transactions | merchant normalizer + keyword rules, user corrections | labels merchants the rules miss |
| Analyzer | Anomalies, subscriptions, behavior | `engines/analyze.py` | writes a short overview |
| Forecaster | Trajectory and goals | `engines/forecast.py` (weekday profile, Monte Carlo, goal ETA) | explains the result |
| Simulator | What-if / duel | `engines/simulate.py` | narrates the verdict |
| Advisor | Specific actions | builds candidates from findings and forecast | drafts text and claims for each candidate |
| Verifier | Audit | `engines/verify.py` (recompute claims from raw transactions) | optional rewrite of corrected text |
| Chat | Q&A | read-only query tools over the user's data | picks a tool and phrases the answer |

Coordinator: `finpilot/graph.py` is a LangGraph `StateGraph` that routes on request type.

```
START --route--> pipeline: categorize -> analyze -> forecast -> advise -> verify -> persist -> END
              \-> afford:  prepare -> afford (Simulator) -> END
              \-> duel:    prepare -> duel   (Simulator) -> END
              \-> chat:    prepare -> chat   (Chat)      -> END
```

`prepare` fits the forecast model if it is not already in state. The API (`finpilot/api.py`) is FastAPI; state lives in SQLite via SQLAlchemy; the UI is a single page of vanilla JS with Chart.js (loaded from a CDN).

### How the agents cooperate (one CSV upload)

`POST /api/upload` parses and validates the CSV and starts the pipeline in a background thread; the browser follows `/trace/{run_id}/stream`. Trace events appear in this order: Categorizer (`N by rule, M by LLM, K by your corrections, J uncategorized`) -> Analyzer (engines, then overview) -> Forecaster (fit and 1,000 paths) -> Advisor (candidate actions, then drafted recommendations) -> Verifier (one `ok`, `caught`, `corrected` or `dropped` event per recommendation) -> coordinator (`saved results`). The dashboard then renders from `/api/state/{profile}`; each recommendation has a **Why?** panel with finding ids, evidence JSON and the verifier note.

### Verifier: guarantees and limits

What it does, for each recommendation:
- Recomputes every claim (`monthly_cost`, `annual_cost`, `month_total`, `pct_change`, `txn_amount`) from the raw transactions and compares with the stated value (tolerance: the larger of 1 or 2% of the true value; 3 points for `pct_change`).
- Extracts the numbers from the advice text and requires each to match a claim value or a number in the engine evidence within 1%. Dates and a few constants (12, 24, 30, 60, 90, 100) are ignored.
- On a mismatch it fixes the claims to the recomputed values and rewrites the text (LLM rewrite if available and itself clean, else a template). A claim that cannot be recomputed drops the recommendation.

What it does not guarantee:
- Recommendations of type `behavior_trigger`, `forecast_risk` and `goal_behind` have no recomputable claims. They are marked `verification: evidence_only`: their text numbers are checked only against the engine's own evidence, not against raw transactions.
- Numbers below 10 are not checked. A `$` amount is not treated as a currency token (only `₹`, `Rs`, `INR`), so `$5000` is treated as the bare number 5000 (still compared, but not recognised as a currency amount).
- The 1% tolerance means a number within 1% of a real value passes.
- It checks that numbers are consistent with the data, not that the advice is wise or that the narrative text from other agents (overview, forecast narrative) is correct.

**Example of a Verifier catch (scripted LLM in tests, not a live-model run).** `tests/test_graph.py::test_pipeline_verifier_catches_llm_hallucination` scripts a fake LLM that triples every claim value and appends "save ₹99999". This is the Verifier trace from running it on the `generate(seed=0)` data:

```
caught     | Cancel HOTSTAR: mismatch: claimed 897.0 vs actual 299.0; mismatch: claimed 10764.0 vs actual 3588.0; ungrounded_number: 99999.0
corrected  | corrected via template: Cancel HOTSTAR
caught     | Dispute duplicate charge at AMAZON: mismatch: claimed 6597.0 vs actual 2199.0; ungrounded_number: 99999.0
corrected  | corrected via template: Dispute duplicate charge at AMAZON
result     | 8 verified, 0 dropped
```

Reproduce: `python -m pytest tests/test_graph.py -k hallucination -q`. How often a real model produces such mismatches has not been measured here.

## Mesh API usage

All LLM traffic goes through one class, `finpilot/llm.py` (`LLM`), which uses the OpenAI SDK pointed at Mesh API: `base_url = https://api.meshapi.ai/v1` (defined in `finpilot/config.py`), key from `MESH_API_KEY`, model from `FINPILOT_MODEL` (default `openai/gpt-4o-mini`), temperature 0, one retry. With no key or on failure it raises `LLMUnavailable` and each agent falls back to its deterministic path.

## Self-evaluation

Source of truth: `eval/results.md` and `eval/results.json`. The committed run is `python -m eval.run_eval --no-llm`; **the LLM-dependent sections have NOT been run** because no `MESH_API_KEY` was available in the build environment. Cells show mean (min-max) across the three synthetic personas.

### Categorization
| Method | Accuracy | Macro-F1 | Coverage |
|---|---|---|---|
| rules_only | 0.8578 (0.8522-0.8685) | 0.7646 (0.7599-0.7683) | 0.8578 (0.8522-0.8685) |
| hybrid (rules + LLM) | not run: to be filled by `make eval` with a Mesh key | | |

### Detection (precision / recall; engines scored on oracle categories)
| Task | Precision | Recall |
|---|---|---|
| large_charge | 1.0 | 1.0 |
| duplicate | 1.0 | 1.0 |
| category_spike | 0.3333 | 1.0 |
| subscriptions | 1.0 | 1.0 |
| unused_subscriptions | 1.0 | 1.0 |
| payday_trigger | 1.0 | 1.0 |

`category_spike` precision is capped at 1/3 by construction: it is precision@3 (`max_findings=3`) and each persona has one planted spike. Perfect scores elsewhere are on synthetic data where the same generator planted the targets; they show the engines work on that data, not that they generalize.

### Forecast vs baselines (held-out months; 3 personas x 3 origins = 9 months)
| Method | Balance MAPE | Month-end MAE / monthly spend | Pinball / monthly spend | Net-flow APE | p10-p90 coverage (nominal 0.80) |
|---|---|---|---|---|---|
| model | 0.0432 (0.0322-0.0556) | 0.1339 (0.0964-0.1795) | 0.0247 (0.0225-0.027) | 1.0089 (0.6762-1.304) | 0.805 (0.7627-0.8477) |
| flat_balance | 0.2325 (0.2069-0.2802) | 0.1619 (0.0999-0.2777) | 0.2404 (0.2009-0.2963) | 1.0 (1.0-1.0) | - |
| seasonal_naive (last-month replay) | 0.044 (0.0241-0.0569) | 0.1024 (0.0825-0.1318) | 0.0381 (0.0307-0.0513) | 0.9657 (0.3661-1.7367) | - |
| trailing-30d-mean (spikes on salary) | 0.4239 (0.3198-0.5464) | 0.8297 (0.7413-0.8841) | 0.4076 (0.3644-0.4528) | 6.1373 (2.8037-7.8293) | - |

Honest reading:
- On balance MAPE the model (0.0432) beats flat-balance (0.2325) and the trailing-30d-mean (0.4239) baselines, and **ties** the seasonal-naive baseline (0.0440).
- On month-end MAE relative to monthly spend the model is **worse** than seasonal-naive (0.134 vs 0.102).
- The model's p10-p90 band covers 0.805 of held-out daily points against a nominal 0.80, and its pinball loss is lower than the baselines'. The baselines are point forecasts used as all three quantiles, so that comparison favours the model.
- The net-flow APE of about 1.0 equals a flat forecast (exactly 1.0 by definition). It should not be read as skill.
- n = 9 forecast months from three personas drawn from one generator: the differences are indicative, not statistically tested.

### Calibration
Event: balance falls more than 10% below its starting level within 30 days. n = 102 (overlapping origins, so the effective n is far smaller), mean predicted 0.6833, observed 0.7255, Brier 0.096 against 0.1992 for a constant base-rate forecast, Brier skill score 0.5182.

| Predicted bin | Count | Mean predicted | Observed |
|---|---|---|---|
| 0.0-0.2 | 18 | 0.0457 | 0.2222 |
| 0.2-0.4 | 10 | 0.2782 | 0.2 |
| 0.4-0.6 | 9 | 0.5078 | 0.5556 |
| 0.6-0.8 | 7 | 0.7174 | 1.0 |
| 0.8-1.0 | 58 | 0.9741 | 0.9655 |

The skill score beats the base rate, but the reliability bins show weakness: the model under-predicts in the low and 0.6-0.8 bins (0.0457 predicted vs 0.2222 observed; 0.7174 vs 1.0), with small counts.

### Grounding and ablation (LLM-only vs rules-only vs hybrid)
Not run. Produced by `make eval` (equivalently `python -m eval.run_eval`) with `MESH_API_KEY` set; results are written to `eval/results.md` and `eval/results.json`. No numbers are claimed here.

### Advice backtest (last 90 days; assumes every recommendation was followed)
| Persona | Cancel savings (headline) | % of spend | Estimated excess over median (noisy upper bound) | Cancelled (correct/total) | Unused in truth | Recall |
|---|---|---|---|---|---|---|
| priya | ₹5,397 | 3.1% | ₹3,528 (2.1%) | 2/2 | 2 | 2/2 |
| arjun | ₹5,397 | 3.5% | ₹13,754 (9.1%) | 2/2 | 2 | 2/2 |
| meera | ₹897 | 0.6% | ₹2,976 (1.9%) | 1/1 | 1 | 1/1 |

This is an optimistic upper bound: it assumes the user complied with every recommendation on the cutoff day, and each persona has only 1-2 cancellations, so cancel precision/recall are anecdotal.

### Real-data check
Not run: no dataset supplied (see `eval/real_data.py`).

### Limitations
- All data is synthetic and comes from one generator that also planted the detection targets; the three personas are not independent samples.
- Detection, forecast and calibration engines are scored on oracle ground-truth categories, so end-to-end quality is bounded by categorization accuracy (0.86 rules-only; hybrid not measured).
- Unused-subscription detection is a heuristic.
- Tiny samples: 9 forecast months, 1-2 cancellations per persona.
- The full list is in `eval/results.md` ("Methodology and limitations").

## Key design decisions

- **Hybrid categorization.** Rules first (cheap, deterministic), LLM only for the remainder, user corrections override. Whether hybrid beats LLM-only is exactly what the ablation is meant to measure; it has not been run.
- **Weekday-matched bootstrap Monte Carlo.** Daily discretionary residuals are resampled within the same weekday; known recurring bills are projected exactly rather than sampled.
- **Common random numbers in the simulator.** Both sides of a duel (and baseline vs purchase) run on identical random futures, so the difference reflects the decision, not sampling noise.
- **Verifier before display.** Recommendations reach the UI only after the audit; failures are corrected or dropped, and the audit result is stored with the recommendation.
- **Degradation paths.** No key, network error or malformed model output falls back to deterministic templates; the test suite runs entirely without an LLM.
- **UI safety.** Rendering uses `textContent`, so uploaded merchant names cannot inject markup.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set MESH_API_KEY
make run                      # http://127.0.0.1:8000 ; set FINPILOT_AUTOSEED=1 to preload priya
make test                     # python -m pytest -q
make eval                     # full eval, needs MESH_API_KEY
python -m eval.run_eval --no-llm   # works without a key
```

Docker (the Dockerfile was written but could **not** be built or run in the build environment because the Docker daemon was not running; treat it as untested):

```bash
docker build -t finpilot .
docker run --rm -p 8000:8000 -e MESH_API_KEY=$MESH_API_KEY finpilot
```

Without Docker, the same autoseeded start was smoke-tested locally with `FINPILOT_AUTOSEED=1 uvicorn finpilot.api:app` (page and `/api/state/priya` returned data).

Environment variables:

| Variable | Meaning |
|---|---|
| `MESH_API_KEY` | Mesh API key (server-side only) |
| `FINPILOT_MODEL` | model name, default `openai/gpt-4o-mini` |
| `FINPILOT_DB` | SQLAlchemy URL, default `sqlite:///finpilot.db` |
| `FINPILOT_AUTOSEED` | `1` loads persona `priya` at startup (set in the Dockerfile) |
| `FINPILOT_SCHEDULER` | `1` starts the scheduler (Dockerfile sets 1; `.env.example` sets 0) |
| `SMTP_HOST`, `SMTP_PORT` (default 587), `SMTP_USER`, `SMTP_PASS`, `DIGEST_TO` | email delivery; without `SMTP_HOST` and `DIGEST_TO` digests are only stored for the UI |
| `FINPILOT_SYNC` | `1` runs pipelines synchronously (used by tests) |

Scheduler jobs (server local time, missed runs coalesced with 1 h grace): weekly digest every Monday 09:00; pace check daily 18:00 (alert skipped if the same one was raised in the last 24 h).

### Security notes
- `POST /api/digest/{profile}` is unauthenticated. The recipient is fixed to `DIGEST_TO` and cannot be chosen by the caller, but anyone who can reach the server can trigger a digest email. Do not configure SMTP on a public deployment unless that is acceptable (use a throwaway address).
- Uploads are limited to 5 MB and 50,000 rows. Profile names must match `^[A-Za-z0-9_-]{1,40}$`.
- There is no user authentication: profiles are not private to a visitor. Do not upload real financial data to a shared deployment.
- The CI workflow `.github/workflows/agent-arena-2026-checks.yml` (required by the hackathon) downloads a script from the hackathon server and runs it with the repository secrets available. Use a revocable Mesh key.

## Before you submit (owner checklist)

- [ ] Add repo secrets `MESH_API_KEY` and `SUBMISSION_TOKEN`; confirm the Actions checks pass after pushing.
- [ ] Run `make eval` with a Mesh key and paste the new categorization (hybrid), grounding and ablation numbers into this README (and commit `eval/results.*`).
- [ ] Deploy (Render/Railway/Fly from the Dockerfile; `MESH_API_KEY` as a server secret) and verify **Load persona** works from a phone. Deployed demo URL: TODO(owner): add URL after deploying
- [ ] Record the 60-second demo following `docs/DEMO.md`.
- [ ] Optional: run the real-data check (`eval/real_data.py`) on a real CSV and add a clearly labelled table.
