# FinPilot

A multi-agent personal-finance copilot (Agent Arena 2026, Track 02). You upload a bank-statement CSV (or load a demo persona); a LangGraph coordinator runs a team of agents that categorize transactions, find anomalies and wasteful subscriptions, forecast your balance with uncertainty, simulate purchases, and give advice. Every numeric claim in a recommendation that can be recomputed is recomputed from the raw transactions before the recommendation is shown; other narration is restricted to numbers present in engine-computed evidence (see "Verifier: guarantees and limits").

Deployed demo: see "Before you submit" at the end (not deployed yet).

## What it does

1. **Ingest.** Upload a CSV (`date, description, amount`, plus an opening balance and a profile name, default `me`) or click **Load persona** to load one of three synthetic personas (priya, arjun, meera).
2. **Categorize.** Deterministic rules label what they can; the LLM labels the rest; your corrections override both. Corrections are API-only (no UI control): `POST /api/correct/{profile}` with `{"merchant": ..., "category": ...}`; the merchant is matched after the same normalization as statement descriptions (so `upi-blue tokai coffee BLR` becomes `BLUE TOKAI COFFEE`), and an empty merchant is rejected.
3. **Analyze.** Engines find large charges, duplicate charges, category spikes, recurring subscriptions, likely-unused subscriptions and payday-driven spending. Category spikes are computed on monthly totals with flagged large charges removed (they are reported separately; the Chat `month_total` tool reports the plain total including them) and never on `rent`, `income`, `uncategorized` or `other`; a spike describes a past month ("would have saved about ₹X that month"), not a recurring saving.
4. **Forecast.** A weekday-profile model plus 1,000 weekday-matched bootstrap Monte Carlo paths give p10/p50/p90 balance bands, overdraft probability and goal ETAs.
5. **Advise.** The Advisor turns findings into specific actions (e.g. cancel a subscription, dispute a duplicate charge), each carrying finding ids, evidence and numeric claims.
6. **Verify.** The Verifier recomputes each recommendation's recomputable claims from raw transactions and checks the numbers in its text. Mismatching recommendations are corrected or dropped; only then are recommendations displayed. Overview, forecast, simulator and chat narration is not recomputed from raw transactions (numbers there are restricted to engine evidence).
7. **Ask.** "Can I afford it?" simulates a purchase (optionally financed; the horizon is extended to cover the installments, up to 365 days) against the Monte Carlo futures and only suggests another date if it is really better; "Decision duel" runs two options on identical random futures and picks a winner by overdraft risk, then median end balance, then median lowest balance, then average median balance (a tie only if all are equal), always with the numbers; Chat answers questions from read-only query tools (without a key, a keyword router handles category + month questions such as "How much did I spend on food delivery in July?"); Goals track a target balance by a deadline.
8. **Nudges.** A scheduler sends a weekly digest and a daily mid-month pace check (email only if SMTP is configured). Every digest and pace alert is stored and shown in the **Digest** card on the dashboard (`GET /api/digests/{profile}`), which also has a **Send digest now** button (`POST /api/digest/{profile}`). The digest's "One action" prefers recurring savings (unused subscriptions) over one-off items such as past spikes.

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
- Recomputes every claim (`monthly_cost`, `annual_cost`, `month_total`, `pct_change`, `txn_amount`) from the raw transactions and compares with the stated value (tolerance: the larger of 1 or 2% of the true value; 3 points for `pct_change`). `month_total` and `pct_change` use the same aggregation as the spike engine (`engines/analyze.spend_matrix`: monthly category totals with the large charges detected in that data removed, a month without spend counting as 0, baseline = median over all months), so an engine finding and its recomputation agree by construction.
- Requires every claim value to be stated in the text (same tolerance as below; a `pct_change` claim is satisfied by any number in the text within tolerance, with or without `%`). Advice that omits or contradicts its own claims is flagged (`claim_not_in_text`).
- Extracts the numbers from the advice text and requires each to match a claim value or a number in the engine evidence within 1% of that value or within 1 in absolute terms, whichever is larger (so 15 vs 16 passes). A percentage must match a percent-like value (a probability times 100, a `pct_change` claim, or an evidence field named like `pct`/`share`/`rate`). Dates are not checked; bare numbers under 10 and a few bare constants (12, 24, 30, 60, 90, 100) are not checked, but they are checked when followed by `%` (or `percent`, `per cent`, `pct`, `percentage points`) or preceded by a currency token (`₹`, `Rs`, `INR`, `$`, `USD`). A goal's own name (e.g. "Bike 150cc") is not treated as a number.
- On any issue it fixes the claims to the recomputed values and rewrites the text (LLM rewrite if available and itself clean, else a template that states the claim values). A claim that cannot be recomputed drops the recommendation.

What it does not guarantee:
- Recommendations of type `behavior_trigger`, `forecast_risk` and `goal_behind` have no recomputable claims. They are marked `verification: evidence_only`: their text numbers are checked only against the engine's own evidence, not against raw transactions.
- Bare numbers below 10 (no `%`, no currency) and bare constants are not checked.
- Non-numeric statements are not checked at all: a narrative such as "your account is compromised, move your savings" contains no number and passes the number check. The only non-numeric check is in the afford narrative, which is replaced by the deterministic text if it uses a verdict word (safe/risky/unsafe) other than the computed verdict.
- The tolerance (1% or +/-1 absolute, whichever is larger) means a number slightly off a real value passes.
- It checks that numbers are consistent with the data, not that the advice is wise or that the narrative text from other agents (overview, forecast narrative) is correct.

**Example of a Verifier catch (scripted LLM in tests, not a live-model run).** `tests/test_graph.py::test_pipeline_verifier_catches_llm_hallucination` scripts a fake LLM that triples every claim value and appends "save ₹99999". This is an excerpt of the Verifier trace from running it on the `generate(seed=0)` data:

```
caught     | Dispute duplicate charge at AMAZON: mismatch: claimed 6597.0 vs actual 2199.0; ungrounded_number: 99999.0; claim_not_in_text: txn_amount 6597.0
corrected  | corrected via template: Dispute duplicate charge at AMAZON
caught     | Cancel HOTSTAR: mismatch: claimed 897.0 vs actual 299.0; mismatch: claimed 10764.0 vs actual 3588.0; ungrounded_number: 99999.0; claim_not_in_text: monthly_cost 897.0; claim_not_in_text: annual_cost 10764.0
corrected  | corrected via template: Cancel HOTSTAR
...        | (six more caught/corrected pairs)
result     | 8 verified, 0 dropped
```

Reproduce: `python -m pytest tests/test_graph.py -k hallucination -q`. How often a real model produces such mismatches has not been measured here.

## Mesh API usage

All LLM traffic goes through one class, `finpilot/llm.py` (`LLM`), which uses the OpenAI SDK pointed at Mesh API: `base_url = https://api.meshapi.ai/v1` (defined in `finpilot/config.py`), key from `MESH_API_KEY`, model from `FINPILOT_MODEL` (default `openai/gpt-4o-mini`), temperature 0, a 20-second request timeout, SDK retries disabled and one retry of our own (two attempts in total, 1 s apart). With no key or on failure it raises `LLMUnavailable` and each agent falls back to its deterministic path.

## Self-evaluation

Source of truth: `eval/results.md` and `eval/results.json`. The committed run is `python -m eval.run_eval --no-llm`; **the LLM-dependent sections have NOT been run** because no `MESH_API_KEY` was available in the build environment. Cells show mean (min-max) across the three synthetic personas.

When run with an LLM, `run_eval` first pings the model and aborts with an error if it is unavailable (so rules-only numbers are never recorded as "hybrid"), records how many rows the LLM actually labelled (`n_llm_labeled`), and computes the grounding rate only over recommendations the LLM wrote (template recommendations are excluded; with none it reports "n/a").

### Categorization
<!-- eval-results:begin categorization -->
| Method | Accuracy | Macro-F1 | Coverage |
|---|---|---|---|
| rules_only | 0.8578 (0.8522–0.8685) | 0.7646 (0.7599–0.7683) | 0.8578 (0.8522–0.8685) |
| hybrid | n/a | n/a | n/a |
<!-- eval-results:end categorization -->

### Detection (precision / recall; engines scored on oracle categories)
<!-- eval-results:begin detection -->
| Task | Precision | Recall |
|---|---|---|
| large_charge | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| duplicate | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| category_spike | 0.3333 (0.3333–0.3333) | 1.0 (1.0–1.0) |
| subscriptions | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| unused_subscriptions | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| payday_trigger | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
<!-- eval-results:end detection -->

`category_spike` precision is capped at 1/3 by construction: it is precision@3 (`max_findings=3`) and each persona has one planted spike. Perfect scores elsewhere are on synthetic data where the same generator planted the targets; they show the engines work on that data, not that they generalize.

### Forecast vs baselines (held-out months; 3 personas x 3 origins = 9 months)
<!-- eval-results:begin forecast -->
| Method | Balance MAPE | Month-end MAE / monthly spend | Pinball / monthly spend | Net-flow APE | p10-p90 coverage (nominal 0.80) |
|---|---|---|---|---|---|
| model | 0.0432 (0.0322–0.0556) | 0.1339 (0.0964–0.1795) | 0.0247 (0.0225–0.027) | 1.0089 (0.6762–1.304) | 0.805 (0.7627–0.8477) |
| flat_balance | 0.2325 (0.2069–0.2802) | 0.1619 (0.0999–0.2777) | 0.2404 (0.2009–0.2963) | 1.0 (1.0–1.0) | - |
| seasonal_naive | 0.044 (0.0241–0.0569) | 0.1024 (0.0825–0.1318) | 0.0381 (0.0307–0.0513) | 0.9657 (0.3661–1.7367) | - |
| trailing-30d-mean (spikes on salary) | 0.4239 (0.3198–0.5464) | 0.8297 (0.7413–0.8841) | 0.4076 (0.3644–0.4528) | 6.1373 (2.8037–7.8293) | - |
<!-- eval-results:end forecast -->

Honest reading:
- On balance MAPE the model (0.0432) beats flat-balance (0.2325) and the trailing-30d-mean (0.4239) baselines, and **ties** the seasonal-naive baseline (0.0440).
- On month-end MAE relative to monthly spend the model is **worse** than seasonal-naive (0.134 vs 0.102).
- The model's p10-p90 band covers 0.805 of held-out daily points against a nominal 0.80, and its pinball loss is lower than the baselines'. The baselines are point forecasts used as all three quantiles, so that comparison favours the model.
- The net-flow APE of about 1.0 equals a flat forecast (exactly 1.0 by definition). It should not be read as skill.
- n = 9 forecast months from three personas drawn from one generator: the differences are indicative, not statistically tested.

#### Forecast refinement attempt: no improvement found
We tried to improve the forecast and adopted nothing; the engine above is unchanged. Rule, fixed before scoring: adopt a change only if, over 3 personas x 3 origins on **each** generator, month-end MAE / monthly spend improves by at least 0.01, or band coverage moves at least 0.03 toward 0.80, while balance MAPE worsens by at most 0.002 (original) / 0.01 (OOD) and coverage moves at most 0.02 away from 0.80. Knobs were tuned only by inner backtests on the training slice. Model rows only (MAPE / month-end MAE per monthly spend / coverage); reproduce with `python -m eval.forecast_experiments` (full log in `eval/forecast_experiments.md`).

| Candidate | Original generator | OOD generator | Meets rule |
|---|---|---|---|
| committed engine | 0.0432 / 0.1339 / 0.805 | 0.2457 / 0.1848 / 0.7468 | - |
| a: band calibration (spread x k, k tuned in-sample) | 0.0432 / 0.1339 / 0.9408 | 0.2457 / 0.1848 / 0.9419 | no: overshoots coverage |
| b1: last amount for low-cv recurring bills | 0.0432 / 0.1338 / 0.805 | 0.2459 / 0.1853 / 0.7397 | no |
| b2: resample non-recurring credits (refunds, reimbursements) | 0.0432 / 0.1339 / 0.805 | 0.2403 / 0.1507 / 0.9131 | no: no gain on original, OOD band too wide |
| c: blend with last month's same-day spend (w tuned in-sample) | 0.0418 / 0.128 / 0.765 | 0.2457 / 0.1848 / 0.7468 | no |
| d: day-of-month profile blend (v tuned in-sample) | 0.0496 / 0.1447 / 0.6832 | 0.3075 / 0.2229 / 0.4662 | no: worse |
| b2 + c | 0.0418 / 0.128 / 0.765 | 0.2403 / 0.1507 / 0.9131 | no |
| fit window tuned in-sample (60/90/120/180 days) | 0.045 / 0.1467 / 0.8447 | 0.2222 / 0.1514 / 0.7621 | no |
| sensitivity only: fixed 60-day window | 0.0397 / 0.1075 / 0.7824 | 0.2087 / 0.1497 / 0.8159 | passes, but not eligible |

The fixed 60-day window passes the numeric rule, but choosing it would mean picking the window by looking at held-out results (120 and 180 days are mixed), and the in-sample choice of the same knob fails, so it was not adopted. Re-running the committed engine with four other Monte Carlo seeds moves month-end MAE by up to about 0.004 (original) and 0.012 (OOD) and coverage by up to 0.029, so differences of that size are noise. The main remaining error is a positive bias in discretionary spend when one-off charges or a spike month fall inside the 90-day window.

### Calibration
Event: balance falls more than 10% below its starting level within 30 days. n = 102 (overlapping origins, so the effective n is far smaller), mean predicted 0.6833, observed 0.7255, Brier 0.096 against 0.1992 for a constant base-rate forecast, Brier skill score 0.5182.

<!-- eval-results:begin calibration -->
| Predicted bin | Count | Mean predicted | Observed |
|---|---|---|---|
| 0.0-0.2 | 18 | 0.0457 | 0.2222 |
| 0.2-0.4 | 10 | 0.2782 | 0.2 |
| 0.4-0.6 | 9 | 0.5078 | 0.5556 |
| 0.6-0.8 | 7 | 0.7174 | 1.0 |
| 0.8-1.0 | 58 | 0.9741 | 0.9655 |
<!-- eval-results:end calibration -->

The skill score beats the base rate, but the reliability bins show weakness: the model under-predicts in the low and 0.6-0.8 bins (0.0457 predicted vs 0.2222 observed; 0.7174 vs 1.0), with small counts.

### Out-of-distribution check (second generator, engines unchanged)
Personas: kavya, rohan, isha. Same author wrote both generators: a robustness check, not independent validation. Planted per persona: 2 duplicates, 2 large charges, 1 category spike (travel), unused subscriptions [2, 1, 2], refunds [16, 19, 10]. Cells show mean (min–max) across personas.

<!-- eval-results:begin ood -->
| Categorization (rules-only) | Accuracy | Macro-F1 | Coverage |
|---|---|---|---|
| rules_only | 0.3327 (0.3015–0.3653) | 0.3564 (0.3169–0.4175) | 0.4201 (0.4027–0.4337) |

| Detection | Precision | Recall |
|---|---|---|
| large_charge | 0.1609 (0.1176–0.2222) | 1.0 (1.0–1.0) |
| duplicate | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| category_spike | 0.3333 (0.3333–0.3333) | 1.0 (1.0–1.0) |
| subscriptions | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| unused_subscriptions | 1.0 (1.0–1.0) | 0.3333 (0.0–1.0) |
| payday_trigger | 0.0 (0.0–0.0) | 1.0 (1.0–1.0) |

| Forecast (3 personas x 3 origins = 9 months) | Balance MAPE | Month-end MAE / monthly spend | Pinball / monthly spend | p10-p90 coverage (nominal 0.80) |
|---|---|---|---|---|
| model | 0.2457 (0.1954–0.328) | 0.1848 (0.0931–0.2978) | 0.0565 (0.0468–0.0615) | 0.7468 (0.6086–0.8491) |
| flat_balance | 0.9679 (0.743–1.1049) | 0.0885 (0.0392–0.1443) | 0.2432 (0.2378–0.2533) | - |
| seasonal_naive | 0.5423 (0.4272–0.7199) | 0.4533 (0.2882–0.5654) | 0.164 (0.1345–0.1909) | - |
| trailing_mean | 0.8749 (0.6958–0.9688) | 0.3799 (0.2338–0.588) | 0.2283 (0.2242–0.2306) | - |
<!-- eval-results:end ood -->

What dropped: the rules-only categorizer fell from 0.8578 accuracy to 0.3327 (coverage 0.4201); large-charge precision fell to 0.1609 (recall 1.0) because high-variance merchants such as rides and groceries produce many false flags; the payday trigger raised false findings where none was planted (precision 0.0); unused-subscription recall fell to 0.3333, because the engine matches service groups on exact merchant strings and merchants that normalise differently (e.g. an AUTOPAY suffix) are missed; category_spike recall stayed 1.0 with precision 0.3333 (capped by precision@3). What held: duplicates (precision 1.0, recall 1.0) and monthly subscription detection (precision 1.0, recall 1.0). Forecast: the model's balance MAPE is 0.2457 against 0.9679 for flat balance and 0.5423 for seasonal-naive, but its month-end MAE relative to monthly spend (0.1848) is worse than flat balance (0.0885), and its p10-p90 coverage is 0.7468 against a nominal 0.80. The refinement attempt above did not find an in-sample-tuned change that fixes this on both generators.

Disclosure: the second generator was adjusted once after its first draft scored 0.19 rules-only categorization accuracy (the share of rule-known mainstream merchants was raised and a landlord string containing RENT was added), so the 0.33 headline is an upper-biased view of how the rules cope with messy data, not an unbiased measurement. Further caveats: friend-reimbursement rows carry truth 'dining'; cash-withdrawal and EMI rows carry truth 'other', which the rules never emit, so they count as categorization misses; opening balances were hand-sized to keep balances plausible, and month-end MAE relative to monthly spend is scale-sensitive to that choice. Two generator truth-labelling bugs were also fixed after a first scoring run (see the methodology list in `eval/results.md`).

This is a robustness check, not independent validation: the same author wrote both generators and detection and forecast are scored on oracle categories. It says nothing about real bank data.

### Grounding and ablation (LLM-only vs rules-only vs hybrid)
Not run. Produced by `make eval` (equivalently `python -m eval.run_eval`) with `MESH_API_KEY` set; results are written to `eval/results.md` and `eval/results.json`. No numbers are claimed here.

### Advice backtest (last 90 days; assumes every recommendation was followed)
<!-- eval-results:begin backtest -->
| Persona | Cancel savings (headline) | % of spend | Estimated excess over median (noisy upper bound) | Cancelled (correct/total) | Unused in truth | Recall |
|---|---|---|---|---|---|---|
| priya | ₹5,397 | 3.1% | ₹3,528 (2.1%) | 2/2 | 2 | 2/2 |
| arjun | ₹5,397 | 3.5% | ₹13,754 (9.1%) | 2/2 | 2 | 2/2 |
| meera | ₹897 | 0.6% | ₹2,976 (1.9%) | 1/1 | 1 | 1/1 |
<!-- eval-results:end backtest -->

This is an optimistic upper bound: it assumes the user complied with every recommendation on the cutoff day, and each persona has only 1-2 cancellations, so cancel precision/recall are anecdotal.

### Real-data check
Not run: no dataset supplied (see `eval/real_data.py`).

### Limitations
- All data is synthetic. The main tables come from one generator that also planted the detection targets; the out-of-distribution check adds a second generator by the same author (a robustness check, not independent validation), and several engine results degrade on it (see that section). That generator was tuned once after a 0.19 first-draft categorization score, so its 0.33 is upper-biased, not an unbiased measurement.
- Detection, forecast and calibration engines are scored on oracle ground-truth categories, so end-to-end quality is bounded by categorization accuracy (0.86 rules-only; hybrid not measured).
- Unused-subscription detection is a heuristic.
- Tiny samples: 9 forecast months, 1-2 cancellations per persona.
- Forecast skill: the model ties seasonal-naive on balance MAPE, is worse than it on month-end error (original generator), and is worse than a flat balance on month-end error out of distribution; a measured refinement attempt found no change worth adopting (see "Forecast refinement attempt").
- The full list is in `eval/results.md` ("Methodology and limitations").

## Key design decisions

- **Hybrid categorization.** Rules first (cheap, deterministic), LLM only for the remainder, user corrections override. Whether hybrid beats LLM-only is exactly what the ablation is meant to measure; it has not been run.
- **Weekday-matched bootstrap Monte Carlo.** Daily discretionary residuals are resampled within the same weekday; known recurring bills are projected exactly rather than sampled.
- **Common random numbers in the simulator.** Both sides of a duel (and baseline vs purchase) run on identical random futures, so the difference reflects the decision, not sampling noise.
- **Verifier before display.** Recommendations reach the UI only after the audit; failing recommendations are corrected or dropped, and the audit result is stored with each recommendation. Other narration is not recomputed from raw transactions; it is restricted to numbers in engine evidence.
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

The image keeps SQLite in `/tmp` (`FINPILOT_DB=sqlite:////tmp/finpilot.db`), so data is ephemeral: each restart re-seeds the demo persona and loses uploads and corrections. The scheduler also runs inside the container (`FINPILOT_SCHEDULER=1`); it only emails if SMTP is configured.

Tested environment: local Python 3.14 with pandas 3.0.6 (335 tests pass). The Docker image targets Python 3.12 and the CI workflow uses Python 3.11; neither the Docker build nor the CI run has been tested.

Without Docker, the same autoseeded start was smoke-tested locally with `FINPILOT_AUTOSEED=1 uvicorn finpilot.api:app` (page and `/api/state/priya` returned data).

Environment variables:

| Variable | Meaning |
|---|---|
| `MESH_API_KEY` | Mesh API key (server-side only) |
| `FINPILOT_MODEL` | model name, default `openai/gpt-4o-mini` |
| `FINPILOT_DB` | SQLAlchemy URL, default `sqlite:///finpilot.db` |
| `FINPILOT_AUTOSEED` | `1` loads persona `priya` at startup (set in the Dockerfile) |
| `FINPILOT_SCHEDULER` | `1` starts the scheduler (Dockerfile sets 1; `.env.example` sets 0) |
| `SMTP_HOST`, `SMTP_PORT` (default 587), `SMTP_USER`, `SMTP_PASS`, `DIGEST_TO` | email delivery; without `SMTP_HOST` and `DIGEST_TO` digests are only stored and shown in the dashboard's Digest card |
| `FINPILOT_SYNC` | `1` runs pipelines synchronously (used by tests) |

Scheduler jobs (server local time, missed runs coalesced with 1 h grace): weekly digest every Monday 09:00 (built from the stored recommendations, not a fresh pipeline run); pace check daily 18:00 against today's date. Pace alerts are deduplicated on their stable id `pace:{category}:{month}`: at most one per id per 24 h, and after that only if the overage grew by more than 25%.

### Security notes
- `POST /api/digest/{profile}` is unauthenticated. The recipient is fixed to `DIGEST_TO` and cannot be chosen by the caller, but anyone who can reach the server can trigger a digest email. Do not configure SMTP on a public deployment unless that is acceptable (use a throwaway address).
- Input limits: uploads 5 MB and 50,000 rows; `financed_months` 0..60; simulation `horizon` 1..365 days; duel `cuts` must map a known category to a number in [0, 1]; goals need a 1-60 character name, a target in (0, 1e12] and a deadline between one year before and ten years after the data's last date. Profile names must match `^[A-Za-z0-9_-]{1,40}$`.
- There is no user authentication: profiles are not private to a visitor. Do not upload real financial data to a shared deployment.
- The CI workflow `.github/workflows/agent-arena-2026-checks.yml` (required by the hackathon) downloads a script from the hackathon server and runs it with the repository secrets available. Use a revocable Mesh key.

## Before you submit (owner checklist)

- [ ] Add repo secrets `MESH_API_KEY` and `SUBMISSION_TOKEN`; confirm the Actions checks pass after pushing.
- [ ] Run `make eval` with a Mesh key and paste the new categorization (hybrid), grounding and ablation numbers into this README (and commit `eval/results.*`).
- [ ] Deploy (Render/Railway/Fly from the Dockerfile; `MESH_API_KEY` as a server secret) and verify **Load persona** works from a phone. Deployed demo URL: TODO(owner): add URL after deploying
- [ ] Record the 60-second demo following `docs/DEMO.md`.
- [ ] Optional: run the real-data check (`eval/real_data.py`) on a real CSV and add a clearly labelled table.
