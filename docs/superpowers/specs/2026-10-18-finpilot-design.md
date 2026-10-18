# FinPilot — Design Spec

Agent Arena 2026, Track 02. Submission deadline: Oct 30, 2026 11:00 PM IST.

## 1. Goal

A multi-agent system that ingests transactions, categorizes them, finds real
patterns and anomalies, forecasts the user's trajectory, simulates what-if
decisions, and proactively gives specific, auditable recommendations. Every
claim is verified against the data, and accuracy is measured and published.

**Principle:** deterministic engines compute every number; LLM agents interpret,
explain, and decide. No number in any recommendation originates from the LLM.

## 2. Constraints

- All LLM calls go through Mesh API (`https://api.meshapi.ai/v1`, `MESH_API_KEY`)
  via the OpenAI SDK, in a single module `llm.py`. Default model
  `openai/gpt-4o-mini`, overridable by `FINPILOT_MODEL`.
- Python, FastAPI, SQLite (SQLAlchemy), LangGraph, APScheduler, Jinja2 + HTMX.
- `requirements.txt` lists fastapi and openai. `.env` is gitignored; `.env.example` is committed.
- CI workflow `.github/workflows/agent-arena-2026-checks.yml` from the hackathon.

## 3. Architecture

LangGraph state machine. A **Coordinator** routes by request type:

- CSV upload / sync: `categorize → analyze → forecast → advise → verify → persist`
- "Can I afford X?": `simulate → advise(optional) → verify`
- Chat question: `chat`
- Scheduled run: full pipeline, then `digest`

| Agent | Job | Tools | Notes |
|---|---|---|---|
| Categorizer | Label transactions | merchant normalizer, rule matcher, LLM batch fallback | cache by merchant; user corrections override and become rules |
| Analyzer | Patterns, anomalies, behavior | `monthly_aggregates`, `zscore_anomalies`, `detect_subscriptions`, `unused_subscriptions`, `behavior_triggers` | emits `findings` rows with stable IDs |
| Forecaster | Trajectory | `project_cashflow`, `monte_carlo`, `goal_eta` | explains result and uncertainty |
| Simulator | What-if | `simulate_purchase`, `best_date`, `cheapest_fix` | runs on forecaster engine |
| Advisor | Specific actions | reads findings + forecast | must cite finding IDs; uncited output rejected |
| Verifier | Audit | `recompute(claim)` from raw transactions | independent prompt and tools; rejects or rewrites mismatches |
| Chat | Free-form Q&A | read-only query tools | answers only from tool results |

## 4. Engines (deterministic, unit-tested)

- **Categorization rules:** keyword/regex on normalized merchant plus amount and cadence cues (large monthly debit → rent). Unmatched → LLM batch.
- **Anomalies:** per-category robust z-score (median/MAD) on monthly and per-transaction amounts; duplicate-charge detector (same merchant, amount, within 48h); new-merchant large-spend flag.
- **Subscriptions:** recurring-charge detection by merchant, amount stability, and cadence (weekly/monthly/annual). "Unused" = recurring charge in a category with no usage signal (e.g. streaming/gym with no related activity) or overlapping services; heuristic documented in README.
- **Behavior triggers:** spend clusters by weekday, hour (when present), days-since-payday; yields statements like "delivery spend is 2.3x higher in the 3 days after payday".
- **Forecast:** known recurring bills projected exactly; discretionary spend per category from trailing mean with weekday/month-end profile; income from detected recurring credits.
- **Monte Carlo:** ~1,000 paths resampling discretionary residuals. Outputs p10/p50/p90 balance bands, P(overdraft), P(goal met by deadline).
- **Simulator:** inject outflow (purchase, financed purchase, goal contribution) into the paths; report balance floor, P(overdraft), goal delay. `best_date` scans candidate dates; `cheapest_fix` searches the smallest category cuts that bring risk under a threshold.
- **Pace alerts:** mid-month, compare month-to-date spend per category to the forecast pace; flag when projected month total exceeds the user's norm by a set margin.

## 5. Verifier and audit trail

Every recommendation row stores: `text`, `action`, `est_monthly_savings`,
`trigger_type`, `finding_ids`, `evidence` (JSON of the metric values),
`verified` (bool), `verifier_notes`, `model`, `created_at`.

The Verifier extracts each numeric claim from advisor output, recomputes it from
raw transactions, and compares within tolerance. Mismatch → advisor retry once
with the correction → otherwise the recommendation is dropped and logged. The UI
shows a "Why?" panel per recommendation with the triggering evidence and the
verifier result.

## 6. Action-ready advice

For each unused subscription: drafted cancellation message and the monthly/annual
saving. Delivered as copyable text (no auto-sending anything on the user's behalf).

## 7. Data model (SQLite)

`transactions`, `findings`, `forecasts`, `recommendations`, `goals` (+ progress
snapshots per run so advice adapts when behind/ahead), `digests`, `corrections`.

## 8. Delivery and UI

- APScheduler weekly digest job + mid-month pace-alert job. Channels: Telegram
  bot (primary; reply with a question routes to Chat) and SMTP email. If neither
  is configured, digests are stored and shown in the UI. `--dry-run` for demos.
- Pages: upload, dashboard (category spend, anomalies, forecast fan chart),
  "Can I afford…?" simulator, goals, recommendations with "Why?", chat.

## 9. Self-evaluation (results committed to `eval/results.json` and README)

Synthetic generator `eval/generate.py` creates several personas with known ground
truth: true categories, planted anomalies and duplicate charges, planted unused
subscriptions, planted behavior triggers, and held-out future months.

| Metric | Method |
|---|---|
| Categorization | accuracy, macro-F1, confusion matrix |
| Anomaly detection | precision/recall vs planted |
| Subscription detection | precision/recall vs planted |
| Forecast | MAPE on held-out month; p10–p90 coverage; vs naive baseline |
| Simulation | calibration: predicted P(overdraft) vs frequency in held-out outcomes |
| Grounding | % of claims verified; hallucinated claims caught |
| **Ablation** | LLM-only vs rules-only vs hybrid on the above |
| **Advice backtest** | replay history as if advice were followed 3 months ago; report balance delta |

`make eval` regenerates all numbers; CI runs a small eval with the LLM stubbed.

## 10. Errors and degradation

Mesh failure/rate limit: retry once, then degrade. Categorizer falls back to rules
(unmatched = "uncategorized"); Advisor falls back to templated text from evidence;
Verifier still runs. The pipeline never crashes on LLM failure.

## 11. Testing

pytest unit tests per engine on synthetic data, LLM stub for CI, one end-to-end
pipeline test, one verifier test with a deliberately wrong claim.

## 12. Build order and cut line

1. Repo, CI, LLM client, DB, synthetic generator (all depends on this)
2. Categorizer + eval
3. Analyzer engines + eval
4. Forecast + Monte Carlo + simulator + eval
5. Advisor + Verifier + audit trail + grounding eval
6. LangGraph wiring + FastAPI UI
7. Goals, Chat, scheduler/digest
8. Ablation table + advice backtest
9. **Cut line — drop in this order if short on time:** action-ready drafts, Telegram (keep email/UI), behavior triggers, pace alerts.
10. README, design-decisions note, results, demo script

## 13. Out of scope

Bank integrations, auth/multi-user, real payments, auto-sending cancellations,
ML-trained models.
