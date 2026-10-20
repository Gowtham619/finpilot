# Demo script

Numbers below are what the **seeded demo persona Priya** shows (synthetic data; balance ₹98,465; safety buffer ₹13,212). They come from the Guided-demo run recorded in the task-7a report and a fresh local run of `GET /api/weather/priya` (seed 0, 1,000 paths); they are not general results. Start the app with `make run` and `FINPILOT_AUTOSEED=1` (or the Docker image). A Mesh key is optional: without one, the template narration runs.

## (a) One click: press "Guided demo"

The button in the header plays a self-driving tour over the real UI (about 67 s). Pause, Skip step, Replay and Esc work; if a step fails the tour stops with a red caption instead of faking success. Say one sentence per beat:

| Time | Step | Caption on screen (abridged) | What to say |
|---|---|---|---|
| 0:00 | 1 Load | "Loading Priya, a synthetic persona with 12 months of bank transactions..." (the persona selected in the header) then "Pipeline done..." | "Five agents just ran live; you can see each one in the trace." |
| 0:04 | 2 Dashboard | "Balance ₹98,465. Overdraft risk over the next 60 days: 0%." then the fan chart | "A Monte Carlo of recent habits and recurring bills: p10, median, p90." |
| 0:15 | 3 Money Weather | "Normal 60 days: Clear skies. 0% chance of dipping below the safety buffer." | "Honest start: Priya is fine, so nothing is stormy." |
| 0:19 | | "Now a big purchase: Storm ahead. 77% chance of a dip..." | "A ₹94,000 purchase puts 77% of futures below the buffer." |
| 0:23 | | Autopsy, then an umbrella applied: "With the umbrella: 69% (was 77%)." | "The autopsy says which habits drive the failing futures; the umbrella is the cut with the best measured risk reduction per ₹1,000." |
| 0:34 | | Non-bill spending slider: 85% gives 35%, 70% gives 5%, back to 100% gives 69% | "Spend 30% less on groceries, dining and shopping and the storm nearly disappears." |
| 0:45 | 4 Why? | "Why? opens the evidence. Verifier (recomputed): all claims recomputed from transactions" | "Every recommendation shows its evidence and what the Verifier re-computed." |
| 0:50 | 5 Afford | "Can I afford ₹59,000 (about 60% of the balance)?" then "Verdict: safe." | "It simulates the purchase against the same futures." |
| 0:56 | 6 Duel | "spend ₹94,000 (95% of the balance) now, or on Nov 2, just after the next payday (Nov 1)" then "Result: Wait until Nov 2 wins on overdraft risk: overdraft risk 9% vs 13%..." | "Two options on identical random futures; the winner comes with numbers." |
| 1:03 | 7 Digest | "Send digest now" is clicked and a digest card appears | "A stored weekly digest, emailed only if SMTP is configured." |
| 1:07 | 8 End | "That is FinPilot..." with Replay | "Engines compute, models narrate, the Verifier checks." |

## (b) Manual 90-second path (live presentation)

1. Press **Load persona** (priya). Point at the trace panel: Categorizer, Analyzer, Forecaster, Advisor, Verifier.
2. Tab **Money Weather**: Normal 60 days is **Clear skies** (0%). Say so: this is why stress scenarios exist.
3. Click **Big purchase**: amount prefills 94,000, the chip shows **Storm ahead**, 77% headline, worst day 2026-10-31 (74.9%), first storm window 2026-10-24 to 2026-10-31.
4. Click the first **storm card**: its days highlight in the calendar. Read the **Autopsy**: top drivers are groceries (about 46% of the gap) and shopping (about 31%). It is an attribution over simulated paths, not proof of cause.
5. In **Umbrellas**, the top one is food delivery -20% (77% to 69%, about ₹1,227/month sacrifice). Click **Apply to the map**.
6. Drag the **Non-bill spending** slider (groceries, dining, shopping...) to 85% (about 35%), then 70% (about 5%), both with the umbrella applied. Without the umbrella the same slider gives about 44% and 8%.
7. **Can I afford...?**: enter 59,000, run it, read the verdict (safe) and the date options.
8. **Decision duel**: option A buy now 94,000; option B the same amount on 2026-11-02 (just after the Nov 1 payday). Click "Run both futures" and read the result line aloud (overdraft risk 9% vs 13%). The tour picks this preset by running 95%, 90% and 85% of the balance through the duel endpoint and using the first whose options visibly differ.
9. **Recommendations**: open **Why?** on "Cancel HOTSTAR" (finding ids, evidence JSON, verifier note: recomputed).
10. **Dashboard**: the **Digest** card, then **Send digest now**.

## (c) Questions judges will ask

- **Is the Verifier independent?** Partly. Money claims (subscription cost, a transaction amount) are re-computed straight from the raw transactions. Aggregations (`month_total`, `pct_change`) reuse the spike engine's code, and the weather claims (`breach_probability`, `storm_peak`, `storm_peak_after_cut`) re-run the same Monte Carlo from the raw data with seed 0. Those are consistency checks, not an independent implementation. It also cannot verify number-free prose.
- **Is the data real?** No. Everything is synthetic: three personas from one generator that also planted the anomalies, plus a second generator by the same author as a robustness check, on which several engines degrade (rules-only categorization 0.86 to 0.33). A real-data check exists (`eval/real_data.py`) but has not been run.
- **Why does it need no key?** Engines do all the arithmetic; the LLM only labels leftovers, phrases and drafts. Every agent has a deterministic template path, so the app, the Verifier and the whole test suite run without a key. The LLM-dependent evaluations (hybrid categorization, ablation, grounding rate) are therefore **not run**; they need `make eval` with a key.

## (d) If the live model produces no Verifier catch

That is normal: the Verifier only fires when a draft contains a wrong number, and a clean run shows green "ok" events. Do not present it as a catch. Show the scripted cases and say on screen that the LLM is **scripted** to lie:

```
.venv/bin/python -m pytest tests/test_weather_claims.py -k swapped -q
.venv/bin/python -m pytest tests/test_graph.py -k hallucination -q
```

The README's Trust model section prints the real output of the first one: the scripted text "Storm peak 18%, cutting groceries by 30% lowers it to 42%." is flagged `claim_order` and corrected to peak 42%, after-cut 18% (on the test fixture, Priya with a ₹10,000 balance). It shows the mechanism, not how often a real model errs, which has not been measured.

## (e) 60-second spoken script

"FinPilot is a team of agents for your money. I load Priya, a synthetic persona, and you can watch the categorizer, analyzer, forecaster, advisor and verifier work in the trace. The twist is Money Weather: the same Monte Carlo turns her next 60 days into a weather map. Today it is clear skies, which is honest. Now a ₹94,000 purchase: storm ahead, 77% of simulated futures dip below her safety buffer. The autopsy shows which habits drive those failures, and the best umbrella is the cut with the biggest measured risk reduction per thousand rupees. If she spends 30% less on groceries, dining and shopping, the risk falls to about 5%. Every recommendation has a Why panel, and the Verifier checks its numbers against values recomputed from the raw transactions where it can, otherwise against the engine's own evidence. I will be upfront that for the weather numbers it re-runs the same engine, so that is a consistency check, not independence. The data is synthetic and the app works with no API key at all. Engines compute, models narrate, the Verifier checks."
