# 60-second demo script

Tabs in the UI: **Dashboard, Can I afford...?, Decision duel, Recommendations, Goals, Chat**. The right-hand panel is the live agent trace. Start the app with `FINPILOT_AUTOSEED=1` (or the Docker image) and a valid `MESH_API_KEY` for the live-model version.

```
0:00  Click "Load persona". The trace panel streams: Categorizer (N by rule, M by LLM, ...) -> Analyzer
      -> Forecaster -> Advisor -> Verifier -> coordinator "saved results".
0:15  Point at the Verifier events. "ok" (green) = recomputed and verified. A red "caught" event means
      a number in a draft recommendation did not match the raw transactions and was corrected.
      See the note below: with a live model this does not happen on every run.
0:25  Dashboard tab: spend chart, forecast fan chart (p10/p50/p90), KPIs, and the Findings list
      (duplicate Amazon charge, large charge, category spikes, subscriptions).
0:35  Recommendations tab: open "Why?" on "Cancel HOTSTAR" -> finding ids, evidence JSON, verifier note.
0:45  Decision duel tab: Option A "Buy now" (e.g. 80000) vs Option B "Wait 2 months" (e.g. 80000 on a
      date two months out). Click "Run both futures": two fan charts and a winner with numbers.
0:55  Open the README "Self-evaluation" section: show the forecast-vs-baselines table and the honest
      reading under it (and the ablation table if `make eval` has been run with a key).
```

## About the "Verifier caught" step

The Verifier only fires `caught` when a draft contains a wrong number. With a real model, that may not occur on a given run: if every claim is already correct you will only see green `ok` events, and that is the correct behaviour. Do not present a clean run as a catch.

To show a catch honestly:
- If one happens in a live run, point at it.
- Otherwise, run the scripted test and say clearly on screen that the LLM is scripted to give wrong numbers:
  `python -m pytest tests/test_graph.py -k hallucination -q`
  and show the Verifier trace (the README contains the output of that scripted run). This demonstrates the mechanism deterministically; it is not a measurement of how often a real model errs.
- If the model is unavailable (no key), loading a persona still works and shows template recommendations; the Verifier then reports `verified (recomputed)` or `verified (evidence only)` for them, with no catches.

Rehearse once before recording and say what you are showing (live model vs scripted test).
