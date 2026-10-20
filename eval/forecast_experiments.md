# Forecast refinement experiments

Each candidate wraps forecast.fit / forecast.monte_carlo as seen by the committed harness
(eval.run_eval.eval_forecast): same generators, same 3 monthly origins per persona, same metric code.
Nothing here changes finpilot/engines/forecast.py; this is an experiment log, not shipped behaviour.

Tuned knobs (k, w, v, window) are chosen ONLY by inner backtests on the training slice: for a held-out
cutoff, the inner origins are the first days of the last 3 complete months before the cutoff (each with
>= 60 days of earlier history); the model is refit on data before each inner origin and scored on that
inner month. No held-out month is used to choose anything. The fixed-window rows (60/120/180 days) are a
sensitivity analysis whose value would have to be picked by looking at held-out results, so they are
NOT eligible for adoption whatever they score.

Adoption rule (fixed before any candidate was scored): adopt only if, averaged over 3 personas x 3
origins on EACH generator, month-end MAE / monthly spend improves by >= 0.01 on each generator OR band
coverage moves toward 0.80 by >= 0.03 on each generator, AND balance MAPE worsens by no more than 0.002
(original) / 0.01 (OOD) AND coverage moves away from 0.80 by no more than 0.02 on either generator.

| Candidate | Orig MAPE | Orig month-end MAE/spend | Orig coverage | OOD MAPE | OOD month-end MAE/spend | OOD coverage | Meets rule | Knob values chosen |
|---|---|---|---|---|---|---|---|---|
| baseline (committed engine) | 0.0432 | 0.1339 | 0.805 | 0.2457 | 0.1848 | 0.7468 | - | - |
| a: band calibration, k tuned in-sample | 0.0432 | 0.1339 | 0.9408 | 0.2457 | 0.1848 | 0.9419 | no | k in [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5] |
| b1: last amount for recurring items with cv<=0.1 | 0.0432 | 0.1338 | 0.805 | 0.2459 | 0.1853 | 0.7397 | no | - |
| b2: resample non-recurring credits (refunds etc.) | 0.0432 | 0.1339 | 0.805 | 0.2403 | 0.1507 | 0.9131 | no | - |
| c: seasonal blend, w tuned in-sample | 0.0418 | 0.128 | 0.765 | 0.2457 | 0.1848 | 0.7468 | no | w in [0.0, 0.25] |
| d: day-of-month profile blend, v tuned in-sample | 0.0496 | 0.1447 | 0.6832 | 0.3075 | 0.2229 | 0.4662 | no | v in [0.0, 0.25, 0.5, 0.75, 1.0] |
| b2+c combined | 0.0418 | 0.128 | 0.765 | 0.2403 | 0.1507 | 0.9131 | no | w in [0.0, 0.25] |
| e: fit window tuned in-sample (60/90/120/180) | 0.045 | 0.1467 | 0.8447 | 0.2222 | 0.1514 | 0.7621 | no | win in [60, 90, 120, 180] |
| e [task 5b, pre-registered, parameter-free]: robust bootstrap pool (drop flagged large charges + later duplicate) | 0.0322 | 0.08 | 0.8843 | 0.2287 | 0.1445 | 0.5796 | no | - |
| sensitivity: fixed 60-day window | 0.0397 | 0.1075 | 0.7824 | 0.2087 | 0.1497 | 0.8159 | yes (not eligible) | - |
| sensitivity: fixed 120-day window | 0.0427 | 0.1365 | 0.8517 | 0.1986 | 0.1246 | 0.7906 | no (not eligible) | - |
| sensitivity: fixed 180-day window | 0.0379 | 0.1164 | 0.8948 | 0.2199 | 0.1503 | 0.7147 | no (not eligible) | - |
| noise floor: committed engine, MC seed 1 | 0.0431 | 0.1317 | 0.8121 | 0.2477 | 0.1768 | 0.751 | - | - |
| noise floor: committed engine, MC seed 2 | 0.0421 | 0.1316 | 0.8123 | 0.2412 | 0.1789 | 0.7471 | - | - |
| noise floor: committed engine, MC seed 3 | 0.0428 | 0.1321 | 0.8337 | 0.2434 | 0.1727 | 0.7435 | - | - |
| noise floor: committed engine, MC seed 4 | 0.0416 | 0.1302 | 0.8157 | 0.2475 | 0.1834 | 0.7684 | - | - |
| noise floor: candidate e robust_pool=True, MC seed 1 | 0.032 | 0.0819 | 0.8661 | 0.2313 | 0.1454 | 0.5761 | - | - |
| noise floor: candidate e robust_pool=True, MC seed 2 | 0.0322 | 0.0821 | 0.8624 | 0.2241 | 0.1414 | 0.5723 | - | - |
| noise floor: candidate e robust_pool=True, MC seed 3 | 0.0315 | 0.0803 | 0.866 | 0.2281 | 0.1458 | 0.5794 | - | - |
| noise floor: candidate e robust_pool=True, MC seed 4 | 0.0317 | 0.0794 | 0.8551 | 0.2287 | 0.1442 | 0.5905 | - | - |

## Candidate f: robust centre, full-pool spread (task 5c)

Candidate f (task 5c, pre-registered, parameter-free; third candidate family, motivated by the failure of
candidate e on the same held-out months): p50 from `fit(robust_pool=True)`, p10/p90 = robust p50 + (full-pool
p10/p90 - full-pool p50), both simulations with the same seed (`forecast.central_robust_bands`). Rule, fixed before
scoring: adopt only if, averaged over 3 personas x 3 origins on EACH generator, month-end MAE / monthly spend improves
by >= 0.01, AND balance MAPE worsens by no more than 0.002 (original) / 0.01 (OOD), AND |coverage - 0.80| exceeds the
committed model's |coverage - 0.80| by no more than 0.02; the MAE gain must exceed the spread across Monte Carlo seeds
0-4; and the same rule must also hold on fresh personas never used in any attempt (original generator seeds 11, 12,
13; OOD generator seeds 21, 22, 23; other parameters copied from the committed personas in order; these personas are
NOT added to the committed sets). Fresh seeds of the same generators are not independent data: they share the
generators' author and design.

| Data | MC seed | Method | Orig MAPE | Orig month-end MAE/spend | Orig coverage | OOD MAPE | OOD month-end MAE/spend | OOD coverage |
|---|---|---|---|---|---|---|---|---|
| committed personas | 0 | model | 0.0432 | 0.1339 | 0.805 | 0.2457 | 0.1848 | 0.7468 |
| committed personas | 0 | model_robust_centre | 0.0322 | 0.08 | 0.9417 | 0.2287 | 0.1445 | 0.8846 |
| committed personas | 1 | model | 0.0431 | 0.1317 | 0.8121 | 0.2477 | 0.1768 | 0.751 |
| committed personas | 1 | model_robust_centre | 0.032 | 0.0819 | 0.9306 | 0.2313 | 0.1454 | 0.8486 |
| committed personas | 2 | model | 0.0421 | 0.1316 | 0.8123 | 0.2412 | 0.1789 | 0.7471 |
| committed personas | 2 | model_robust_centre | 0.0322 | 0.0821 | 0.9342 | 0.2241 | 0.1414 | 0.8667 |
| committed personas | 3 | model | 0.0428 | 0.1321 | 0.8337 | 0.2434 | 0.1727 | 0.7435 |
| committed personas | 3 | model_robust_centre | 0.0315 | 0.0803 | 0.9377 | 0.2281 | 0.1458 | 0.8594 |
| committed personas | 4 | model | 0.0416 | 0.1302 | 0.8157 | 0.2475 | 0.1834 | 0.7684 |
| committed personas | 4 | model_robust_centre | 0.0317 | 0.0794 | 0.9196 | 0.2287 | 0.1442 | 0.8774 |
| fresh personas | 0 | model | 0.0419 | 0.1616 | 0.6344 | 1.6408 | 0.1882 | 0.6915 |
| fresh personas | 0 | model_robust_centre | 0.0351 | 0.1078 | 0.7635 | 1.7172 | 0.1207 | 0.8518 |

| Check | Generator | MAE gain (>= 0.01) | MAPE worsening (<= 0.002 orig / 0.01 OOD) | Coverage-gap increase (<= 0.02) | Pass |
|---|---|---|---|---|---|
| committed personas, seed 0 | original | 0.0539 | -0.011 | 0.1367 | no |
| committed personas, seed 0 | ood | 0.0403 | -0.017 | 0.0314 | no |
| fresh personas, seed 0 | original | 0.0538 | -0.0068 | -0.1291 | yes |
| fresh personas, seed 0 | ood | 0.0675 | 0.0764 | -0.0567 | no |

Seed noise (spread of the committed model's month-end MAE/spend over MC seeds 0-4): original 0.0037, OOD 0.0121. MAE gain of f across seeds 0-4: original 0.0495-0.0539, OOD 0.0269-0.0403; gain exceeds noise: yes.

Decision: NOT ADOPTED.
