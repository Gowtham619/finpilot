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
