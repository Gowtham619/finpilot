# Evaluation results

Personas: priya, arjun, meera (synthetic, known ground truth)

Cells show mean (min–max) across personas.

## Categorization
| Method | Accuracy | Macro-F1 | Coverage |
|---|---|---|---|
| rules_only | 0.8578 (0.8522–0.8685) | 0.7646 (0.7599–0.7683) | 0.8578 (0.8522–0.8685) |
| hybrid | n/a | n/a | n/a |

## Detection (precision / recall)
| Task | Precision | Recall |
|---|---|---|
| large_charge | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| duplicate | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| category_spike | 0.3333 (0.3333–0.3333) | 1.0 (1.0–1.0) |
| subscriptions | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| unused_subscriptions | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |
| payday_trigger | 1.0 (1.0–1.0) | 1.0 (1.0–1.0) |

## Forecast (held-out months; 3 personas x 3 origins = 9 months)
| Method | Balance MAPE | Month-end MAE / monthly spend | Pinball / monthly spend | Net-flow APE | p10-p90 coverage (nominal 0.80) |
|---|---|---|---|---|---|
| model | 0.0432 (0.0322–0.0556) | 0.1339 (0.0964–0.1795) | 0.0247 (0.0225–0.027) | 1.0089 (0.6762–1.304) | 0.805 (0.7627–0.8477) |
| flat_balance | 0.2325 (0.2069–0.2802) | 0.1619 (0.0999–0.2777) | 0.2404 (0.2009–0.2963) | 1.0 (1.0–1.0) | - |
| seasonal_naive | 0.044 (0.0241–0.0569) | 0.1024 (0.0825–0.1318) | 0.0381 (0.0307–0.0513) | 0.9657 (0.3661–1.7367) | - |
| trailing-30d-mean (spikes on salary) | 0.4239 (0.3198–0.5464) | 0.8297 (0.7413–0.8841) | 0.4076 (0.3644–0.4528) | 6.1373 (2.8037–7.8293) | - |

Coverage is over n=9 held-out months (daily points within each). A flat-balance forecast scores exactly 1.0 on net-flow APE, so it is reported only for completeness.

## Calibration
Event: balance falls >10% below its starting level within 30 days. n=102 (overlapping origins), mean predicted 0.6833, observed 0.7255, Brier 0.096, constant-base-rate Brier 0.1992, Brier skill score 0.5182.

| Predicted bin | Count | Mean predicted | Observed |
|---|---|---|---|
| 0.0-0.2 | 18 | 0.0457 | 0.2222 |
| 0.2-0.4 | 10 | 0.2782 | 0.2 |
| 0.4-0.6 | 9 | 0.5078 | 0.5556 |
| 0.6-0.8 | 7 | 0.7174 | 1.0 |
| 0.8-1.0 | 58 | 0.9741 | 0.9655 |

## Ablation: LLM-only vs rules-only vs hybrid
n/a (--no-llm)

## Advice backtest (last 90 days; assumes every recommendation was followed)
Headline is cancel savings (deterministic). Excess is an estimated excess over median (noisy upper bound), shown separately.

| Persona | Cancel savings (headline) | % of spend | Estimated excess over median (noisy upper bound) | Cancelled (correct/total) | Unused in truth | Recall |
|---|---|---|---|---|---|---|
| priya | ₹5,397 | 3.1% | ₹3,528 (2.1%) | 2/2 | 2 | 2/2 |
| arjun | ₹5,397 | 3.5% | ₹13,754 (9.1%) | 2/2 | 2 | 2/2 |
| meera | ₹897 | 0.6% | ₹2,976 (1.9%) | 1/1 | 1 | 1/1 |

## Real-data check
not run: no dataset supplied (see eval/real_data.py)

## Grounding
n/a (--no-llm)

## Methodology and limitations
- Detection, forecast and calibration engines are scored on oracle ground-truth categories to isolate them from the categorizer, so end-to-end performance is bounded by the categorization accuracy.
- All data is synthetic, produced by one generator that also planted the detection targets.
- The personas share that generator, so results are not independent samples; min-max shows spread across only three personas.
- category_spike precision is capped at 1/3 by construction (max_findings=3, one planted spike per persona); it is reported as precision@3 plus recall.
- Macro-F1 averages over labels present in the ground truth only.
- precision_recall returns 1.0 for an empty prediction set by convention.
- Calibration origins overlap (7-day stride, 30-day windows) and sit mid-month rather than month-aligned, so the effective n is far smaller than n.
- A flat-balance forecast scores exactly 1.0 on net-flow APE, so that metric is reported only for completeness.
- Pinball loss and month-end MAE are divided by the persona's mean monthly spend; baselines are point forecasts used as all three quantiles.
- Advice backtest: the headline is cancel savings, the deterministic post-cutoff spend at merchants the system recommended cancelling. It assumes the user cancelled every flagged subscription on the cutoff day.
- Backtest excess is a separate, noisy upper bound: for each flagged category spike, post-cutoff monthly spend above the pre-cutoff median baseline (prorated by days present for partial months, net of cancelled merchants in that category to avoid double counting). It nets nothing off for months below baseline, so it only ever adds, and it is not a forecast of behaviour.
- Each persona has only 1-2 cancellations, so cancel precision and recall (reported as k/n counts) are anecdotal.
- The LLM-only baselines receive a compressed monthly merchant table, not raw transactions; this is a design choice.
- The LLM-only advice is audited with no engine evidence, so any derived number in its text that is not in its claims counts as ungrounded; this inflates the gap relative to the hybrid, whose recommendations carry engine evidence.
- The LLM-only forecast uses the same 3 monthly origins per persona as the model (monthly net-flow history before the origin -> that month's net flow). A net-flow APE of 1.0 equals the flat-balance forecast.
- The ablation is a single run at temperature 0 over 3 synthetic personas (only when run with an LLM); differences are indicative, not statistically tested.