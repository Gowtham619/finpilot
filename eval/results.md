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
| model_robust_centre (candidate f, not adopted) | 0.0322 (0.0216–0.0393) | 0.08 (0.0575–0.1032) | 0.021 (0.0183–0.0234) | 0.6503 (0.3453–1.1129) | 0.9417 (0.9122–0.957) |
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

## Out-of-distribution check (second generator, engines unchanged)
Personas: kavya, rohan, isha. Same author wrote both generators: a robustness check, not independent validation. Planted per persona: 2 duplicates, 2 large charges, 1 category spike (travel), unused subscriptions [2, 1, 2], refunds [16, 19, 10]. Cells show mean (min–max) across personas.

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
| model_robust_centre | 0.2287 (0.2005–0.2821) | 0.1445 (0.0931–0.2232) | 0.051 (0.0441–0.0599) | 0.8846 (0.8484–0.9351) |
| flat_balance | 0.9679 (0.743–1.1049) | 0.0885 (0.0392–0.1443) | 0.2432 (0.2378–0.2533) | - |
| seasonal_naive | 0.5423 (0.4272–0.7199) | 0.4533 (0.2882–0.5654) | 0.164 (0.1345–0.1909) | - |
| trailing_mean | 0.8749 (0.6958–0.9688) | 0.3799 (0.2338–0.588) | 0.2283 (0.2242–0.2306) | - |

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
- model_robust_centre is the third forecast candidate (task 5c, pre-registered, not adopted): p50 from a bootstrap pool without flagged one-off charges, p10/p90 = that p50 plus the full-pool offsets (same seed). It is scored for transparency only; the shipped forecaster uses the model row. See eval/forecast_experiments.md.
- The ablation is a single run at temperature 0 over 3 synthetic personas (only when run with an LLM); differences are indicative, not statistically tested.
- Out-of-distribution check: a second generator (eval/generate_messy.py) with different merchants, noisy UPI/NEFT/IMPS/card description formats, refunds, cash, EMIs, an annual subscription, a price increase and a travel spike was scored with the engines and thresholds unchanged. The same author wrote both generators and knows the engines, so this is a robustness check, not independent validation; it is not evidence about real bank data.
- OOD behaviour: rules-only categorization accuracy fell from 0.8578 to 0.3327 (coverage 0.4201); detection and forecast were scored on oracle categories, so the categorizer drop does not feed into them. The oracle-category scoring flatters the OOD detection and forecast numbers.
- OOD refunds are positive credits whose truth category is shopping; the rules label every positive amount as income, so refunds are always miscategorized by the rules-only categorizer. Cash withdrawals and EMIs have truth category 'other', which the rules never emit (they return uncategorized).
- OOD payday_trigger has no planted target (truth set empty), so precision 0.0 (0.0–0.0) means any finding is a false positive. OOD subscription truth is the monthly subscriptions only; the annual one is not expected.
- OOD generator provenance: spend level and opening balances were set so balances stay plausible, and the share of mainstream merchants the rules know (Swiggy, Zomato, Uber, Amazon, BigBasket) was raised once after the first draft scored rules-only accuracy below the 0.3 sanity floor. Only categorization accuracy was looked at when doing so. After a first scoring run two truth-labelling bugs in the generator were fixed because they scored correct detections as misses (subscription truth now uses the engine's normalised merchant strings; a same-day duplicate pair whose 'second row' was ambiguous now sits on consecutive days). No engine, threshold or planted target was changed to move a result.