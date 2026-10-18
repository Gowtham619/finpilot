# Evaluation results

Personas: priya, arjun, meera (synthetic, known ground truth)

## Categorization
| Method | Accuracy | Macro-F1 | Coverage |
|---|---|---|---|
| rules_only | 0.8578 | 0.7646 | 0.8578 |
| hybrid | n/a | n/a | n/a |

## Detection (precision / recall)
| Task | Precision | Recall |
|---|---|---|
| large_charge | 1.0 | 1.0 |
| duplicate | 1.0 | 1.0 |
| category_spike | 0.3333 | 1.0 |
| subscriptions | 1.0 | 1.0 |
| unused_subscriptions | 1.0 | 1.0 |
| payday_trigger | 1.0 | 1.0 |

## Forecast (held-out months)
| Method | Net-flow APE | Balance MAPE | p10-p90 coverage |
|---|---|---|---|
| model | 1.0089 | 0.0432 | 0.805 |
| naive | 6.1373 | 0.4239 | - |

## Calibration
n=102, mean predicted 0.6833, observed 0.7255, Brier 0.096

## Grounding
n/a (--no-llm)