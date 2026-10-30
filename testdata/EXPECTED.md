# FinPilot test data: what to upload and what you should see

All files are synthetic and deterministic (`python testdata/generate_testdata.py`). Format: `date,description,amount`
(negative = money out). Data ends **2026-10-19**, so the 60-day weather map starts 2026-10-20.

Upload in the UI header: choose the file, type the **profile name** and **opening balance** from the table, click *Upload CSV*.
Two scripts check everything for you:

```bash
.venv/bin/python testdata/verify_testdata.py      # all files through the real API, no key needed, free
.venv/bin/python testdata/score_categories.py     # categorizer accuracy with the real model (needs MESH_API_KEY)
```

"Observed" = what this build printed with **no LLM key**. "With key" = what should additionally happen once the model
is connected (not yet observed by me: that is what you are testing).

| File | Profile | Opening ₹ | Purpose |
|---|---|---|---|
| `good_01_everything.csv` | ananya | 60000 | every detector has something to find |
| `good_02_tight_budget_storms.csv` | rohit | 8000 | storms in the *Normal* scenario |
| `good_03_freelancer_irregular_income.csv` | kabir | 80000 | irregular income, no salary |
| `good_04_subscription_overlap.csv` | neha | 40000 | overlapping / unused subscriptions |
| `good_05_llm_categorization_stress.csv` (+ `good_05_truth_labels.csv`) | llmtest | 20000 | unknown merchants for the LLM |
| `good_06_mixed_channels_known_limitation.csv` | mixed | 30000 | a known weakness, expected to be missed |
| `bad_*.csv` | badtest | 0 | must be rejected (or loaded safely) |

## good_01 (ananya): the "everything" file
Planted: December food-delivery spike (+14 orders), ₹48,000 CROMA charge (March), two duplicate charges
(AMAZON PAY INDIA ₹3,499 on Jun 20-21; SWIGGY ₹612 on Aug 9-10), an unused gym (CULT FIT GYM ₹1,800) and an overlapping
streaming service (HOTSTAR next to NETFLIX), a refund (Jul), a payday delivery habit, and a month-to-date October delivery
overspend (14 extra orders, which should trigger the pace alert).
- **Observed:** 656 rows, balance ₹232,311; findings: duplicate_charge ×2, large_charge ×1, category_spike ×3
  (shopping Jun, food delivery Dec, food delivery Oct), unused_subscription ×2 (CULT FIT GYM no usage, HOTSTAR overlap),
  subscription ×4, behavior_trigger ×1; 9 recommendations; the digest's *One action* is "Cancel CULT FIT GYM".
- **Weather (Observed):** Normal = clear (0%). Big purchase (default ₹93k): 13 storm days, worst Oct 29 at 100%.
  Job-loss drill: clear (the ₹232k balance absorbs two missed paydays). Emergency default: clear.
- **With key:** SIP / THIRD WAVE COFFEE / KOSHYS BAKERY / NYKAA / PVR / REDBUS / URBAN COMPANY etc. get categories from the
  model (without a key they are "uncategorized"). ZERODHA COIN SIP should come out as `savings`.
- **Try:** Dashboard → Recommendations → open *Why?* on "Cancel CULT FIT GYM"; Can I afford ₹80,000?; Decision duel.

## good_02 (rohit, opening ₹8,000): storms without any stress test
Salary ₹32,000 barely covers rent, EMI, bills and a credit-card bill, so the balance hovers near the safety buffer.
- **Observed:** balance ₹16,415. Normal: **59% of futures breach the buffer**, worst day Nov 30 at 48%, 9 storm days;
  a `storm_warning` recommendation ("Storm warning: 2026-11-25 to 2026-11-30, peak 48% ... buffer ₹7,435").
  Job-loss drill: 100%, 51 storm days. Big purchase: 100%, 35 storm days.
- **Try:** Money Weather → Normal → click the storm card → read the Autopsy → apply an umbrella → drag the slider down
  and watch the risk fall.

## good_03 (kabir): irregular income
Invoices arrive on random days (2-4 per month), no recurring salary, quarterly GST payment of ₹12,000.
- **Observed:** Job-loss drill says "no recurring income found, so there is no salary to skip". Normal clear; Big purchase
  default: storm; Emergency default: storm.
- **Expected false positive:** the quarterly GST ₹12,000 payments are flagged as `large_charge` (they look unusual because
  they only happen every 3 months). This is a real limitation worth knowing about.

## good_04 (neha): subscriptions
Seven monthly subscriptions: NETFLIX, HOTSTAR, PRIME VIDEO (video), SPOTIFY, GAANA PLUS (music), CULT FIT GYM, GOLDS GYM
(fitness, no gym-related spend), plus a one-off ₹19,900 ADOBE charge.
- **Observed (no key):** 5 subscriptions detected, **3 flagged**: CULT FIT GYM (no usage), HOTSTAR and PRIME VIDEO (overlap
  with NETFLIX). GAANA PLUS and GOLDS GYM are not in the keyword rules, so without the model they are "uncategorized" and
  not seen as subscriptions. ADOBE flagged as a large charge.
- **With key (expected, unverified):** 7 detected; GAANA PLUS flagged (overlaps SPOTIFY) and GOLDS GYM flagged.

## good_05: the LLM categorization stress set
About 205 rows of merchants the keyword rules do not know (bakeries, clinics, airlines, SIPs, ...). Without a key every row
is "uncategorized" and nothing is found (Observed). With a key run `score_categories.py`: it prints rules-only vs
rules+LLM accuracy against `good_05_truth_labels.csv` and the first mislabelled rows. Expect rules-only near zero; the
model should lift it a lot but not to 100% (ambiguous: SIP vs savings, IRCTC vs transport/travel).

## good_06: known limitation (expected to be MISSED)
NETFLIX and SPOTIFY are paid through alternating channels (UPI handle one month, card the next), so the app sees two
different merchant names each and finds no subscription. **Observed:** 0 findings. If this ever starts detecting them,
the normalizer was improved.

## bad_* files
| File | Expected |
|---|---|
| `bad_missing_columns.csv` | HTTP 400: CSV must have columns date, description, amount |
| `bad_dates.csv` | HTTP 400: need at least 5 valid rows |
| `bad_too_few_rows.csv` | HTTP 400: need at least 5 valid rows |
| `bad_binary.csv` | HTTP 400: could not read CSV |
| `bad_injection.csv` | **loads (HTTP 200)**. Descriptions contain `<script>`, "IGNORE ALL PREVIOUS INSTRUCTIONS", SQL, `<img onerror>`. They must appear as plain text or be normalised, never run |
| `bad_extreme_values.csv` | loads (HTTP 200): a NaN row is dropped; ₹1e12 and ₹1e9 amounts must not crash anything |
| `bad_too_many_rows.csv`, `bad_over_5mb.csv` | only with `--big`: HTTP 400 (>50,000 rows) and 413 (>5 MB) |

## Things to ask the Chat tab (no key works; with a key the wording is nicer)
- "How much did I spend on food delivery in 2025-12?"  (ananya)
- "What subscriptions do I pay for?"
- "Show my top merchants"
- "Can I afford 50000?"
- "How am I doing on my goal?" (add a goal first: Goals tab → *Europe trip*, target 400000, deadline 2027-06-01)

## Corrections (category override)
```bash
curl -X POST localhost:8000/api/correct/ananya -H 'Content-Type: application/json' \
  -d '{"merchant":"ZERODHA COIN SIP","category":"savings"}'
```
It should return a run id and re-run the pipeline with that merchant fixed.

## Known data-driven caveats (honest)
- Precision is not perfect: on good_01 one category spike (shopping in June) is partly caused by the planted duplicate
  and the big Amazon purchases, not a separate habit.
- Quarterly or annual payments look like outliers to the large-charge detector (good_03 GST, good_04 ADOBE).
- UPI handles like `@okaxis` are not stripped from merchant names: merchants stay consistent only if their payment
  channel stays consistent (good_06 shows what happens when it does not).
