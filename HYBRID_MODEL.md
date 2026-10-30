# Hybrid recommendation model

```
Score(u,i) = w1*CBF(u,i) + w2*CF(u,i) + w3*Popularity(i) + w4*BudgetCompatibility(u,i)      then x trip-length fit (soft penalty)
```
All four components are scaled 0..1. Code: `ml/hybrid.py` (model + tuner), `app.py::recommend_destinations` (serving), `ml/tune_hybrid.py` (validation + tuning).

| Component | What it measures | How it is computed |
|---|---|---|
| CBF | how well destination attributes match the user's preferences | 5/6 interest match (TF-IDF cosine + destination-type match) + 1/6 season fit |
| CF | what similar users liked | profile neighbours (travel style, budget/person/day, group size) -> smoothed, popularity-discounted share of neighbours who went there; plus item-item similarity to the user's own past destinations when known. Works for a brand-new visitor |
| Popularity | general acceptance | 50% visitor rating of the destination's attractions (shrunk toward the national mean), 25% review volume (log), 25% BanTABSA city rating |
| Budget | does the cost fit | 1.0 while estimated trip cost <= budget, falling linearly to 0 at 25% over |

Hotels use the same four-signal form (CBF = facility match, CF = 0, Popularity = rating + review volume, Budget = nightly price vs lodging share) with fixed weights .20/0/.35/.45. These are **not tuned**: there is no labelled hotel data and no real hotel-level interactions.

## How the weights were tuned (`python -m ml.tune_hybrid`)
* **V (validation)**: each user's second-latest trip is hidden, CF is fitted without it. 1,200 cases.
* **T (test)**: each user's latest trip is hidden, CF fitted without it. 1,200 cases. **Never used for tuning or selection.**
* **S (scenarios)**: the 25 hand-written scenarios in `data/recommendation_test_set_25.csv`.
* Only trips to a district the user had not visited before count; visited districts are removed from the ranking.
* Grid search over all weight vectors in steps of 0.05 (1,771), objective = NDCG@10 on V and NDCG@5 on S, each divided by the best that source reaches.
* **One-standard-error rule**: every vector within one SE of the best is statistically tied. Among those, pick the one that keeps the top-5 within a *tight budget* (S with budgets cut to 40%), then the one nearest equal weights. This tie-break is a design decision, made because the relevance data cannot separate the tied vectors and budget awareness is otherwise unmeasured.

**Result:** CBF 0.20, CF 0.05, Popularity 0.20, Budget 0.55 (saved in `artifacts/hybrid_weights.json`; whole grid in `artifacts/hybrid_weight_grid.csv`).
Disagreement between sources is itself a finding: V alone prefers CF 0.35 / CBF 0.15; S alone prefers CF 0 / CBF 0.25.

## Results (every number carries its data label; full tables in `artifacts/hybrid_report.json`)
**T: interaction test, SYNTHETIC trips, 1,200 cases** (95% CI on NDCG about +/-0.02)

| Model | HR@10 | NDCG@10 | Top-10 within budget |
|---|---|---|---|
| Random | 0.156 | 0.074 | 0.989 |
| Popularity only | 0.472 | 0.260 | 0.965 |
| Content (CBF) only | 0.378 | 0.186 | 0.983 |
| CF only | 0.583 | 0.334 | 0.971 |
| Hand-set prior (.50/.10/.15/.25) | 0.527 | 0.263 | 0.989 |
| **Tuned hybrid (shipped)** | 0.514 | 0.269 | 0.997 |

**S: 25 scenarios, and the same scenarios with budgets cut to 40%** (CI about +/-0.12 on NDCG; tuned rows are in-sample)

| Model | hit@5 | NDCG@5 | S40 top-5 within budget |
|---|---|---|---|
| Popularity only | 0.80 | 0.312 | 0.528 |
| Content (CBF) only | 0.84 | 0.459 | 0.744 |
| Hand-set prior | 0.92 | 0.539 | 0.904 |
| **Tuned hybrid (shipped)** | 0.88 | 0.524 | 0.984 |
| Shipped without budget | 0.88 | 0.542 | 0.688 |

Honest estimate on S (repeated 5-fold CV, weights tuned without the held-out scenarios): NDCG@5 **0.515** tuned vs 0.539 hand-set vs 0.459 CBF only vs 0.312 popularity only (CI about +/-0.25).

### What this does and does not show
* The hybrid clearly beats popularity-only and content-only on the scenarios, and keeps results affordable when budgets are tight (0.98 vs 0.69 without the budget term). That part is solid.
* The tuned and hand-set weights are **statistically tied** on relevance. The tuned ones win on budget feasibility, which is why they were chosen. Do not claim tuning "improved accuracy".
* **CF cannot be validated yet.** Its data is synthetic (the `rating` column is empty and the trips were generated). On synthetic T, CF-only scores best, which says the pipeline works and that the synthetic trips contain a style/budget pattern, not that real travellers behave this way. While CF data is synthetic its final share is also capped at 10% (`SYNTHETIC_CF_CAP`).
* The scenarios cannot measure budget awareness (even Random stays within budget), hence the 40% stress test.
* S is one person's judgement of 25 cases; its SE is large, which is why the 1-SE band is wide.

## Plugging in real data (this is what makes the CF weight meaningful)
1. Collect responses with `user_survey_template.csv` from the starter kit (or the project's `data/user_survey_template.csv`) and save as `data/user_survey.csv`. Both schemas load; only consented, visited districts rated >= 4 count as "liked".
2. `python -m ml.tune_hybrid`  -> re-tunes weights, rewrites `artifacts/hybrid_weights.json`, reports on real held-out trips.
3. Restart the app. CF now gets full weight once >= 150 real users exist (scaled down below that).
Aim for 10 or more rated districts per person across 150+ people before reading anything into CF-vs-popularity.

`python -m ml.test_hybrid` runs the unit checks (weights, CF cold-start, survey loader, tuner).

## Known behaviour to be aware of
The tuned budget weight is high (0.55) and the budget score falls to zero at 25% over budget. When **no** destination matching the interests is affordable (e.g. a couple wanting a beach for 3 days on 6,000 BDT total), the ranking returns affordable places that do not match the interest (Rajshahi, Mymensingh, ...) instead of the cheapest beach. This is the intended "respect the budget" trade-off, but it can look odd in a demo. Options if you want different behaviour: lower the budget weight in `artifacts/hybrid_weights.json`, widen the 25% tolerance in `affordability()`, or show a notice when the best interest match is over budget. Validate any change with `python -m ml.tune_hybrid` (the S40 column).
