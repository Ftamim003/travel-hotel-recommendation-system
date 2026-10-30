# TravelBD model experiments (29 Oct 2026)

Run: `python -m ml.run_experiments` (writes `artifacts/ml_experiments.json`), `python -m ml.train_all` (saves the final models).
Every table says what data is behind it. Quote the "data" line together with any number.

## Why the system "does not work properly": the data, not the models
| Needed by the proposal | What exists | Consequence |
|---|---|---|
| User-item ratings for CF / SVD | The `rating` column of all 10,000 reviews is empty; every row is synthetic | CF and SVD have nothing real to learn from |
| Hotel prices to predict cost | 24 real hotels, 23 with a price | too few to train a regression |
| Labelled relevance for tuning | 25 scenarios, one person's judgement | weights cannot be tuned reliably |
| Real review text | BanTABSA: 1,242 real Bengali reviews | usable, models below |

## E1 Collaborative models (SYNTHETIC data, pipeline check only)
Leave-latest-trip-out, 2,345 users, 64 districts; SVD factors and kNN k chosen on a separate validation split.
| Model | HR@10 (95% CI +/-0.02) | NDCG@10 | Coverage |
|---|---|---|---|
| Random | 0.137 | 0.061 | 1.00 |
| Popularity | 0.466 | 0.263 | 0.30 |
| Popularity by travel style | 0.502 | 0.293 | 0.64 |
| Item-kNN CF (k=20) | 0.443 | 0.244 | 1.00 |
| SVD (validation picked 1 factor) | 0.469 | 0.264 | 0.30 |
Reading: personalised models do not beat popularity. SVD with 1 factor is popularity; more factors scored near random. The synthetic data has a weak travel-style effect and no other structure, so this is expected. These models become meaningful only on real survey data: fill `data/user_survey_template.csv`, save it as `data/user_survey.csv`, re-run.

## E2 Review sentiment (REAL data, 5-fold CV)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.743 | 0.284 |
| Naive Bayes (words) | 0.782 | 0.417 |
| Logistic Regression (words) | 0.733 | 0.519 |
| Logistic Regression (char n-grams) | 0.820 | 0.634 |
| Linear SVM (char n-grams) | 0.843 | 0.641 |
Character n-grams suit Bengali spelling variation. Reviews are 74% positive, so macro-F1 is the fair number. `data/bantabsa_aspect_profile.csv` gives per-city positive share for the 8 most-mentioned aspects.

## E3 Fare prediction (transport table = estimates; unseen destinations)
| Model | MAE (BDT) | R2 |
|---|---|---|
| Mean fare | 635 | -0.01 |
| Ridge | 205 | 0.85 |
| Random Forest | 91 | 0.88 |
| Gradient Boosting | 95 | 0.91 |
The table is itself an estimate, so this shows the model reproduces that estimate for new destinations, not real ticket prices.

## E4 Hybrid vs single models (25 scenarios, 6x repeated 5-fold CV)
| Model | recall@5 | hit@5 | NDCG@5 | MRR | diversity |
|---|---|---|---|---|---|
| Popularity only | 0.285 | 0.80 | 0.274 | 0.471 | 0.93 |
| Content TF-IDF only | 0.448 | 0.84 | 0.467 | 0.688 | 0.55 |
| Content LSA only | 0.440 | 0.80 | 0.461 | 0.673 | 0.57 |
| Hybrid, hand-set weights | 0.498 | 0.92 | 0.511 | 0.711 | 0.56 |
| Hybrid, weights tuned on validation folds | 0.465 | 0.83 | 0.493 | 0.719 | 0.66 |
- The hybrid beats every single model, which supports the hybrid design.
- Tuning did NOT help on held-out scenarios. A fit on all 25 scores 0.559 in-sample but the fold weights vary a lot (e.g. content weight 0.20 +/- 0.16): overfitting. Keep the hand-set weights until there are more labels.
- The hand-set weights were chosen while looking at this test set, so their advantage is optimistic.
- Budget weight tunes to ~0.02 because almost every top-5 already fits the test budgets: this test set cannot measure budget awareness.

## E5 Four-signal hybrid (CBF + CF + Popularity + Budget)
Superseded the E4 six-signal formula in the app. Method, tuned weights, validation/test split and honest caveats: see `HYBRID_MODEL.md`; raw numbers `artifacts/hybrid_report.json`. Re-run `python -m ml.tune_hybrid`.
