"""python -m ml.tune_hybrid [--max-cases N]

Tunes the hybrid weights  Score = w1*CBF + w2*CF + w3*Popularity + w4*Budget  on VALIDATION data and reports on separate TEST data.
Writes artifacts/hybrid_weights.json (read by app.py) and artifacts/hybrid_report.json (every number below, with its data label).

Data used (read the labels before quoting a number):
  V  interaction validation  - each user's second-latest trip is hidden; CF is fitted WITHOUT it.          (SYNTHETIC unless data/user_survey.csv exists)
  T  interaction test        - each user's latest trip is hidden; CF is fitted without it. Never used for tuning.  (same caveat)
  S  25 scenarios            - data/recommendation_test_set_25.csv, one person's expected top-5 per scenario.     (small, hand-labelled)
Only trips to a district the user had NOT visited before count (a recommender should suggest somewhere new); already-visited districts are
removed from the ranking. Weights are chosen on V + S with the one-standard-error rule (nearest the prior among near-ties), then:
  * scored on T (never seen while tuning), and
  * scored on S by repeated 5-fold CV where each fold's weights are tuned on V + the other scenarios only.
"""
import os, sys, json, time, argparse, datetime, warnings; warnings.filterwarnings('ignore'); os.environ.setdefault('TRAVELBD_NO_BROWSER', '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A
from ml import hybrid as H
from sklearn.model_selection import KFold

ap = argparse.ArgumentParser(); ap.add_argument('--max-cases', type=int, default=1200); ap.add_argument('--seed', type=int, default=0); args = ap.parse_args()
rng = np.random.default_rng(args.seed)
DISTRICTS = sorted(A.PLACES['district'].astype(str).unique()); DI = {d: i for i, d in enumerate(DISTRICTS)}; N = len(DISTRICTS)
Z2D = dict(zip(A.DEST['destination_id'].astype(str), A.DEST['district']))
SEAS = {'winter': 'Winter', 'monsoon': 'Monsoon', 'summer': 'Summer', 'any': 'Any season'}

def case_arrays(rows):
    """recommend_destinations rows -> (components N x 4, trip_fit N, cost_low N, type sets) in DISTRICTS order."""
    rows = sorted(rows, key=lambda r: DI[str(r['destination']['district'])])
    C = np.array([[r['components'][k] for k in H.COMPONENTS] for r in rows])
    return C, np.array([r['trip_fit'] for r in rows]), np.array([r['trip_cost_low'] or 0 for r in rows]), [set(r['destination'].get('type_list') or []) for r in rows]

class Set:
    """A block of evaluation cases."""
    def __init__(s, name, k): s.name, s.k, s.C, s.TF, s.REL, s.EX, s.LO, s.BUD = name, k, [], [], [], [], [], []
    def add(s, rows, rel, excl, budget):
        C, tf, lo, ty = case_arrays(rows); s.C.append(C); s.TF.append(tf); s.REL.append(rel); s.EX.append(excl); s.LO.append(lo); s.BUD.append(budget); s.types = ty
    def freeze(s):
        for a in ('C', 'TF', 'REL', 'EX', 'LO', 'BUD'): setattr(s, a, np.array(getattr(s, a)))
        return s
    def source(s, idx=None):
        i = np.arange(len(s.C)) if idx is None else idx
        return dict(C=s.C[i], rel=s.REL[i], k=s.k, exclude=s.EX[i], trip_fit=s.TF[i])
    def metrics(s, w, idx=None):
        """w: weight dict, 'random', or callable. -> dict of mean metrics (+ 95% CI half-width for NDCG)."""
        i = np.arange(len(s.C)) if idx is None else idx
        if isinstance(w, str) and w == 'random': sc = rng.random(s.C[i].shape[:2])
        else: sc = (s.C[i] @ np.array([w[c] for c in H.COMPONENTS])) * s.TF[i]
        nd, hit, rr = H.ndcg_at_k(sc, s.REL[i], s.k, s.EX[i])
        sc2 = sc.copy(); sc2[s.EX[i]] = -1e9; top = np.argsort(-sc2, axis=1, kind='stable')[:, :s.k]
        within = (np.take_along_axis(s.LO[i], top, 1) <= s.BUD[i][:, None]).mean()
        rec = (np.take_along_axis(s.REL[i], top, 1).sum(1) / np.maximum(1, s.REL[i].sum(1))).mean()
        return dict(cases=int(len(i)), **{f'hit@{s.k}': round(float(hit.mean()), 3), f'recall@{s.k}': round(float(rec), 3), f'ndcg@{s.k}': round(float(nd.mean()), 3), 'mrr': round(float(rr.mean()), 3),
                    f'top{s.k}_within_budget': round(float(within), 3), 'coverage': round(len(set(top.ravel().tolist())) / N, 3), 'ndcg_ci95': round(float(1.96 * nd.std(ddof=1) / np.sqrt(len(nd))), 3)})

# ------------------------------------------------------------------ interaction cases (V, T)
inter, info = H.load_interactions(A.DATA, Z2D)
inter['rk'] = inter.groupby('user').cumcount(ascending=False); inter['n'] = inter.groupby('user')['user'].transform('size')

def interaction_set(name, held_rk, min_trips, cf_min_rk):
    train = inter[inter.rk >= cf_min_rk]; cf = H.CollaborativeFilter().fit(train, DISTRICTS)
    hist = train.groupby('user')['district'].apply(set).to_dict(); cand = inter[(inter.rk == held_rk) & (inter.n >= min_trips)]
    cand = cand[[r.district not in hist.get(r.user, set()) for r in cand.itertuples()]]          # new-destination trips only
    if len(cand) > args.max_cases: cand = cand.sample(args.max_cases, random_state=args.seed)
    S = Set(name, 10); t0 = time.time()
    for n, r in enumerate(cand.itertuples()):
        nights = int(r.nights) if pd.notna(r.nights) else 2; days = nights + 1; group = int(r.group) if pd.notna(r.group) and r.group >= 1 else 2
        bpd = float(r.budget_pppd) if pd.notna(r.budget_pppd) else 3000.0; total = bpd * days * group; h = hist.get(r.user, set())
        rows = A.recommend_destinations(total, days, group, getattr(r, 'origin', 'Anywhere') or 'Anywhere', str(r.style), '', '', str(getattr(r, 'facilities', '') or '').replace(';', ','),
                                        getattr(r, 'season', 'Any season') or 'Any season', 'Any', limit=N, history=sorted(h), weights=H.PRIOR_WEIGHTS, cf_model=cf)
        rel = np.zeros(N); rel[DI[r.district]] = 1; ex = np.zeros(N, bool)
        for d in h: ex[DI[d]] = True
        S.add(rows, rel, ex, total)
        if n and n % 300 == 0: print(f'  {name}: {n}/{len(cand)} cases, {time.time()-t0:.0f}s', flush=True)
    return S.freeze()

print('building validation cases (V)...', flush=True); V = interaction_set('V', 1, 3, 2)
print('building test cases (T)...', flush=True); T = interaction_set('T', 0, 2, 1)

# ------------------------------------------------------------------ scenario cases (S)
tests = pd.read_csv(A.DATA / 'recommendation_test_set_25.csv', encoding='utf-8-sig'); S5 = Set('S', 5)
cf_all = H.CollaborativeFilter().fit(inter, DISTRICTS)
for t in tests.itertuples():
    rows = A.recommend_destinations(float(t.total_budget_bdt), int(t.trip_days), int(t.group_size), t.origin, 'Any style', f"{t.interest}, {t.query_text}", '', '', SEAS.get(str(t.season).lower(), 'Any season'), 'Any',
                                    limit=N, weights=H.PRIOR_WEIGHTS, cf_model=cf_all)
    exp = {Z2D[i] for i in t.expected_top5_destination_ids_pipe_separated.split('|')}
    S5.add(rows, np.array([float(d in exp) for d in DISTRICTS]), np.zeros(N, bool), float(t.total_budget_bdt))
S5.freeze()
# Budget stress test: same scenarios with the budget cut to 40%. The normal scenarios cannot measure budget awareness (even Random stays within budget).
S5t = Set('S40', 5)
for t in tests.itertuples():
    b = float(t.total_budget_bdt) * .4
    rows = A.recommend_destinations(b, int(t.trip_days), int(t.group_size), t.origin, 'Any style', f"{t.interest}, {t.query_text}", '', '', SEAS.get(str(t.season).lower(), 'Any season'), 'Any', limit=N, weights=H.PRIOR_WEIGHTS, cf_model=cf_all)
    exp = {Z2D[i] for i in t.expected_top5_destination_ids_pipe_separated.split('|')}
    S5t.add(rows, np.array([float(d in exp) for d in DISTRICTS]), np.zeros(N, bool), b)
S5t.freeze()
print(f'cases: V={len(V.C)} T={len(T.C)} S={len(S5.C)}', flush=True)

# ------------------------------------------------------------------ tune
def feasibility(w):
    """share of the top-5 whose low-end trip cost fits the (tight, 40%) budget; a relevance-free check that the budget signal is doing its job."""
    return S5t.metrics(dict(zip(H.COMPONENTS, w)))['top5_within_budget']
UNIFORM = {c: .25 for c in H.COMPONENTS}          # shrinkage target: neutral, NOT the hand-set prior (otherwise tuning just returns the prior)
G = H.simplex_grid(4, .05); Vsrc = V.source(); Vsrc['stats'] = H._grid_stats(G, Vsrc)     # grid scored once on V; reused by every CV fold
chosen, best, table, se = H.tune_weights([Vsrc, S5.source()], prior=UNIFORM, tiebreak=feasibility)
chosen_V, best_V, _, _ = H.tune_weights([Vsrc], prior=UNIFORM, tiebreak=feasibility); chosen_S, best_S, _, _ = H.tune_weights([S5.source()], prior=UNIFORM, tiebreak=feasibility)
pd.DataFrame(table, columns=H.COMPONENTS + ['objective']).sort_values('objective', ascending=False).round(4).to_csv(A.ART / 'hybrid_weight_grid.csv', index=False)
shipped = H.effective_weights(chosen, info)                                                    # what the app actually uses (CF guard applied)

# repeated 5-fold CV on scenarios: weights tuned on V + training scenarios, scored on the held-out scenarios
cv = {k: [] for k in ('Hand-set prior (chosen while looking at S)', 'Tuned on V + scenarios (CV)', 'Tuned on scenarios only (CV)', 'Popularity only', 'Content (CBF) only')}; fold_w = []
for rep in range(4):
    for tr, te in KFold(5, shuffle=True, random_state=rep).split(np.arange(len(S5.C))):
        w, _, _, _ = H.tune_weights([Vsrc, S5.source(tr)], prior=UNIFORM, tiebreak=feasibility); ws, _, _, _ = H.tune_weights([S5.source(tr)], prior=UNIFORM, tiebreak=feasibility); fold_w.append([w[c] for c in H.COMPONENTS])
        cv['Tuned on V + scenarios (CV)'].append(S5.metrics(w, te)); cv['Tuned on scenarios only (CV)'].append(S5.metrics(ws, te))
        cv['Hand-set prior (chosen while looking at S)'].append(S5.metrics(H.PRIOR_WEIGHTS, te)); cv['Popularity only'].append(S5.metrics(dict(cbf=0, cf=0, popularity=1, budget=0), te)); cv['Content (CBF) only'].append(S5.metrics(dict(cbf=1, cf=0, popularity=0, budget=0), te))
cv_tbl = [dict(model=k, **{c: round(float(np.mean([m[c] for m in v])), 3) for c in v[0] if c != 'cases'}) for k, v in cv.items()]; fold_w = np.array(fold_w)

# ------------------------------------------------------------------ report
def row(name, w, *sets): return dict(model=name, weights=({c: round(w[c], 3) for c in H.COMPONENTS} if isinstance(w, dict) else None), **{f'{s.name}:{k}': v for s in sets for k, v in s.metrics(w).items() if k in (f'hit@{s.k}', f'ndcg@{s.k}', f'top{s.k}_within_budget', 'ndcg_ci95', 'coverage')})
one = lambda c: {k: float(k == c) for k in H.COMPONENTS}
abl = lambda w, c: H.normalise_weights({k: (0 if k == c else v) for k, v in w.items()})
models = [('Random', 'random'), ('Popularity only', one('popularity')), ('Content (CBF) only', one('cbf')), ('CF only', one('cf')), ('Budget only', one('budget')), ('Equal weights', {c: .25 for c in H.COMPONENTS}),
          ('Hand-set prior (chosen while looking at S)', H.PRIOR_WEIGHTS), ('Tuned weights (V+S, 1-SE)', chosen), ('SHIPPED = tuned + CF evidence guard', shipped)] + [(f'Shipped without {c}', abl(shipped, c)) for c in H.COMPONENTS]
table_rows = [row(n, w, T, S5, S5t) for n, w in models]
OUT = dict(generated=datetime.date.today().isoformat(), interaction_data=info, n_cases=dict(V_validation=len(V.C), T_test=len(T.C), S_scenarios=len(S5.C)),
           weights=dict(tuned=chosen, best_on_grid=best, tuned_on_V_only=chosen_V, tuned_on_S_only=chosen_S, shipped_with_cf_guard=shipped), results_test_and_scenarios=table_rows, scenario_cv=cv_tbl,
           scenario_cv_fold_weights=dict(mean={c: round(float(x), 3) for c, x in zip(H.COMPONENTS, fold_w.mean(0))}, std={c: round(float(x), 3) for c, x in zip(H.COMPONENTS, fold_w.std(0))}),
           notes=['T hits/NDCG are for an interaction set that is SYNTHETIC unless interaction_data.real is true; do not present them as evidence about real travellers.',
                  'S is 25 hand-written scenarios; the prior weights were set while looking at it, so the prior is optimistic there. The CV rows tune without the held-out scenarios.',
                  'Weights were tuned on V + S. T was never used for tuning or selection. S numbers for the final weights are in-sample; use the CV table for an honest estimate.'])
json.dump(OUT, open(A.ART / 'hybrid_report.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
json.dump(dict(weights=chosen, tuned=True, source='ml/tune_hybrid.py: grid search (step .05) on V + S; among weights within one SE of the best, the one that keeps top-5 within a tight budget, then nearest equal weights', tuned_on=date if (date := datetime.date.today().isoformat()) else '',
               cf_data=info['source'], cf_data_is_real=info['real'], n_validation_cases=len(V.C), n_scenarios=len(S5.C),
               note='CF weight is further limited at run time while CF data is synthetic (ml/hybrid.py: SYNTHETIC_CF_CAP).'), open(A.ART / 'hybrid_weights.json', 'w', encoding='utf-8'), indent=1)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('\nDATA:', info['source'], '| users', info['n_users'], '\nweights  tuned:', chosen, '\n         best-on-grid:', best, '\n         V-only:', chosen_V, ' S-only:', chosen_S, '\n         shipped (CF guard):', {k: round(v, 3) for k, v in shipped.items()})
print('\n(S40 = scenarios with budget cut to 40%: look at S40:top5_within_budget)')
print('\n== T (interaction TEST, never tuned on) + S (scenarios, in-sample for tuned rows)')
print(pd.DataFrame(table_rows).drop(columns=['weights']).to_string(index=False))
print('\n== S by repeated 5-fold CV (honest estimate)'); print(pd.DataFrame(cv_tbl).to_string(index=False))
print('\nfold weights mean', OUT['scenario_cv_fold_weights']['mean'], 'std', OUT['scenario_cv_fold_weights']['std'])
