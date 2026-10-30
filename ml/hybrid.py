"""Hybrid recommendation model for TravelBD.

    Score(u, i) = w1*CBF(u,i) + w2*CF(u,i) + w3*Popularity(i) + w4*Budget(u,i)        (then x trip-length fit)

Every component is scaled to 0..1 so the weights are comparable. The weights live in artifacts/hybrid_weights.json and are
produced by `python -m ml.tune_hybrid` (grid search on a VALIDATION split, then scored once on a separate TEST split).

CBF         content similarity between the user's stated preferences and the destination (TF-IDF text + destination-type
            match + season fit). Computed in app.py because it needs the live TF-IDF model.
CF          what similar users chose. Cold-start safe: a new visitor has no history, so neighbours are found from the
            profile they type in (travel style, budget per person per day, group size); a returning user's own past
            destinations add an item-item signal. Needs interaction data (see load_interactions).
Popularity  general acceptance: visitor ratings of the destination's attractions (shrunk toward the national mean so 3
            ratings cannot beat 3,000), review volume, and the location-level BanTABSA review score.
Budget      1.0 while the estimated trip cost is within budget, falling to 0 at 25% over.

Nothing here imports app.py, so app.py can import this module.
"""
import json, math, re
from pathlib import Path
import numpy as np, pandas as pd

COMPONENTS = ['cbf', 'cf', 'popularity', 'budget']
# Starting point (a prior, NOT claimed optimal). Used when no tuned file exists and as the shrinkage target while tuning.
PRIOR_WEIGHTS = {'cbf': .50, 'cf': .10, 'popularity': .15, 'budget': .25}
# CF learned from SYNTHETIC interactions is a pipeline check, not evidence of real preference. Until real survey data
# exists the app caps its weight (the freed weight is shared proportionally by the other components).
SYNTHETIC_CF_CAP = .10
MIN_REAL_USERS_FOR_FULL_CF = 150


# ------------------------------------------------------------------ weights
def normalise_weights(w):
    v = np.array([max(0.0, float(w.get(k, 0))) for k in COMPONENTS])
    if v.sum() <= 0: v = np.array([PRIOR_WEIGHTS[k] for k in COMPONENTS])
    return dict(zip(COMPONENTS, (v / v.sum()).tolist()))

def load_weights(path):
    """-> (weights dict summing to 1, meta dict). Falls back to PRIOR_WEIGHTS if the file is missing or malformed."""
    try:
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        return normalise_weights(d['weights']), {k: v for k, v in d.items() if k != 'weights'}
    except Exception:
        return normalise_weights(PRIOR_WEIGHTS), {'source': 'prior (no tuned weights file found)', 'tuned': False}

def effective_weights(weights, cf_info):
    """Apply the evidence guard to the CF weight. The cap applies to the FINAL (normalised) CF share; the other three keep their relative proportions.
    real interaction data  -> full weight once >= MIN_REAL_USERS_FOR_FULL_CF users, scaled down below that
    synthetic only         -> at most SYNTHETIC_CF_CAP;  no interaction data at all -> 0"""
    w = normalise_weights(weights); cf = w['cf']
    if not cf_info or not cf_info.get('n_users'): target = 0.0
    elif cf_info.get('real'): target = cf * min(1.0, cf_info['n_users'] / MIN_REAL_USERS_FOR_FULL_CF)
    else: target = min(cf, SYNTHETIC_CF_CAP)
    rest = {k: v for k, v in w.items() if k != 'cf'}; rs = sum(rest.values())
    out = {k: (v / rs * (1 - target) if rs > 0 else (1 - target) / 3) for k, v in rest.items()}; out['cf'] = target
    return {k: float(out[k]) for k in COMPONENTS}

def hybrid_score(components, weights):
    """components: dict name -> array (or float); weights: dict name -> float summing to 1."""
    return sum(weights[k] * components[k] for k in COMPONENTS)


# ------------------------------------------------------------------ interaction data
def _budget_midpoint(v):
    """'2000-3000' / '2,000 to 3,000' / 2500 -> 2500.0 (NaN if unparseable)."""
    nums = [float(x.replace(',', '')) for x in re.findall(r'\d[\d,]*\.?\d*', str(v))]
    return float(np.mean(nums[:2])) if nums else float('nan')

def _truthy(v): return str(v).strip().lower() in ('1', 'true', 'yes', 'y')

def load_interactions(data_dir, zone_to_district, min_rating=4):
    """Return (DataFrame[user, district, date, style, budget_pppd, group, nights], info).
    Prefers REAL survey responses in data/user_survey.csv (either the project's own template or the starter-kit template).
    Only visited districts rated >= min_rating (default 4) count as 'liked'. Falls back to the synthetic reviews (flagged)."""
    data_dir = Path(data_dir); f = data_dir / 'user_survey.csv'
    if f.exists():
        s = pd.read_csv(f, encoding='utf-8-sig')
        if 'district_rated' in s.columns:        # starter-kit schema
            s = s[s.get('consent_to_use_answers_for_student_project', 'yes').map(_truthy)] if 'consent_to_use_answers_for_student_project' in s else s
            if 'actually_visited_district_yes_no' in s: s = s[s['actually_visited_district_yes_no'].map(_truthy)]
            s = pd.DataFrame({'user': s['participant_code_anonymous'], 'district': s['district_rated'], 'rating': pd.to_numeric(s['district_rating_1_to_5'], errors='coerce'),
                              'date': s.get('visit_year_optional', s.get('survey_date_yyyy_mm_dd')), 'style': s.get('travel_style', ''),
                              'budget_pppd': s['budget_range_bdt_per_person_per_day'].map(_budget_midpoint), 'group': pd.to_numeric(s.get('usual_group_size'), errors='coerce'),
                              'nights': pd.to_numeric(s.get('usual_trip_length_days'), errors='coerce') - 1})
        else:                                     # data/user_survey_template.csv schema
            s = pd.DataFrame({'user': s['user_id'], 'district': s['district'], 'rating': pd.to_numeric(s['rating'], errors='coerce'), 'date': s.get('visit_date'),
                              'style': s.get('travel_style', ''), 'budget_pppd': s['budget_range_bdt_per_person_per_day'].map(_budget_midpoint),
                              'group': pd.to_numeric(s.get('group_size'), errors='coerce'), 'nights': pd.to_numeric(s.get('trip_days'), errors='coerce') - 1})
        s = s[s['rating'] >= min_rating].dropna(subset=['user', 'district'])
        s['date'] = pd.to_datetime(s['date'].astype(str), errors='coerce')
        info = dict(real=True, source='data/user_survey.csv (real survey)')
    else:
        r = pd.read_csv(data_dir / 'reviews_cleaned.csv')
        s = pd.DataFrame({'user': r['user_id'], 'district': r['selected_destination_id'].astype(str).map(zone_to_district), 'date': pd.to_datetime(r['review_date'], errors='coerce'),
                          'style': r['travel_style'], 'budget_pppd': r['user_budget_bdt_per_person_per_day'], 'group': r['group_size'], 'nights': r['nights'],
                          'origin': r['home_origin'], 'season': r['travel_season'], 'facilities': r['preferred_facilities']})
        s = s.dropna(subset=['district'])
        info = dict(real=False, source='reviews_cleaned.csv (SYNTHETIC placeholder trips; ratings column is empty)')
    s = s.sort_values('date', kind='stable').reset_index(drop=True)
    info.update(n_rows=int(len(s)), n_users=int(s['user'].nunique()), n_districts=int(s['district'].nunique()))
    return s, info


# ------------------------------------------------------------------ collaborative filtering
class CollaborativeFilter:
    """Neighbourhood CF that works for brand-new users.

    Profile neighbours (always available):  every training user v gets a similarity to the visitor from travel style
        (same = 1, different = `style_other`), budget per person per day (log-ratio) and group size (log-ratio). A district's score
        is the similarity-weighted share of neighbours who went there, smoothed toward the overall share by `shrink` pseudo-users
        (so 2 lucky neighbours cannot dominate) and divided by overall_share**pop_power so it measures 'chosen MORE than usual by people like
        you' instead of just repeating popularity.
    Item neighbours (only with history):  cosine similarity between districts' visitor sets; scores districts similar to those the user liked.
    """
    def __init__(self, shrink=8.0, pop_power=.5, style_other=.25, budget_scale=.7, group_scale=1.0, k_items=20, history_share=.5):
        self.shrink, self.pop_power, self.style_other, self.budget_scale, self.group_scale, self.k_items, self.history_share = shrink, pop_power, style_other, budget_scale, group_scale, k_items, history_share

    def fit(self, inter, districts):
        self.districts = list(districts); idx = {d: i for i, d in enumerate(self.districts)}
        inter = inter[inter['district'].isin(idx)]
        if inter.empty:                                  # no interaction data: CF scores are all zero and the CF weight is switched off
            self.M = np.zeros((0, len(self.districts))); self.n_users = 0; self.g = np.ones(len(self.districts)) / len(self.districts); return self
        users = inter['user'].drop_duplicates().tolist(); uidx = {u: i for i, u in enumerate(users)}
        M = np.zeros((len(users), len(self.districts)))
        M[inter['user'].map(uidx).astype(int).values, inter['district'].map(idx).astype(int).values] = 1.0
        prof = inter.groupby('user').agg(style=('style', 'first'), budget=('budget_pppd', 'median'), group=('group', 'median')).reindex(users)
        self.M = M; self.n_users = len(users)
        self.style = prof['style'].astype(str).str.lower().values
        self.lb = np.log(pd.to_numeric(prof['budget'], errors='coerce').clip(lower=100).fillna(prof['budget'].median() if prof['budget'].notna().any() else 2000).values)
        self.lg = np.log(pd.to_numeric(prof['group'], errors='coerce').clip(lower=1).fillna(2).values)
        self.g = (M.sum(0) + 1) / (M.sum() + len(self.districts))               # overall share per district (add-one so no zeros)
        if self.n_users:
            Mi = M; norm = np.sqrt((Mi ** 2).sum(0)); norm[norm == 0] = 1
            S = (Mi.T @ Mi) / np.outer(norm, norm); np.fill_diagonal(S, 0)
            for i in range(S.shape[0]):
                row = S[i]
                if self.k_items < len(row): row[np.argsort(row)[:-self.k_items]] = 0
            self.S = S
        return self

    def _profile_scores(self, style, budget_pppd, group):
        sim = np.ones(self.n_users)
        st = str(style or '').strip().lower()
        if st and st not in ('any style', 'any', 'anywhere'): sim = sim * np.where(self.style == st, 1.0, self.style_other)
        if budget_pppd and budget_pppd > 0: sim = sim * np.exp(-np.abs(self.lb - math.log(max(100.0, budget_pppd))) / self.budget_scale)
        if group and group > 0: sim = sim * np.exp(-np.abs(self.lg - math.log(max(1.0, group))) / self.group_scale)
        share = (sim @ self.M + self.shrink * self.g) / (sim.sum() + self.shrink)
        return share / np.power(self.g, self.pop_power)

    def score(self, style=None, budget_pppd=None, group=None, history=()):
        """0..1 score per district (order = self.districts). All zeros when there is no interaction data."""
        if not self.n_users: return np.zeros(len(self.districts))
        s = self._profile_scores(style, budget_pppd, group)
        s = s / s.max() if s.max() > 0 else s
        idx = {d: i for i, d in enumerate(self.districts)}; h = [idx[d] for d in history if d in idx]
        if h:
            it = self.S[h].sum(0); it = it / it.max() if it.max() > 0 else it
            s = (1 - self.history_share) * s + self.history_share * it
        return np.clip(s, 0, 1)


def build_cf(data_dir, zone_to_district, districts):
    """Fit the CF model on everything available. -> (CollaborativeFilter, info dict)."""
    inter, info = load_interactions(data_dir, zone_to_district)
    return CollaborativeFilter().fit(inter, districts), info


# ------------------------------------------------------------------ weight tuning
def simplex_grid(n=4, step=.05):
    """All weight vectors with entries in multiples of `step` that sum to 1 (n=4, step=.05 -> 1,771 vectors)."""
    k = int(round(1 / step)); out = []
    def rec(prefix, left, slots):
        if slots == 1: out.append(prefix + [left]); return
        for i in range(left + 1): rec(prefix + [i], left - i, slots - 1)
    rec([], k, n); return np.array(out, float) / k

def ndcg_at_k(scores, relevant, k, exclude=None):
    """scores (cases x items); relevant (cases x items, 0/1); exclude (cases x items bool, optional) is removed from the ranking.
    Returns (ndcg per case, hit per case, reciprocal rank per case)."""
    s = scores.astype(float).copy()
    if exclude is not None: s[exclude] = -1e9
    top = np.argsort(-s, axis=1, kind='stable')[:, :k]
    rel = np.take_along_axis(relevant, top, 1); disc = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.array([disc[:int(min(k, n))].sum() for n in relevant.sum(1)])
    nd = (rel * disc).sum(1) / np.maximum(ideal, 1e-9)
    rr = np.array([1 / (np.argmax(r) + 1) if r.any() else 0.0 for r in rel])
    return nd, (rel.sum(1) > 0).astype(float), rr

def _grid_stats(G, src):
    """mean and standard error of NDCG@k for every weight vector in G on one validation source."""
    if 'stats' in src: return src['stats']          # precomputed (lets repeated CV reuse a big source)
    m = np.zeros(len(G)); se = np.zeros(len(G))
    for j, w in enumerate(G):
        sc = src['C'] @ w
        if src.get('trip_fit') is not None: sc = sc * src['trip_fit']
        nd, _, _ = ndcg_at_k(sc, src['rel'], src['k'], src.get('exclude'))
        m[j] = nd.mean(); se[j] = nd.std(ddof=1) / math.sqrt(len(nd)) if len(nd) > 1 else 0.0
    return m, se

def tune_weights(sources, prior=None, step=.05, tiebreak=None):
    """Grid-search the weight simplex on VALIDATION data only.
    sources: list of dicts {C: cases x items x 4 components, rel: cases x items 0/1, k, exclude?, trip_fit?}. With several sources the objective is the
    average of each source's NDCG@k divided by the best NDCG@k that source reaches, so a big source cannot drown out a small one.
    One-standard-error rule: all vectors within one SE of the best objective are statistically tied on relevance. Among them, `tiebreak(weights_array)`
    (optional; higher is better, e.g. how often the top-k respects a tight budget) decides, and any remaining tie goes to the vector nearest the prior.
    -> (chosen weights, best-scoring weights, grid table [w1..w4, objective], se of best)"""
    prior_v = np.array([(prior or PRIOR_WEIGHTS)[c] for c in COMPONENTS]); G = simplex_grid(len(COMPONENTS), step)
    J = np.zeros(len(G)); V = np.zeros(len(G))
    for src in sources:
        m, se = _grid_stats(G, src); top = max(m.max(), 1e-9)
        J += m / top / len(sources); V += (se / top / len(sources)) ** 2
    b = int(np.argmax(J)); se_b = math.sqrt(V[b])
    ok = np.where(J >= J[b] - se_b)[0]
    if tiebreak is not None:
        tb = np.round(np.array([tiebreak(G[i]) for i in ok]), 3); ok = ok[tb >= tb.max() - 1e-9]
    pick = ok[np.argmin(((G[ok] - prior_v) ** 2).sum(1))]
    to = lambda v: dict(zip(COMPONENTS, [round(float(x), 4) for x in v]))
    return to(G[pick]), to(G[b]), np.column_stack([G, J]), float(se_b)
