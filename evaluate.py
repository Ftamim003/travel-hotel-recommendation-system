"""Offline sanity evaluation of the destination recommender.   Run: python evaluate.py

WHAT THIS IS: a persona-based check. Each persona has a free-text query and a set of destination TYPES
that would obviously satisfy it (from the `destination_type` column). We measure, for the top-5 results:
  * precision@5  - share of results whose destination_type is in the persona's accepted set
  * in-budget@5  - share of results whose estimated trip cost fits the persona's budget
and compare the current ranking with (a) the previous scoring formula (reconstructed), (b) cheapest-first,
(c) most-reviewed-first and (d) random order (mean of 300 shuffles).

WHAT THIS IS NOT: proof the system works for real travellers. The "relevant" labels come from the same
destination table the TF-IDF model reads, so absolute precision is optimistic; the useful signal is the
comparison with the baselines. A small user study (e.g. 10-20 people rating the top-5 lists) is still needed.
"""
import json, math, os, random
os.environ.setdefault('TRAVELBD_NO_BROWSER','1')
import app as A

PERSONAS = [
  ('Beach lover',        dict(interests='beach, sea, sunset', travel_style='Nature'),                    {'Beach','Island'}),
  ('Heritage / history', dict(interests='history, heritage, museum, temple', travel_style='Cultural'),    {'Heritage','Archaeological'}),
  ('Hill trekker',       dict(interests='hiking, trekking, hill, waterfall', travel_style='Adventure'),  {'Hill','Waterfall'}),
  ('Wildlife / forest',  dict(interests='forest, wildlife, mangrove, boat'),                              {'Wildlife/Forest','Mangrove'}),
  ('Tea / wetland',      dict(interests='tea garden, haor, lake, boat ride'),                             {'Tea Estate','Haor/Wetland','Lake'}),
  ('Village life',       dict(interests='village, farm, local food, market'),                             {'Rural/Agro'}),
]
BUDGETS = [1500, 2000, 5000]   # BDT per person per day
DAYS, GROUP, K = 3, 2, 5

def run(kwargs, per_day):
    total = per_day*DAYS*GROUP
    base = dict(budget=total, days=DAYS, group_size=GROUP, origin='Anywhere', travel_style='Any style', interests='',
                activities='', facilities='', season='Any season', accessibility='Any', limit=len(A.PLACES))
    base.update(kwargs)
    return A.recommend_destinations(**base), total

def legacy_order(rows, total):
    """Previous formula: budget score = 1 - cost/budget (cheaper always better), weights .43/.17/.09/.20/.04/.02/.05."""
    def old(r):
        cost = r['estimated_trip_cost']
        bs = max(0, min(1, 1-(cost/max(total,1)))) if cost is not None else .5
        return .43*r['content']+.17*r['rating_score']+.09*r['popularity']+.20*bs+.04*.5+.05*r['bantabsa_score']
    return sorted(rows, key=old, reverse=True)

def metrics(order, accepted, total):
    top = order[:K]
    p = sum(bool(set(r['destination']['type_list']) & accepted) for r in top)/K   # a destination may offer several types
    b = sum((r['estimated_trip_cost'] or 0) <= total for r in top)/K
    return p, b

def main():
    systems = ['current','previous formula','cheapest first','most reviewed','random']
    acc = {s:{'p':[], 'b':[]} for s in systems}
    rng = random.Random(7)
    for name,kw,accepted in PERSONAS:
        for per_day in BUDGETS:
            rows,total = run(kw, per_day)
            orders = {'current':rows, 'previous formula':legacy_order(rows,total),
                      'cheapest first':sorted(rows,key=lambda r:r['destination']['estimated_daily_budget_bdt']),
                      'most reviewed':sorted(rows,key=lambda r:-r['review_count'])}
            for s,o in orders.items():
                p,b = metrics(o,accepted,total); acc[s]['p'].append(p); acc[s]['b'].append(b)
            ps=bs=0
            for _ in range(300):
                o = rows[:]; rng.shuffle(o); p,b = metrics(o,accepted,total); ps+=p; bs+=b
            acc['random']['p'].append(ps/300); acc['random']['b'].append(bs/300)
    n = len(PERSONAS)*len(BUDGETS)
    report = {'k':K,'personas':len(PERSONAS),'budgets_bdt_per_person_day':BUDGETS,'cases':n,
              'results':{s:{'precision_at_5':round(sum(v['p'])/n,3),'in_budget_at_5':round(sum(v['b'])/n,3)} for s,v in acc.items()},
              'caveat':'Proxy labels come from destination_type, which the TF-IDF text also contains, so absolute precision is optimistic. Compare systems against each other; confirm with a user study.'}
    print(f"{'system':18s} precision@{K}  in-budget@{K}")
    for s,v in report['results'].items(): print(f"{s:18s} {v['precision_at_5']:>10.3f}  {v['in_budget_at_5']:>11.3f}")
    print('\n'+report['caveat'])
    (A.ART/'evaluation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__ == '__main__': main()
