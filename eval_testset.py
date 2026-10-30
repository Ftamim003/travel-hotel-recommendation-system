"""Run data/recommendation_test_set_25.csv through the real recommender. python eval_testset.py [-v]
Expected IDs are AREA ids (D018 ...); the app ranks DISTRICTS, so an expected area counts as hit if its district is in the top 5.
Metrics: recall@5 (share of expected districts found), hit@5 (>=1 found), top1 (best result is expected)."""
import sys, os, warnings; warnings.filterwarnings('ignore'); os.environ.setdefault('TRAVELBD_NO_BROWSER','1')
import pandas as pd, app as A
T=pd.read_csv(A.DATA/'recommendation_test_set_25.csv',encoding='utf-8-sig'); d2dist=dict(zip(A.DEST['destination_id'].astype(str),A.DEST['district']))
SEASON={'winter':'Winter','monsoon':'Monsoon','summer':'Summer','any':'Any season'}
def run(t):
    rows=A.recommend_destinations(float(t.total_budget_bdt),int(t.trip_days),int(t.group_size),t.origin,'Any style',f"{t.interest}, {t.query_text}",'','',SEASON.get(str(t.season).lower(),'Any season'),'Any',limit=5)
    return [r['destination']['district'] for r in rows]
def main(v=False):
    rec=[];hit=[];top1=[]
    for t in T.itertuples():
        exp=list(dict.fromkeys(d2dist[i] for i in t.expected_top5_destination_ids_pipe_separated.split('|')))
        got=run(t); f=[e for e in exp if e in got]
        rec.append(len(f)/len(exp)); hit.append(bool(f)); top1.append(got[0] in exp)
        if v: print(f"{t.test_id} {t.interest:12s} {t.trip_days}d {int(t.total_budget_bdt):>6} | got {got} | expected {exp} | {len(f)}/{len(exp)}")
    n=len(T); print(f"recall@5 {sum(rec)/n:.2f}   hit@5 {sum(hit)/n:.2f}   top1 {sum(top1)/n:.2f}   ({n} tests)")
if __name__=='__main__': main('-v' in sys.argv)
