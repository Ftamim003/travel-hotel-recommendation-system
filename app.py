from pathlib import Path
import json, pickle, re, math, sqlite3, os, secrets, csv, smtplib, ssl, hmac, hashlib, time, threading
from email.message import EmailMessage
from functools import wraps
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, flash, session, abort
from sklearn.metrics.pairwise import cosine_similarity
from itinerary import generate_itinerary
from ml import hybrid as H

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; ART=ROOT/'artifacts'
app=Flask(__name__)
INSTANCE=ROOT/'instance'; INSTANCE.mkdir(exist_ok=True)
def _load_secret_key():
    """Env var wins; otherwise a per-install random key is created once in instance/ (never shipped)."""
    env=os.environ.get('TRAVELBD_SECRET_KEY')
    if env: return env
    f=INSTANCE/'secret_key.txt'
    if f.exists():
        key=f.read_text(encoding='utf-8').strip()
        if key: return key
    key=secrets.token_hex(32); f.write_text(key,encoding='utf-8')
    return key
app.config['SECRET_KEY']=_load_secret_key()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
                  SESSION_COOKIE_SECURE=os.environ.get('TRAVELBD_HTTPS')=='1',  # set TRAVELBD_HTTPS=1 behind HTTPS
                  MAX_CONTENT_LENGTH=1024*1024)
DB_PATH=INSTANCE/'travelbd_admin.sqlite3'

# User-visible modules that administrators can enable/disable per account.
USER_MENUS = {
    'explore': ('Trip recommendations', 'Create personalized destination recommendations'),
    'itinerary': ('Day-wise itinerary planner', 'Generate and save trip itineraries'),
    'destinations': ('Destination details', 'Open destination, hotel and transport details'),
    'model_info': ('Model & Data', 'View recommendation methodology and data limitations'),
    'profile': ('My profile', 'View saved searches, itineraries and account settings'),
}

def record_user_login(user_id):
    # Record a successful login for admin activity metrics.
    with db_connect() as con:
        con.execute('UPDATE users SET last_login_at=CURRENT_TIMESTAMP WHERE id=?',(user_id,))
        con.execute('INSERT INTO login_events(user_id) VALUES(?)',(user_id,))

def menu_allowed(menu_key, user_id=None):
    # Admins bypass user menu overrides; normal users default to allowed unless overridden.
    uid = user_id if user_id is not None else session.get('user_id')
    if not uid:
        return False
    with db_connect() as con:
        row=con.execute('SELECT role FROM users WHERE id=?',(uid,)).fetchone()
        if not row: return False
        if row['role']=='admin': return True
        perm=con.execute('SELECT allowed FROM user_menu_permissions WHERE user_id=? AND menu_key=?',(uid,menu_key)).fetchone()
    return bool(perm['allowed']) if perm else True

def menu_required(menu_key):
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            # Public visitors can browse destination information; signed-in users are subject to their menu rights.
            if not user_logged_in():
                return fn(*args, **kwargs)
            if not menu_allowed(menu_key):
                abort(403, description='Your account does not currently have access to this menu. Contact the administrator.')
            return fn(*args, **kwargs)
        return wrapped
    return decorator

def norm(v): return re.sub(r'\s+',' ',str(v or '')).strip().lower()
def num(v, default=0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except (ValueError,TypeError): return default

CONTENT_KEYS=['destination_name','district','division','destination_type','attractions','activities','best_season','accessibility_level','accessibility_modes','accessibility_notes']

def build_model(dest):
    """Fit the TF-IDF destination model. Used at startup (if the pickle is missing or from another scikit-learn) and after admin edits."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    text=dest.apply(lambda r:' '.join(str(r.get(k,'') or '') for k in CONTENT_KEYS),axis=1).fillna('')
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=12000,sublinear_tf=True,strip_accents='unicode')
    return {'vectorizer':vec,'matrix':vec.fit_transform(text),'destination_ids':dest['destination_id'].astype(str).tolist()}

def load():
    required=[DATA/'destinations_cleaned.csv', DATA/'hotels_cleaned.csv', DATA/'bpc_hotels_cleaned.csv', DATA/'reviews_cleaned.csv', DATA/'transport_cleaned.csv']
    if not all(p.exists() for p in required): raise RuntimeError('Processed data missing. Run: python train_model.py')
    dest=pd.read_csv(DATA/'destinations_cleaned.csv').fillna('')
    model=None; pkl=ART/'destination_content_model.pkl'
    if pkl.exists():
        try:
            with open(pkl,'rb') as f: model=pickle.load(f)   # local, self-generated artifact only
            if model.get('destination_ids')!=dest['destination_id'].astype(str).tolist(): model=None
        except Exception: model=None
    if model is None: model=build_model(dest)
    return (dest,pd.read_csv(DATA/'hotels_cleaned.csv').fillna(''),pd.read_csv(DATA/'bpc_hotels_cleaned.csv').fillna(''),pd.read_csv(DATA/'reviews_cleaned.csv').fillna(''),pd.read_csv(DATA/'transport_cleaned.csv').fillna(''),pd.read_csv(DATA/'destination_review_summary.csv').fillna(''),pd.read_csv(DATA/'hotel_review_summary.csv').fillna(''),pd.read_csv(DATA/'bantabsa_city_summary.csv').fillna('') if (DATA/'bantabsa_city_summary.csv').exists() else pd.DataFrame(),model)
DEST,HOTELS,BPC,REVIEWS,TRANSPORT,DEST_STATS,HOTEL_STATS,BANTABSA_CITY,MODEL=load()
DEST_STATS['review_mean_rating']=pd.to_numeric(DEST_STATS['review_mean_rating'],errors='coerce').fillna(0)
DEST_STATS['review_count']=pd.to_numeric(DEST_STATS['review_count'],errors='coerce').fillna(0)
HOTEL_STATS['review_mean_rating']=pd.to_numeric(HOTEL_STATS['review_mean_rating'],errors='coerce').fillna(0)
HOTEL_STATS['review_count']=pd.to_numeric(HOTEL_STATS['review_count'],errors='coerce').fillna(0)
if not BANTABSA_CITY.empty:
    BANTABSA_CITY['mean_rating']=pd.to_numeric(BANTABSA_CITY['mean_rating'],errors='coerce').fillna(0)
    BANTABSA_CITY['review_count']=pd.to_numeric(BANTABSA_CITY['review_count'],errors='coerce').fillna(0)

# ---------------- Destination (district) layer ----------------
# The 78 rows in destinations_cleaned.csv are "areas" (e.g. "Lalbagh Fort & Old Dhaka"). A DESTINATION is where a traveller
# stays overnight, so areas are grouped by district (64). Attractions stay as the child rows of itinerary_attractions.csv.
def slugify(v): return re.sub(r'[^a-z0-9]+','-',norm(v)).strip('-') or 'place'

def _uniq(values,sep=';'):
    out,seen=[],set()
    for v in values:
        for x in str(v).split(sep):
            x=x.strip()
            if x and x.lower() not in seen: seen.add(x.lower()); out.append(x)
    return out

COST_PARTS=['hotel_cost_per_room_per_night','meals_per_person_per_day','local_transport_per_group_per_day','attraction_fees_per_person_per_trip','intercity_round_trip_per_person']
def _cost_fields(g):
    """District-level cost inputs = mean over the district's areas (min and max kept apart). Blank fee = unknown (kept NaN, treated as 0 and flagged)."""
    out={}
    for p in COST_PARTS:
        for k in ('min','max'):
            col=f'{p}_{k}_bdt'; v=pd.to_numeric(g[col],errors='coerce').mean() if col in g else float('nan')
            out[col]=round(float(v),1) if v==v else ''
    out['cost_status']=str(g['cost_status'].iloc[0]) if 'cost_status' in g else ''
    out['budget_inclusions']=str(g['budget_inclusions'].iloc[0]) if 'budget_inclusions' in g else ''
    return out

def build_places(dest):
    rows=[]
    for district,g in dest.groupby('district',sort=True):
        g=g.sort_values('destination_id')
        lat=pd.to_numeric(g['latitude'],errors='coerce').mean(); lon=pd.to_numeric(g['longitude'],errors='coerce').mean()
        budget=pd.to_numeric(g['estimated_daily_budget_bdt'],errors='coerce').mean()
        level=g['accessibility_level'].astype(str).mode()
        seasons=_uniq(g['best_season'],'|')
        rows.append({'destination_id':slugify(district),'destination_name':str(district),'district':str(district),'division':str(g['division'].iloc[0]),
            'destination_type':' / '.join(_uniq(g['destination_type'],'|')),'type_list':_uniq(g['destination_type'],'|'),'attractions':'; '.join(_uniq(g['attractions'])),
            'activities':'; '.join(_uniq(g['activities'])),'areas':'; '.join(_uniq(g['destination_name'],'|')),
            'best_season':' | '.join(seasons),'best_season_list':seasons,
            'estimated_daily_budget_bdt':int(round(budget)) if budget==budget else 0,
            **_cost_fields(g),
            'accessibility_level':level.iloc[0] if len(level) else '','accessibility_modes':'; '.join(_uniq(g['accessibility_modes'])),
            'accessibility_notes':str(g['accessibility_notes'].iloc[0]),'latitude':round(lat,4) if lat==lat else '','longitude':round(lon,4) if lon==lon else '',
            'zone_ids':';'.join(g['destination_id'].astype(str)),
            'min_nights':int(pd.to_numeric(g['min_nights'],errors='coerce').max()) if 'min_nights' in g and pd.to_numeric(g['min_nights'],errors='coerce').notna().any() else '',
            'content_text':' '.join(' '.join(str(r.get(k,'') or '') for k in CONTENT_KEYS) for _,r in g.iterrows())})
    return pd.DataFrame(rows)

def build_place_model(places):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=12000,sublinear_tf=True,strip_accents='unicode')
    return {'vectorizer':vec,'matrix':vec.fit_transform(places['content_text'].fillna('')),'destination_ids':places['destination_id'].astype(str).tolist()}

PLACES=build_places(DEST); PLACE_MODEL=build_place_model(PLACES)
CF_MODEL,CF_INFO=None,{}

def attraction_signal(zids):
    """REAL review signal: visitor ratings of the attractions in a destination (supplied tourism geodata), weighted by rating count and
    shrunk toward the national average so a place with 3 ratings cannot outrank one with 3,000. Replaces the synthetic user reviews."""
    import itinerary
    a=itinerary.ATTRACTIONS
    if a is None or len(a)==0 or 'rating' not in a: return {'rating_count':0,'rated_places':0,'mean_rating':0,'raw_mean':0}
    r=pd.to_numeric(a['rating'],errors='coerce'); n=pd.to_numeric(a['rating_count'],errors='coerce').fillna(0)
    ok=(r>0)&(n>0)
    gm=float((r[ok]*n[ok]).sum()/n[ok].sum()) if ok.any() else 0.0
    sel=ok&a['destination_id'].astype(str).isin([str(z) for z in zids])
    cnt=float(n[sel].sum())
    if not cnt: return {'rating_count':0,'rated_places':0,'mean_rating':0,'raw_mean':0}
    raw=float((r[sel]*n[sel]).sum()/cnt); m=50.0
    return {'rating_count':int(cnt),'rated_places':int(sel.sum()),'raw_mean':raw,'mean_rating':(raw*cnt+gm*m)/(cnt+m)}

def get_place(pid):
    m=PLACES[PLACES['destination_id'].astype(str)==str(pid)]
    return m.iloc[0].to_dict() if not m.empty else None

def zone_ids_of(place): return [x for x in str(place.get('zone_ids','')).split(';') if x]

def resolve_zone_ids(did):
    p=get_place(did)
    return zone_ids_of(p) if p else [str(did)]

def place_stats(zids):
    s=DEST_STATS[DEST_STATS['destination_id'].astype(str).isin([str(z) for z in zids])]
    cnt=float(s['review_count'].sum())
    return {'review_mean_rating':float((s['review_mean_rating']*s['review_count']).sum()/cnt) if cnt else 0,'review_count':cnt}

def bantabsa_for_district(district):
    """Return a city-level BanTABSA aggregate; never imply it is a named hotel's rating."""
    if BANTABSA_CITY.empty: return {}
    aliases={"kuakata":"patuakhali", "chittagong":"chattogram"}
    key=norm(district).replace("’","'")
    key=aliases.get(key,key)
    x=BANTABSA_CITY[BANTABSA_CITY['city_key'].astype(str).map(lambda v: aliases.get(norm(v).replace("’","'"),norm(v)))==key]
    if x.empty: return {}
    # Combine location rows only when normalized keys coincide.
    count=float(x['review_count'].sum())
    rating=float((x['mean_rating']*x['review_count']).sum()/count) if count else 0
    return {'mean_rating':rating,'review_count':int(count),'positive_triplets':int(pd.to_numeric(x.get('positive_triplets',0),errors='coerce').fillna(0).sum()),'neutral_triplets':int(pd.to_numeric(x.get('neutral_triplets',0),errors='coerce').fillna(0).sum()),'negative_triplets':int(pd.to_numeric(x.get('negative_triplets',0),errors='coerce').fillna(0).sum()),'location_city':str(x.iloc[0]['location_city'])}

SEASON_MONTHS={'peak':{11,12,1,2},'shoulder':{3,4,10},'monsoon':{5,6,7,8,9}}   # matches the UI / transport labels
_MONTHS={m:i for i,m in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],1)}

def season_key(s):
    """Map 'Peak (Nov-Feb)', 'Shoulder (Mar-Apr, Oct)', 'Monsoon (May-Sep)' to peak/shoulder/monsoon."""
    s=norm(s)
    for k in SEASON_MONTHS:
        if s.startswith(k): return k
    return s

def months_in(text):
    """Months covered by a range such as 'Nov–Feb (dry/cool season)' (wraps over the year end)."""
    found=re.findall(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b',norm(text))
    if len(found)<2: return {_MONTHS[x] for x in found}
    cur,end=_MONTHS[found[0]],_MONTHS[found[1]]; out={cur}
    while cur!=end:
        cur=cur%12+1; out.add(cur)
    return out

def season_overlap(season,best_season):
    """Share of the wanted season's months that fall inside the destination's best-season range (0.5 = unknown)."""
    wanted=SEASON_MONTHS.get(season_key(season)); avail=months_in(best_season)
    return len(wanted&avail)/len(wanted) if wanted and avail else .5

def affordability(cost,budget):
    """1.0 while within budget, then falls linearly to 0 at 25% over budget. Being cheaper than the budget is NOT rewarded."""
    if budget<=0: return 0.0
    r=cost/budget
    return 1.0 if r<=1 else max(0.0,1-(r-1)*4)

def get_destination_stats(did):
    x=DEST_STATS[DEST_STATS['destination_id'].astype(str)==str(did)]
    return x.iloc[0].to_dict() if not x.empty else {}

DEFAULT_CHOICES={'any style','any','any season','anywhere'}
TYPE_W=.5   # share of the content signal that comes from destination-type match
# ---- Hybrid model: Score = w1*CBF + w2*CF + w3*Popularity + w4*Budget (see ml/hybrid.py). Weights come from artifacts/hybrid_weights.json,
# written by `python -m ml.tune_hybrid` (tuned on a validation split); PRIOR_WEIGHTS are used only if that file is missing.
CBF_SEASON_SHARE=1/6        # CBF = 5/6 interest match (TF-IDF + type) + 1/6 season fit
POP_MIX={'rating':.5,'volume':.25,'bantabsa':.25}   # Popularity = shrunk attraction rating + review volume + BanTABSA city score
HYBRID_W,HYBRID_META=H.load_weights(ART/'hybrid_weights.json')
def build_cf_model():
    """Fit CF on the real survey (data/user_survey.csv) if present, else on the synthetic trips (flagged, weight capped). Never blocks start-up."""
    try:
        z2d=dict(zip(DEST['destination_id'].astype(str),DEST['district']))
        return H.build_cf(DATA,z2d,sorted(PLACES['district'].astype(str).unique()))
    except Exception as exc:
        print('CF model unavailable:',exc); return None,{'real':False,'n_users':0,'source':'none'}
CF_MODEL,CF_INFO=build_cf_model()


INTEREST_TYPES={  # free-text interest words -> destination_type values in the data (so "culture" finds Heritage, "birdwatching" finds Haor/Wetland ...)
 'beach':['Beach','Island'],'sea':['Beach','Island'],'sunset':['Beach'],'coast':['Beach','Island','Mangrove'],'coastal':['Beach','Island','Mangrove'],'island':['Island','River/Island','Beach'],
 'hill':['Hill','Waterfall'],'hills':['Hill','Waterfall'],'trek':['Hill','Waterfall'],'trekking':['Hill','Waterfall'],'hiking':['Hill','Waterfall'],'adventure':['Hill','Waterfall','Wildlife/Forest'],'waterfall':['Waterfall'],'waterfalls':['Waterfall'],
 'forest':['Wildlife/Forest','Mangrove'],'wildlife':['Wildlife/Forest','Mangrove'],'mangrove':['Mangrove'],'nature':['Wildlife/Forest','Hill','Lake','Haor/Wetland','Tea Estate'],
 'bird':['Haor/Wetland','Wildlife/Forest'],'birds':['Haor/Wetland','Wildlife/Forest'],'birdwatching':['Haor/Wetland','Wildlife/Forest'],'wetland':['Haor/Wetland'],'wetlands':['Haor/Wetland'],'haor':['Haor/Wetland'],
 'tea':['Tea Estate'],'lake':['Lake','Haor/Wetland'],'lakes':['Lake','Haor/Wetland'],'river':['River/Island'],'riverside':['River/Island'],
 'culture':['Heritage','Archaeological','Urban'],'cultural':['Heritage','Archaeological','Urban'],'heritage':['Heritage','Archaeological'],'history':['Heritage','Archaeological'],'historic':['Heritage','Archaeological'],'museum':['Heritage','Archaeological'],'museums':['Heritage','Archaeological'],
 'palace':['Heritage'],'palaces':['Heritage'],'temple':['Heritage','Archaeological'],'archaeology':['Archaeological'],'city':['Urban','Heritage'],'urban':['Urban','Heritage'],'food':['Urban','Heritage'],
 'rural':['Rural/Agro'],'village':['Rural/Agro'],'farm':['Rural/Agro'],'quiet':['Tea Estate','Lake','Hill','Rural/Agro'],'family':['Beach','Lake','Wildlife/Forest','Heritage'],
}
def implied_types(text):
    out=[]
    for w in re.findall(r'[a-z]+',str(text).lower()):
        for t in INTEREST_TYPES.get(w,[]):
            if t not in out: out.append(t)
    return out

def cheapest_fares(origin):
    """destination_id -> cheapest one-way fare (BDT per person) from `origin`; {} when the origin has no transport data."""
    if not origin or norm(origin) in ('','anywhere'): return {}
    t=TRANSPORT[TRANSPORT['origin'].astype(str).map(norm)==norm(origin)].copy()
    if t.empty: return {}
    t['f']=pd.to_numeric(t['fare_bdt'],errors='coerce')
    return t.dropna(subset=['f']).groupby(t['destination_id'].astype(str))['f'].min().to_dict()

def trip_cost(d,days,group_size,origin_fare_rt=None):
    """Whole-trip cost range (low, high, breakdown) from the researched components; None if the destination has no cost inputs.
    rooms = ceil(group/2); nights = days-1; meals per person per day; local transport per group per day, scaled by ceil(group/3) vehicles
    (assumption: one CNG carries 3); attraction fee per person per trip (blank = unknown -> 0, flagged); intercity round trip per person.
    origin_fare_rt = (low, high) round-trip fare per person from the traveller's own origin when the transport table has it."""
    g=lambda p,k:num(d.get(f'{p}_{k}_bdt'),float('nan'))
    if g(COST_PARTS[0],'min')!=g(COST_PARTS[0],'min'): return None
    rooms=max(1,math.ceil(group_size/2)); nights=max(0,days-1); veh=max(1,math.ceil(group_size/3))
    mult={COST_PARTS[0]:rooms*nights,COST_PARTS[1]:group_size*days,COST_PARTS[2]:days*veh,COST_PARTS[3]:group_size,COST_PARTS[4]:group_size}
    lo={};hi={}
    for p in COST_PARTS:
        a,b=g(p,'min'),g(p,'max'); a=0 if a!=a else a; b=a if b!=b else b
        lo[p]=a*mult[p]; hi[p]=b*mult[p]
    if origin_fare_rt:
        lo[COST_PARTS[4]]=origin_fare_rt[0]*group_size; hi[COST_PARTS[4]]=origin_fare_rt[1]*group_size
    names=['hotel','meals','local_transport','attraction_fees','intercity']
    return sum(lo.values()),sum(hi.values()),{n:(round(lo[p]),round(hi[p])) for n,p in zip(names,COST_PARTS)}

def origin_fares(origin):
    """destination_id (area) -> (low, high) ROUND-TRIP fare per person from `origin`: cheapest mode vs median mode in the transport table. {} when no data or origin is Dhaka/unset (Dhaka uses the researched BRTA-based fares)."""
    if not origin or norm(origin) in ('','anywhere','dhaka'): return {}
    t=TRANSPORT[TRANSPORT['origin'].astype(str).map(norm)==norm(origin)].copy()
    if t.empty: return {}
    t['f']=pd.to_numeric(t['fare_bdt'],errors='coerce'); t=t.dropna(subset=['f'])
    gr=t.groupby(t['destination_id'].astype(str))['f']
    return {k:(2*lo,2*hi) for k,lo,hi in zip(gr.min().index,gr.min().values,gr.median().values)}

_SIG_CACHE={}
def _static_signals(dest_df):
    """Per-destination signals that do not depend on the traveller (attraction ratings, BanTABSA). Cached until the destination table or the
    attraction table object is replaced by an admin refresh (the cache holds references, so ids cannot be reused)."""
    import itinerary
    c=_SIG_CACHE.get('v')
    if c and c[0] is dest_df and c[1] is itinerary.ATTRACTIONS: return c[2]
    val=({str(r['destination_id']):attraction_signal(zone_ids_of(r)) for _,r in dest_df.iterrows()},
         {str(r['district']):bantabsa_for_district(r.get('district','')) for _,r in dest_df.iterrows()})
    _SIG_CACHE['v']=(dest_df,itinerary.ATTRACTIONS,val); return val

def recommend_destinations(budget,days,group_size,origin,travel_style,interests,activities,facilities,season,accessibility,limit=12,interest_tags=(),history=(),weights=None,cf_model=None,cf_info=None):
    """Hybrid ranking: Score = w1*CBF + w2*CF + w3*Popularity + w4*Budget, then x trip-length fit. `budget` is the whole-trip budget.
    history = districts the (logged-in) user has already liked/saved: adds an item-item CF signal. weights/cf_model/cf_info are for offline
    evaluation (validation/test CF models); the app passes none and uses the tuned weights + the CF model fitted at start-up."""
    cf_model=CF_MODEL if cf_model is None else cf_model; cf_info=CF_INFO if cf_info is None else cf_info
    W=weights if weights is not None else H.effective_weights(HYBRID_W,cf_info)
    cf_by_district={}
    if cf_model is not None and cf_model.n_users and W['cf']>0:
        cf_by_district=dict(zip(cf_model.districts,cf_model.score(travel_style,budget/max(1,days*group_size),group_size,history)))
    dest_df,model=PLACES,PLACE_MODEL                      # one consistent snapshot even if an admin refresh happens mid-request
    query=' '.join(x for x in [travel_style,interests,activities,facilities,accessibility] if x and norm(x) not in DEFAULT_CHOICES)
    sims=cosine_similarity(model['vectorizer'].transform([query or 'Bangladesh tourism destination']),model['matrix']).ravel() if query else None
    top=float(sims.max()) if sims is not None and len(sims) else 0.0
    row_of={str(i):n for n,i in enumerate(model['destination_ids'])}
    sigs,bant_by_district=_static_signals(dest_df)
    max_reviews=max([1]+[v['rating_count'] for v in sigs.values()])
    tags={norm(t) for t in interest_tags if norm(t)}
    implied=False
    if not tags: tags={norm(t) for t in implied_types(' '.join([interests or '',travel_style or '',activities or '']))}; implied=bool(tags)   # words typed in the interests box count like ticked chips
    ofares=origin_fares(origin)
    rows=[]
    for _,d in dest_df.iterrows():
        n=row_of.get(str(d['destination_id']))
        content=float(sims[n]) if sims is not None and n is not None else 0.0
        content_rel=content/top if top>0 else 0.0   # best match = 1.0, so similarity is comparable to the other 0-1 signals
        if tags:   # ticked interest chips are categories: share of them that this destination actually offers
            have={norm(t) for t in (d.get('type_list') or [])}
            type_match=len(tags&have)/len(tags)
            content_rel=(1-TYPE_W)*content_rel+TYPE_W*type_match
        else: type_match=0.0
        zf=[ofares[z] for z in zone_ids_of(d) if z in ofares]; ofr=(min(a for a,b in zf),min(b for a,b in zf)) if zf else None
        tc=trip_cost(d,days,group_size,ofr)
        if tc: cost_lo,cost_hi,breakdown=tc; trip_est=(cost_lo+cost_hi)/2   # budget fit is judged on the middle of the researched range
        else:
            daily=num(d.get('estimated_daily_budget_bdt'),0); trip_est=daily*days*group_size if daily>0 else None; cost_lo=cost_hi=trip_est; breakdown={}
        budget_score=affordability(trip_est,budget) if trip_est is not None else .5
        stat=sigs[str(d['destination_id'])]
        rating=num(stat.get('mean_rating'),0); rating_score=min(1,max(0,(rating-3.5)/1.5)) if rating else .5   # 3.5 stars -> 0, 5 stars -> 1
        count=num(stat.get('rating_count'),0)
        popularity=min(1,math.log1p(count)/math.log1p(max_reviews)) if count else 0
        season_match=max(season_overlap(season,b) for b in (d.get('best_season_list') or [d.get('best_season','')])) if season and norm(season)!='any season' else .5
        bantabsa=bant_by_district.get(str(d.get('district','')),{})
        bantabsa_rating=num(bantabsa.get('mean_rating'),0)
        bantabsa_score=(bantabsa_rating/5) if bantabsa_rating else .5
        comps={'cbf':(1-CBF_SEASON_SHARE)*content_rel+CBF_SEASON_SHARE*season_match,
               'cf':float(cf_by_district.get(str(d.get('district','')),0.0)),
               'popularity':POP_MIX['rating']*rating_score+POP_MIX['volume']*popularity+POP_MIX['bantabsa']*bantabsa_score,
               'budget':budget_score}
        score=H.hybrid_score(comps,W)
        # Trip-length fit: min_nights = nights needed for the district's best area (0 = day trip). Soft penalties, never a hard filter.
        mn=d.get('min_nights'); nights=max(0,days-1); trip_fit=1.0
        if mn not in ('',None):
            mn=int(mn)
            if mn==0 and nights>=2: trip_fit=.6              # day-trip-only district for a multi-night trip
            elif mn>=3 and nights<2: trip_fit=.5 if nights==0 else .75   # long-trip destination (3+ nights) squeezed into a day trip / one night
        score*=trip_fit
        rows.append({'destination':d.to_dict(),'content':content,'content_rel':content_rel,'type_match':type_match,'rating':rating,'review_count':int(count),'rated_places':int(stat.get('rated_places',0)),'budget_score':budget_score,'estimated_trip_cost':trip_est,'trip_cost_low':cost_lo,'trip_cost_high':cost_hi,'cost_breakdown':breakdown,'season_match':season_match,'bantabsa':bantabsa,'bantabsa_score':bantabsa_score,'popularity':popularity,'rating_score':rating_score,'trip_fit':trip_fit,'components':comps,'cf_score':comps['cf'],'weights':W,'score':score})
    rows.sort(key=lambda x:x['score'],reverse=True)
    return rows[:limit]

SHOW_SYNTHETIC_HOTELS=False   # synthetic placeholder rows now carry no price/rating/address, so they are hidden; set True only for demos
LODGING_SHARE=0.4   # heuristic assumption, not a measured value

HOTEL_W={'cbf':.20,'cf':0.0,'popularity':.35,'budget':.45}   # same 4-signal form as destinations; NOT tuned (no labelled hotel data). CF=0: no real hotel interactions.
def recommend_hotels(destination_id, budget_per_day, group_size, facilities=''):
    candidates=HOTELS[HOTELS['destination_id'].astype(str).isin(resolve_zone_ids(destination_id))].copy()
    if candidates.empty: return []
    facilities_wanted={norm(x) for x in re.split(r'[;,]',facilities) if norm(x)}
    out=[]
    for _,h in candidates.iterrows():
        price=num(h.get('price_per_night_bdt'),0)
        fac={norm(x) for x in str(h.get('facilities','')).split(';') if norm(x)}
        facility_match=len(facilities_wanted & fac)/len(facilities_wanted) if facilities_wanted else .5
        stats=HOTEL_STATS[HOTEL_STATS['hotel_id'].astype(str)==str(h['hotel_id'])]
        hs=stats.iloc[0].to_dict() if not stats.empty else {}
        rating=num(h.get('hotel_rating'),0) or num(hs.get('review_mean_rating'),0)
        rating_score=rating/5 if rating else .5
        nightly_group=price*max(1,math.ceil(group_size/2)) if price else None
        # Room price shared by up to two guests vs. an ASSUMED lodging share (40%) of the per-person daily budget.
        per_person_night=price/min(2,max(1,group_size)) if price else None
        eligible=str(h.get('budget_filter_eligible','')).lower()=='true' or str(h.get('price_status',''))=='official_portal_listed_range'
        price_known=bool(price) and ('budget_filter_eligible' not in h.index or eligible)   # unverified / synthetic rates never drive the budget score
        budget_score=affordability(per_person_night,max(budget_per_day,1)*LODGING_SHARE) if price_known else .5
        volume=min(1,math.log1p(num(hs.get('review_count'),0))/math.log1p(max(1,num(HOTEL_STATS['review_count'].max(),1))))
        comps={'cbf':facility_match,'cf':0.0,'popularity':(rating_score*.30+volume*.05)/.35,'budget':budget_score}   # cf = 0: there are no real hotel-level interactions to learn from
        score=H.hybrid_score(comps,HOTEL_W)
        out.append({'hotel':h.to_dict(),'components':comps,'price':price or None,'rating':rating,'review_count':int(num(hs.get('review_count'),0)),'facility_match':facility_match,'score':score,'group_nightly_estimate':nightly_group,'source_type':h.get('source_type','Unknown'),'is_real':str(h.get('source_type',''))!='Synthetic placeholder'})
    out.sort(key=lambda x:x['score'],reverse=True)
    # Real listings (official BPC or admin-imported) always come first; synthetic demo rows only fill the remaining slots and are labelled.
    real=[x for x in out if x['is_real']][:8]; demo=[x for x in out if not x['is_real']]
    return real+(demo[:max(0,min(3 if real else 6,8-len(real)))] if SHOW_SYNTHETIC_HOTELS else [])

def get_transport(origin,destination_id,season):
    t=TRANSPORT[TRANSPORT['destination_id'].astype(str).isin(resolve_zone_ids(destination_id))].copy()
    if origin and norm(origin) not in ('','anywhere'):
        direct=t[t['origin'].astype(str).map(norm)==norm(origin)]
        if direct.empty: return []
        t=direct
    if season and season!='Any season':
        key=season_key(season)
        filtered=t[t['travel_season'].astype(str).map(season_key)==key]
        if not filtered.empty: t=filtered
    t['fare_bdt_num']=pd.to_numeric(t['fare_bdt'],errors='coerce')
    t['travel_time_num']=pd.to_numeric(t['travel_time_hrs'],errors='coerce')
    t=t.sort_values(['fare_bdt_num','travel_time_num'],na_position='last').drop_duplicates(subset=['origin','transport_mode'])
    return t.head(6).to_dict('records')

def render_index(error='',selected=(),extra=''):
    districts=sorted(PLACES['district'].astype(str).unique())
    profile=get_current_user() if user_logged_in() else None
    return render_template('index.html',profile_interests=extra or (profile or {}).get('interests',''),districts=districts,preview=PLACES.head(6).to_dict('records'),
        interest_choices=sorted(DEST['destination_type'].astype(str).unique()),selected=set(selected),error=error,total=len(PLACES),area_total=len(DEST),hotel_total=len(HOTELS),transport_total=len(TRANSPORT))

@app.route('/',methods=['GET','POST'])
def index():
    ensure_csrf()
    if request.method=='POST':
        if not user_logged_in():
            flash('Create a free account or sign in to get personalized trip recommendations and save your searches.','success')
            return redirect(url_for('user_login', next=url_for('index')))
        if not menu_allowed('explore'):
            abort(403, description='Trip recommendations are disabled for your account. Contact the administrator.')
        budget=max(100,num(request.form.get('budget_per_person_day'),2000)); days=max(1,min(30,int(num(request.form.get('days'),3))))
        group_size=max(1,min(30,int(num(request.form.get('group_size'),2))))
        origin=request.form.get('origin','Anywhere').strip() or 'Anywhere'
        travel_style=request.form.get('travel_style','Any style').strip()
        chosen=[x.strip() for x in request.form.getlist('interest_tags') if x.strip()]
        extra=request.form.get('interests','').strip()
        interests=' '.join(chosen+([extra] if extra else []))
        activities=request.form.get('activities','').strip(); facilities=request.form.get('facilities','').strip()
        season=request.form.get('season','Any season').strip(); accessibility=request.form.get('accessibility','Any').strip()
        if not interests:
            return render_index('Please pick at least one interest so the model has something to match.',chosen,extra),400
        budget_total=budget*days*group_size
        results=recommend_destinations(budget_total,days,group_size,origin,travel_style,interests,activities,facilities,season,accessibility,interest_tags=chosen)
        if user_logged_in():
            preferences={'budget_per_person_day':budget,'days':days,'group_size':group_size,'origin':origin,'travel_style':travel_style,'interests':interests,'activities':activities,'facilities':facilities,'season':season,'accessibility':accessibility,'recommended_destination_ids':[str(x['destination'].get('destination_id','')) for x in results]}
            with db_connect() as con:
                con.execute('INSERT INTO saved_searches(user_id,preferences_json) VALUES(?,?)',(session['user_id'],json.dumps(preferences,ensure_ascii=False)))
        return render_template('results.html',cf_real=CF_INFO.get('real',False),cf_users=CF_INFO.get('n_users',0),hybrid_w=(results[0]['weights'] if results else {}),results=results,budget=budget,days=days,group_size=group_size,origin=origin,travel_style=travel_style,interests=interests,activities=activities,facilities=facilities,season=season,budget_total=budget_total)
    return render_index()

def _legacy_zone_redirect(endpoint,destination_id):
    """Old links used area IDs such as D001; send them to the destination (district) page."""
    z=DEST[DEST['destination_id'].astype(str)==str(destination_id)]
    if z.empty: return None
    return redirect(url_for(endpoint,destination_id=slugify(z.iloc[0]['district']),**request.args.to_dict()))

@app.route('/destination/<destination_id>')
@menu_required('destinations')
def destination_detail(destination_id):
    d=get_place(destination_id)
    if d is None:
        r=_legacy_zone_redirect('destination_detail',destination_id)
        return r if r else render_template('detail.html',destination=None,hotels=[],transport=[])
    daily=num(d.get('estimated_daily_budget_bdt'),0)
    budget=num(request.args.get('budget'),max(100,daily)); group_size=max(1,int(num(request.args.get('group_size'),2)))
    facilities=request.args.get('facilities',''); origin=request.args.get('origin','Anywhere'); season=request.args.get('season','Any season')
    hotels=recommend_hotels(destination_id,budget,group_size,facilities)
    transport=get_transport(origin,destination_id,season)
    stat=place_stats(zone_ids_of(d))
    bantabsa=bantabsa_for_district(d.get('district',''))
    return render_template('detail.html',destination=d,hotels=hotels,transport=transport,stat=stat,bantabsa=bantabsa,budget=budget,group_size=group_size,origin=origin,season=season,
        days=int(num(request.args.get('days'),3)),interests=request.args.get('interests',''),travel_style=request.args.get('travel_style','Any style'),facilities=facilities)

@app.route('/destination/<destination_id>/itinerary', methods=['GET','POST'])
def destination_itinerary(destination_id):
    ensure_csrf()
    if user_logged_in() and not menu_allowed('itinerary'):
        abort(403, description='The itinerary planner is disabled for your account. Contact the administrator.')
    if request.method=='POST' or request.args.get('auto'):
        if not user_logged_in():
            flash('Sign in to generate and save a personalized day-wise itinerary.','success')
            return redirect(url_for('user_login', next=request.path))
    d=get_place(destination_id)
    if d is None:
        r=_legacy_zone_redirect('destination_itinerary',destination_id)
        return r if r else render_template('itinerary_form.html', destination=None, error='Destination not found.', budget=2000, days=3, group_size=2, travel_style='Any style', interests='', pace='Balanced', stay_budget=2000, facilities='')
    args=request.form if request.method=='POST' else request.args
    default_budget=max(100,num(args.get('daily_budget') or args.get('budget'),num(d.get('estimated_daily_budget_bdt'),2000)))
    default_group=max(1,int(num(args.get('group_size'),2)))
    stay_budget=max(100,num(args.get('stay_budget') or args.get('budget'),default_budget)); facilities=args.get('facilities','')
    days=max(1,min(14,int(num(args.get('days'),3)))); travel_style=args.get('travel_style','Any style').strip()
    interests=args.get('interests','').strip(); pace=args.get('pace','Balanced').strip()
    if request.method=='POST' or args.get('auto'):
        budget=default_budget; group_size=max(1,min(30,default_group))
        plan,note=generate_itinerary(d,days,budget,group_size,travel_style,interests,pace)
        stay=recommend_hotels(destination_id,stay_budget,group_size,facilities)[:3]
        if user_logged_in():
            with db_connect() as con:
                con.execute('INSERT INTO saved_itineraries(user_id,destination_id,destination_name,days,budget_bdt,group_size,travel_style,interests,pace,plan_json,note) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(session['user_id'],str(d.get('destination_id','')),str(d.get('destination_name','')),days,budget,group_size,travel_style,interests,pace,json.dumps(plan,ensure_ascii=False,default=str),note))
        return render_template('itinerary.html',destination=d,plan=plan,note=note,days=days,budget=budget,group_size=group_size,travel_style=travel_style,interests=interests,pace=pace,stay=stay,stay_budget=stay_budget,facilities=facilities)
    return render_template('itinerary_form.html',destination=d,budget=default_budget,days=days,group_size=default_group,travel_style=travel_style,interests=interests,pace=pace,stay_budget=stay_budget,facilities=facilities,error='')

# -------------------- User authentication and saved travel activity --------------------
def user_logged_in():
    return bool(session.get('user_id'))

def user_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not user_logged_in():
            return redirect(url_for('user_login', next=request.path))
        return fn(*args, **kwargs)
    return wrapped

def get_current_user():
    if not user_logged_in():
        return None
    with db_connect() as con:
        row=con.execute('SELECT id, full_name, email, phone, home_district, interests, created_at FROM users WHERE id=?',(session['user_id'],)).fetchone()
    return dict(row) if row else None

# -------------------- Registration with email OTP --------------------
OTP_TTL=600; OTP_RESEND_WAIT=60; OTP_MAX_ATTEMPTS=5
PHONE_RE=re.compile(r'(?:\+?88)?(01[3-9]\d{8})')

def normalize_phone(raw):
    """Accept 01XXXXXXXXX, 8801XXXXXXXXX or +8801XXXXXXXXX; return +8801XXXXXXXXX or None."""
    m=PHONE_RE.fullmatch(re.sub(r'[\s\-()]','',raw or ''))
    return '+88'+m.group(1) if m else None

def _otp_hash(email,otp):
    return hmac.new(app.config['SECRET_KEY'].encode(),f'{email}:{otp}'.encode(),hashlib.sha256).hexdigest()

def _email_settings():
    cfg={}
    f=INSTANCE/'email_config.json'
    if f.exists():
        try: cfg=json.loads(f.read_text(encoding='utf-8'))
        except Exception as exc: app.logger.error('Bad instance/email_config.json: %s',exc)
    g=lambda k,env: os.environ.get(env) or cfg.get(k,'')
    return {'host':g('host','TRAVELBD_SMTP_HOST'),'port':int(g('port','TRAVELBD_SMTP_PORT') or 587),'user':g('user','TRAVELBD_SMTP_USER'),'password':g('password','TRAVELBD_SMTP_PASSWORD').replace(' ',''),'sender':g('from','TRAVELBD_SMTP_FROM')}

def email_configured():
    st=_email_settings(); return bool(st['host'] and st['user'])

_OTP_SENDS={}; OTP_IP_LIMIT=8; OTP_IP_WINDOW=3600
def _otp_ip_limited():
    """At most 8 OTP emails per IP per hour, so the form cannot be used to mail arbitrary addresses from the configured account."""
    ip=request.remote_addr or '?'; now=time.time()
    recent=[t for t in _OTP_SENDS.get(ip,[]) if now-t<OTP_IP_WINDOW]
    if len(recent)>=OTP_IP_LIMIT:
        _OTP_SENDS[ip]=recent; return True
    recent.append(now); _OTP_SENDS[ip]=recent
    if len(_OTP_SENDS)>5000:
        for k in [k for k,v in _OTP_SENDS.items() if not v or now-v[-1]>=OTP_IP_WINDOW]: _OTP_SENDS.pop(k,None)
    return False

def send_otp_email(to_email,otp,purpose='verify'):
    """Send the OTP by SMTP. Without SMTP settings (demo mode) the code is printed in the server terminal only."""
    if _otp_ip_limited():
        app.logger.warning('OTP email rate limit hit for %s',request.remote_addr); return False
    st=_email_settings()
    if not st['host'] or not st['user']:
        print(f'\n[TravelBD demo mode] Email is not configured (create instance/email_config.json). OTP for {to_email}: {otp}\n')
        return True
    msg=EmailMessage()
    reset=purpose=='reset'
    msg['Subject']='Your TravelBD password reset code' if reset else 'Your TravelBD verification code'
    msg['From']=st['sender'] or st['user']; msg['To']=to_email
    what='password reset' if reset else 'verification'
    ignore='If you did not ask to reset your password, ignore this email; your password has not changed.' if reset else 'If you did not try to register, ignore this email.'
    msg.set_content(f'Your TravelBD {what} code is {otp}.\n\nIt expires in {OTP_TTL//60} minutes. {ignore}')
    try:
        if st['port']==465:
            with smtplib.SMTP_SSL(st['host'],465,timeout=20,context=ssl.create_default_context()) as sm:
                sm.login(st['user'],st['password']); sm.send_message(msg)
        else:
            with smtplib.SMTP(st['host'],st['port'],timeout=20) as sm:
                sm.starttls(context=ssl.create_default_context()); sm.login(st['user'],st['password']); sm.send_message(msg)
        return True
    except Exception as exc:
        app.logger.error('Could not send OTP email: %s',exc); return False

def _issue_otp(email):
    """Create a new OTP for a pending registration, store only its hash, and email it."""
    otp=f'{secrets.randbelow(10**6):06d}'
    with db_connect() as con:
        con.execute('UPDATE pending_registrations SET otp_hash=?, expires_at=?, attempts=0, last_sent_at=? WHERE email=?',(_otp_hash(email,otp),time.time()+OTP_TTL,time.time(),email))
    return send_otp_email(email,otp)

@app.route('/register', methods=['GET','POST'])
def user_register():
    if user_logged_in(): return redirect(url_for('user_profile'))
    ensure_csrf()
    if request.method=='POST':
        full_name=request.form.get('full_name','').strip()
        email=request.form.get('email','').strip().lower()
        phone=normalize_phone(request.form.get('phone',''))
        password=request.form.get('password','')
        confirm=request.form.get('confirm_password','')
        if len(full_name)<2: flash('Please enter your full name.','error')
        elif not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+",email): flash('Enter a valid email address.','error')
        elif not phone: flash('Enter a valid Bangladeshi mobile number, e.g. 01712345678.','error')
        elif len(password)<10: flash('Use a password with at least 10 characters.','error')
        elif password!=confirm: flash('Passwords do not match.','error')
        else:
            with db_connect() as con:
                email_taken=con.execute('SELECT 1 FROM users WHERE email=?',(email,)).fetchone()
                phone_taken=con.execute('SELECT 1 FROM users WHERE phone=?',(phone,)).fetchone()
                if not email_taken and not phone_taken:
                    con.execute('INSERT OR REPLACE INTO pending_registrations(email,full_name,phone,password_hash,otp_hash,expires_at,attempts,last_sent_at) VALUES(?,?,?,?,?,?,?,?)',(email,full_name,phone,generate_password_hash(password),'',0,0,0))
            if email_taken: flash('An account with that email already exists. Please sign in.','error')
            elif phone_taken: flash('That mobile number is already registered.','error')
            else:
                session['pending_email']=email
                if _issue_otp(email): flash((f'We sent a 6-digit code to {email}. It expires in {OTP_TTL//60} minutes.' if email_configured() else 'Email sending is not set up on this server (demo mode). The code is shown in the server terminal.'),'success')
                else: flash('We could not send the email right now. Press "Resend code" to try again.','error')
                return redirect(url_for('verify_email'))
    return render_template('user_register.html')

@app.route('/verify-email', methods=['GET','POST'])
def verify_email():
    if user_logged_in(): return redirect(url_for('user_profile'))
    ensure_csrf()
    email=session.get('pending_email','')
    with db_connect() as con: p=con.execute('SELECT * FROM pending_registrations WHERE email=?',(email,)).fetchone() if email else None
    if not p:
        flash('Start by creating your account.','error'); return redirect(url_for('user_register'))
    if request.method=='POST':
        code=re.sub(r'\D','',request.form.get('otp',''))
        if time.time()>p['expires_at']: flash('That code has expired. Press "Resend code" for a new one.','error')
        elif p['attempts']>=OTP_MAX_ATTEMPTS: flash('Too many wrong attempts. Press "Resend code" for a new one.','error')
        elif not p['otp_hash'] or not hmac.compare_digest(p['otp_hash'],_otp_hash(email,code)):
            with db_connect() as con: con.execute('UPDATE pending_registrations SET attempts=attempts+1 WHERE email=?',(email,))
            flash('Incorrect code. Please check your email and try again.','error')
        else:
            try:
                with db_connect() as con:
                    cur=con.execute("INSERT INTO users(full_name,email,password_hash,phone,role) VALUES(?,?,?,?,'user')",(p['full_name'],email,p['password_hash'],p['phone']))
                    con.execute('DELETE FROM pending_registrations WHERE email=?',(email,))
                    user_id=cur.lastrowid
            except sqlite3.IntegrityError:
                flash('An account with that email already exists. Please sign in.','error'); return redirect(url_for('user_login'))
            _reset_session(); session['user_id']=user_id; session['user_name']=p['full_name']; ensure_csrf()
            record_user_login(user_id)
            flash('Email verified. Your TravelBD account is ready.','success')
            return redirect(url_for('user_profile'))
    return render_template('verify_email.html',email=email)

@app.route('/resend-otp', methods=['POST'])
def resend_otp():
    email=session.get('pending_email','')
    with db_connect() as con: p=con.execute('SELECT last_sent_at FROM pending_registrations WHERE email=?',(email,)).fetchone() if email else None
    if not p: return redirect(url_for('user_register'))
    wait=int(OTP_RESEND_WAIT-(time.time()-p['last_sent_at']))
    if wait>0: flash(f'Please wait {wait} seconds before requesting another code.','error')
    elif _issue_otp(email): flash('A new code has been sent.' if email_configured() else 'Demo mode: the new code is shown in the server terminal.','success')
    else: flash('We could not send the email right now. Please try again shortly.','error')
    return redirect(url_for('verify_email'))

# -------------------- Forgot password (email OTP) --------------------
@app.route('/forgot-password', methods=['GET','POST'])
def forgot_password():
    if user_logged_in(): return redirect(url_for('user_profile'))
    ensure_csrf()
    if request.method=='POST':
        email=request.form.get('email','').strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+",email):
            flash('Enter a valid email address.','error')
        else:
            with db_connect() as con:
                exists=con.execute('SELECT 1 FROM users WHERE email=?',(email,)).fetchone()
                rec=con.execute('SELECT last_sent_at FROM password_resets WHERE email=?',(email,)).fetchone()
                recent=bool(rec and time.time()-rec['last_sent_at']<OTP_RESEND_WAIT)
                if exists and not recent:
                    otp=f'{secrets.randbelow(10**6):06d}'
                    con.execute('INSERT OR REPLACE INTO password_resets(email,otp_hash,expires_at,attempts,last_sent_at) VALUES(?,?,?,0,?)',(email,_otp_hash('reset:'+email,otp),time.time()+OTP_TTL,time.time()))
                else:
                    otp=None
                    print(f'[TravelBD] Password reset asked for {email}: ' + ('no account with this email, nothing sent.' if not exists else 'a code was sent less than a minute ago, nothing sent.'))
            if otp: send_otp_email(email,otp,'reset')
            session['reset_email']=email
            # Same message whether or not the account exists, so nobody can probe which emails are registered.
            flash(f'If an account exists for {email}, we sent a 6-digit code. It expires in {OTP_TTL//60} minutes. You can request another after {OTP_RESEND_WAIT} seconds.' if email_configured() else 'Demo mode: email is not set up, so the code is shown in the server terminal.','success')
            return redirect(url_for('reset_password'))
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['GET','POST'])
def reset_password():
    if user_logged_in(): return redirect(url_for('user_profile'))
    ensure_csrf()
    email=session.get('reset_email','')
    if not email: return redirect(url_for('forgot_password'))
    if request.method=='POST':
        code=re.sub(r'\D','',request.form.get('otp',''))
        password=request.form.get('password',''); confirm=request.form.get('confirm_password','')
        with db_connect() as con: rec=con.execute('SELECT * FROM password_resets WHERE email=?',(email,)).fetchone()
        if len(password)<10: flash('Use a password with at least 10 characters.','error')
        elif password!=confirm: flash('Passwords do not match.','error')
        elif not rec or time.time()>rec['expires_at'] or rec['attempts']>=OTP_MAX_ATTEMPTS:
            flash('That code is invalid or has expired. Request a new one.','error')
        elif not hmac.compare_digest(rec['otp_hash'],_otp_hash('reset:'+email,code)):
            with db_connect() as con: con.execute('UPDATE password_resets SET attempts=attempts+1 WHERE email=?',(email,))
            flash('Incorrect code. Please check your email and try again.','error')
        else:
            with db_connect() as con:
                con.execute('UPDATE users SET password_hash=? WHERE email=?',(generate_password_hash(password),email))
                con.execute('DELETE FROM password_resets WHERE email=?',(email,))
            for k in [k for k in _FAILS if k[1]==email]: _FAILS.pop(k,None)
            _reset_session(); ensure_csrf()
            flash('Your password has been changed. Please sign in.','success')
            return redirect(url_for('user_login'))
    return render_template('reset_password.html',email=email)

@app.route('/login', methods=['GET','POST'])
def user_login():
    if user_logged_in(): return redirect(url_for('admin_dashboard') if admin_logged_in() else url_for('user_profile'))
    ensure_csrf()
    if request.method=='POST':
        email=request.form.get('email','').strip().lower(); password=request.form.get('password','')
        key=_throttle_key(email)
        if _locked(key):
            flash('Too many failed attempts. Try again in a few minutes.','error')
            return render_template('user_login.html'),429
        with db_connect() as con: user=con.execute('SELECT id, full_name, email, password_hash, role FROM users WHERE email=?',(email,)).fetchone()
        if user and check_password_hash(user['password_hash'],password):
            _FAILS.pop(key,None)
            _reset_session(); session['user_id']=user['id']; session['user_name']=user['full_name']; ensure_csrf()
            record_user_login(user['id'])
            is_admin=user['role']=='admin'
            next_url=request.args.get('next','')
            if not next_url.startswith('/') or next_url.startswith('//') or '\\' in next_url or (next_url.startswith('/admin') and not is_admin):
                next_url=url_for('admin_dashboard') if is_admin else url_for('user_profile')
            return redirect(next_url)
        _FAILS[key]=(_FAILS.get(key,(0,0))[0]+1,time.time())
        if len(_FAILS)>10000:
            for k in [k for k,v in _FAILS.items() if time.time()-v[1]>=LOCK_SECONDS]: _FAILS.pop(k,None)
        flash('Email or password is incorrect.','error')
    return render_template('user_login.html')

@app.route('/logout', methods=['POST'])
@user_required
def user_logout():
    _reset_session(); flash('You have been signed out.','success')
    return redirect(url_for('index'))

def _phone_used_by_other(phone,uid):
    with db_connect() as con: return bool(con.execute('SELECT 1 FROM users WHERE phone=? AND id<>?',(phone,uid)).fetchone())

@app.route('/profile', methods=['GET','POST'])
@user_required
@menu_required('profile')
def user_profile():
    ensure_csrf()
    if request.method=='POST':
        full_name=request.form.get('full_name','').strip()
        phone_raw=request.form.get('phone','').strip()
        phone=normalize_phone(phone_raw) if phone_raw else ''
        home_district=request.form.get('home_district','').strip()
        interests=request.form.get('interests','').strip()
        if len(full_name)<2:
            flash('Please enter a name with at least 2 characters.','error')
        elif phone is None: flash('Enter a valid Bangladeshi mobile number, e.g. 01712345678.','error')
        elif phone and _phone_used_by_other(phone,session['user_id']): flash('That mobile number is already registered to another account.','error')
        elif len(home_district)>100 or len(interests)>500:
            flash('One or more fields are too long.','error')
        else:
            with db_connect() as con:
                con.execute('UPDATE users SET full_name=?, phone=?, home_district=?, interests=? WHERE id=?',(full_name,phone,home_district,interests,session['user_id']))
            session['user_name']=full_name
            flash('Your profile has been updated.','success')
            return redirect(url_for('user_profile'))
    user=get_current_user()
    if not user:
        _reset_session(); return redirect(url_for('user_login'))
    with db_connect() as con:
        searches=con.execute('SELECT id, preferences_json, created_at FROM saved_searches WHERE user_id=? ORDER BY id DESC LIMIT 10',(user['id'],)).fetchall()
        trips=con.execute('SELECT id, destination_name, days, budget_bdt, group_size, travel_style, interests, pace, note, created_at FROM saved_itineraries WHERE user_id=? ORDER BY id DESC LIMIT 10',(user['id'],)).fetchall()
    search_rows=[]
    for r in searches:
        item=dict(r)
        try: item['preferences']=json.loads(item.pop('preferences_json'))
        except (ValueError,TypeError): item['preferences']={}
        search_rows.append(item)
    return render_template('user_profile.html',user=user,searches=search_rows,trips=[dict(r) for r in trips])

@app.route('/profile/itinerary/<int:trip_id>')
@user_required
def saved_itinerary(trip_id):
    with db_connect() as con:
        trip=con.execute('SELECT * FROM saved_itineraries WHERE id=? AND user_id=?',(trip_id,session['user_id'])).fetchone()
    if not trip: abort(404)
    trip=dict(trip)
    try: plan=json.loads(trip['plan_json'])
    except (ValueError,TypeError): plan=[]
    destination=get_place(trip['destination_id'])
    if destination is None:
        z=DEST[DEST['destination_id'].astype(str)==str(trip['destination_id'])]
        destination=get_place(slugify(z.iloc[0]['district'])) if not z.empty else None
    destination=destination or {'destination_id':trip['destination_id'],'destination_name':trip['destination_name']}
    return render_template('itinerary.html',destination=destination,plan=plan,note=trip['note'],days=trip['days'],budget=trip['budget_bdt'],group_size=trip['group_size'],travel_style=trip['travel_style'],interests=trip['interests'],pace=trip['pace'])

@app.route('/model-info')
def model_info():
    if user_logged_in() and not menu_allowed('model_info'):
        abort(403, description='Model & Data is disabled for your account. Contact the administrator.')
    p=ART/'training_report.json'; report=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rp=ART/'hybrid_report.json'; hr=json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {}
    return render_template('model_info.html',report=report,hybrid_w=HYBRID_W,hybrid_eff=H.effective_weights(HYBRID_W,CF_INFO),hybrid_meta=HYBRID_META,cf_info=CF_INFO,hybrid_report=hr,hotel_w=HOTEL_W)



# -------------------- Admin: persistent CRUD and dataset overview --------------------
ENTITY_CONFIG = {
    'destinations': {
        'label': 'Destinations', 'file': 'destinations_cleaned.csv', 'key': 'destination_id',
        'fields': [
            ('destination_id','Destination ID','text',True), ('destination_name','Destination name','text',True),
            ('district','District','text',True), ('division','Division','text',True),
            ('destination_type','Destination type','text',True), ('attractions','Attractions (separate with semicolons)','textarea',False),
            ('best_season','Best season','text',False), ('estimated_daily_budget_bdt','Estimated daily budget (BDT)','number',True),
            ('activities','Activities (separate with semicolons)','textarea',False), ('accessibility_level','Accessibility level','text',False),
            ('accessibility_modes','Accessibility modes','text',False), ('accessibility_notes','Accessibility notes','textarea',False),
            ('latitude','Latitude','number',False), ('longitude','Longitude','number',False)
        ]
    },
    'attractions': {
        'label': 'Attractions', 'file': 'itinerary_attractions.csv', 'key': 'place_id',
        'fields': [
            ('place_id','Place ID','text',False), ('destination_id','Destination','select_destination',True),
            ('place_name','Attraction name','text',True), ('category','Category','text',True),
            ('latitude','Latitude','number',False), ('longitude','Longitude','number',False),
            ('estimated_visit_hours','Estimated visit duration (hours)','number',True),
            ('estimated_visit_cost_bdt','Estimated visit cost (BDT)','number',True),
            ('rating','Rating (optional)','number',False), ('rating_count','Rating count (optional)','number',False),
            ('source_type','Source type','text',True), ('estimate_note','Notes / verification caveat','textarea',False)
        ]
    },
    'hotels': {
        'label': 'Hotels', 'file': 'hotels_cleaned.csv', 'key': 'hotel_id',
        'fields': [
            ('hotel_id','Hotel ID','text',False), ('destination_id','Destination','select_destination',True),
            ('hotel_name','Hotel name','text',True), ('district','District','text',True),
            ('address','Address','textarea',False), ('contact_phone','Contact phone','text',False),
            ('booking_link','Official listing / booking URL','url',False), ('price_per_night_bdt','Price per night (BDT; leave blank if unknown)','number',False),
            ('star_class','Star class (optional)','number',False), ('hotel_rating','Hotel rating (optional)','number',False),
            ('facilities','Facilities (separate with semicolons)','textarea',False), ('room_types','Room types','text',False),
            ('latitude','Latitude','number',False), ('longitude','Longitude','number',False),
            ('data_source','Data source','textarea',True), ('source_type','Source type','select_source',True),
            ('verification_status','Verification status','select_verification',True)
        ]
    }
}


def db_connect():
    con=sqlite3.connect(DB_PATH)
    con.row_factory=sqlite3.Row
    return con

def db_rows(kind):
    with db_connect() as con:
        rows=con.execute('SELECT record_key, payload FROM records WHERE entity=? ORDER BY record_key',(kind,)).fetchall()
    return [(r['record_key'], json.loads(r['payload'])) for r in rows]

def csv_write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    # Preserve a stable union of fields across records; keep headers even when the final row is deleted.
    fields=[]
    if not rows and path.exists():
        try:
            with path.open('r',encoding='utf-8-sig',newline='') as old:
                fields=next(csv.reader(old),[])
        except OSError:
            fields=[]
        if fields:
            with path.open('w',encoding='utf-8-sig',newline='') as f:
                csv.writer(f).writerow(fields)
        return
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if pd.isna(v) else v) if not isinstance(v,(dict,list)) else json.dumps(v,ensure_ascii=False) for k,v in row.items()})

def compute_content(row):
    return ' '.join(str(row.get(k,'') or '') for k in CONTENT_KEYS)

def sync_entity_to_csv(kind):
    cfg=ENTITY_CONFIG[kind]
    rows=[payload for _,payload in db_rows(kind)]
    if kind=='destinations':
        for r in rows: r['content_text']=compute_content(r)
        csv_write(DATA/'destinations_cleaned.csv',rows)
    elif kind=='attractions':
        csv_write(DATA/'itinerary_attractions.csv',rows)
    elif kind=='hotels':
        csv_write(DATA/'hotels_cleaned.csv',rows)
        bpc=[r for r in rows if str(r.get('source_type',''))=='Official BPC listing']
        csv_write(DATA/'bpc_hotels_cleaned.csv',bpc)

_REFRESH_LOCK=threading.RLock()

def refresh_runtime_data():
    """Reload CSV-backed public data and rebuild TF-IDF after admin changes. New objects are built first and swapped in together."""
    global DEST,HOTELS,BPC,MODEL,PLACES,PLACE_MODEL
    with _REFRESH_LOCK:
        dest=pd.read_csv(DATA/'destinations_cleaned.csv').fillna('')
        hotels=pd.read_csv(DATA/'hotels_cleaned.csv').fillna('')
        bpc=pd.read_csv(DATA/'bpc_hotels_cleaned.csv').fillna('') if (DATA/'bpc_hotels_cleaned.csv').exists() else pd.DataFrame()
        dest['content_text']=dest.apply(lambda r: compute_content(r.to_dict()),axis=1)
        model=build_model(dest)
        ART.mkdir(exist_ok=True)
        tmp=ART/'destination_content_model.pkl.tmp'
        with open(tmp,'wb') as f: pickle.dump(model,f)
        os.replace(tmp,ART/'destination_content_model.pkl')          # atomic: never a half-written pickle
        attractions=pd.DataFrame()
        try:
            if (DATA/'itinerary_attractions.csv').exists():
                attractions=pd.read_csv(DATA/'itinerary_attractions.csv').fillna('')
                for c in ['latitude','longitude','rating','rating_count','estimated_visit_hours','estimated_visit_cost_bdt']:
                    if c in attractions: attractions[c]=pd.to_numeric(attractions[c],errors='coerce')
        except Exception as exc:
            app.logger.exception('Could not refresh itinerary data: %s',exc)
        places=build_places(dest); place_model=build_place_model(places)
        DEST,HOTELS,BPC,MODEL,PLACES,PLACE_MODEL=dest,hotels,bpc,model,places,place_model
        try:
            import itinerary
            itinerary.ATTRACTIONS=attractions
        except Exception as exc:
            app.logger.exception('Could not refresh itinerary data: %s',exc)

def init_admin_db():
    with db_connect() as con:
        con.execute('CREATE TABLE IF NOT EXISTS admins (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
        con.execute('CREATE TABLE IF NOT EXISTS records (entity TEXT NOT NULL, record_key TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(entity,record_key))')
        con.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL, email TEXT NOT NULL UNIQUE COLLATE NOCASE, password_hash TEXT NOT NULL, phone TEXT DEFAULT '', home_district TEXT DEFAULT '', interests TEXT DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        user_columns=[r[1] for r in con.execute('PRAGMA table_info(users)')]
        if 'role' not in user_columns:
            con.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        if 'last_login_at' not in user_columns:
            con.execute('ALTER TABLE users ADD COLUMN last_login_at TEXT')
        con.execute('''CREATE TABLE IF NOT EXISTS login_events (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, logged_in_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)''')
        con.execute('CREATE INDEX IF NOT EXISTS idx_login_events_logged_in_at ON login_events(logged_in_at)')
        con.execute('''CREATE TABLE IF NOT EXISTS user_menu_permissions (user_id INTEGER NOT NULL, menu_key TEXT NOT NULL, allowed INTEGER NOT NULL CHECK(allowed IN (0,1)), updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(user_id,menu_key), FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)''')
        # One-time migration: accounts from the old separate admins table become admin users.
        for a in con.execute('SELECT username,password_hash FROM admins').fetchall():
            email=a['username'].strip().lower()
            if con.execute('SELECT 1 FROM users WHERE email=?',(email,)).fetchone():
                con.execute("UPDATE users SET role='admin' WHERE email=?",(email,))
            else:
                con.execute("INSERT INTO users(full_name,email,password_hash,role) VALUES(?,?,?,'admin')",('Admin',email,a['password_hash']))
        con.execute('''CREATE TABLE IF NOT EXISTS pending_registrations (email TEXT PRIMARY KEY COLLATE NOCASE, full_name TEXT NOT NULL, phone TEXT NOT NULL, password_hash TEXT NOT NULL, otp_hash TEXT NOT NULL, expires_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, last_sent_at REAL NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        con.execute('''CREATE TABLE IF NOT EXISTS password_resets (email TEXT PRIMARY KEY COLLATE NOCASE, otp_hash TEXT NOT NULL, expires_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, last_sent_at REAL NOT NULL)''')
        con.execute('''CREATE TABLE IF NOT EXISTS saved_searches (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, preferences_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)''')
        con.execute('''CREATE TABLE IF NOT EXISTS saved_itineraries (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, destination_id TEXT NOT NULL, destination_name TEXT NOT NULL, days INTEGER NOT NULL, budget_bdt REAL NOT NULL, group_size INTEGER NOT NULL, travel_style TEXT DEFAULT '', interests TEXT DEFAULT '', pace TEXT DEFAULT '', plan_json TEXT NOT NULL, note TEXT DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)''')
        for kind,cfg in ENTITY_CONFIG.items():
            count=con.execute('SELECT COUNT(*) FROM records WHERE entity=?',(kind,)).fetchone()[0]
            source=DATA/cfg['file']
            if count==0 and source.exists():
                df=pd.read_csv(source).fillna('')
                for _,row in df.iterrows():
                    payload=row.to_dict()
                    if kind=='hotels' and not payload.get('verification_status'):
                        payload['verification_status']=('Official source listed; details not independently verified' if payload.get('source_type')=='Official BPC listing' else 'Synthetic/demo data' if payload.get('source_type')=='Synthetic placeholder' else 'Needs verification')
                    key=str(payload.get(cfg['key'],'')).strip()
                    if not key:
                        key=f'{kind[:1].upper()}{secrets.token_hex(4)}'
                        payload[cfg['key']]=key
                    con.execute('INSERT OR IGNORE INTO records(entity,record_key,payload) VALUES(?,?,?)',(kind,key,json.dumps(payload,ensure_ascii=False,default=str)))
    # Add a transparent verification label for existing hotel records if this is an older DB.
    for key,payload in db_rows('hotels'):
        if not payload.get('verification_status'):
            payload['verification_status']=('Official source listed; details not independently verified' if payload.get('source_type')=='Official BPC listing' else 'Synthetic/demo data' if payload.get('source_type')=='Synthetic placeholder' else 'Needs verification')
            with db_connect() as con:
                con.execute('UPDATE records SET payload=?, updated_at=CURRENT_TIMESTAMP WHERE entity=? AND record_key=?',(json.dumps(payload,ensure_ascii=False,default=str),'hotels',key))
    # SQLite is the persistent admin store; materialize its records back to CSV.
    for kind in ENTITY_CONFIG:
        if db_rows(kind): sync_entity_to_csv(kind)
    refresh_runtime_data()

def admin_logged_in():
    uid=session.get('user_id')
    if not uid: return False
    with db_connect() as con:
        row=con.execute('SELECT role FROM users WHERE id=?',(uid,)).fetchone()
    return bool(row and row['role']=='admin')

def admin_required(fn):
    @wraps(fn)
    def wrapped(*args,**kwargs):
        if not user_logged_in():
            return redirect(url_for('user_login',next=request.path))
        if not admin_logged_in():
            abort(403,description='This page is for administrators only.')
        return fn(*args,**kwargs)
    return wrapped

@app.context_processor
def inject_admin_context():
    allowed={key:menu_allowed(key) for key in USER_MENUS} if user_logged_in() else {}
    return {'admin_logged_in':admin_logged_in(), 'user_logged_in':bool(session.get('user_id')), 'user_name':session.get('user_name',''), 'csrf_value':session.get('_csrf_token',''), 'menu_allowed':allowed}

BOOT_ID=secrets.token_hex(8)   # new value on every server start
def _reset_session():
    """Clear the session but keep it marked as belonging to this server run (otherwise the next request would treat it as an old one and log the user out)."""
    session.clear(); session['_boot_id']=BOOT_ID
@app.before_request
def start_as_visitor_after_restart():
    """Every time the server is (re)started, old browser sessions are dropped so the site opens as a visitor page; sign in again if needed.
    Browsers can keep session cookies across restarts (Edge 'continue where you left off'), so the cookie alone is not enough.
    Set TRAVELBD_KEEP_LOGIN=1 to keep people signed in across restarts."""
    if os.environ.get('TRAVELBD_KEEP_LOGIN')=='1': return
    if session.get('_boot_id')!=BOOT_ID:
        session.clear(); session['_boot_id']=BOOT_ID

@app.before_request
def protect_admin_posts():
    if request.method=='POST' and (request.path=='/' or request.path.endswith('/itinerary') or request.path.startswith('/admin') or request.path in ('/register','/login','/logout','/profile','/verify-email','/resend-otp','/forgot-password','/reset-password')):
        token=session.get('_csrf_token','')
        submitted=request.form.get('_csrf_token','')
        if not token or not secrets.compare_digest(token,submitted):
            abort(400,description='Invalid or expired form token. Refresh the page and try again.')

def ensure_csrf():
    if not session.get('_csrf_token'):
        session['_csrf_token']=secrets.token_urlsafe(32)

@app.route('/admin')
def admin_home():
    if not user_logged_in(): return redirect(url_for('user_login',next='/admin'))
    if not admin_logged_in(): abort(403,description='This page is for administrators only.')
    return redirect(url_for('admin_dashboard'))

# Brute-force protection: 5 failed attempts per (IP, username) -> 5 minute lockout.
import time
_FAILS={}; MAX_FAILS=5; LOCK_SECONDS=300
def _throttle_key(username): return (request.remote_addr or '?', username.lower())
def _locked(key):
    rec=_FAILS.get(key)
    if rec and rec[0]>=MAX_FAILS:
        if time.time()-rec[1]<LOCK_SECONDS: return True
        _FAILS.pop(key,None)
    return False

@app.route('/admin/login')
def admin_login():
    return redirect(url_for('user_login',next='/admin'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    counts={kind:len(db_rows(kind)) for kind in ENTITY_CONFIG}
    with db_connect() as con:
        counts['users']=con.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0]
        counts['admins']=con.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        counts['logins_today']=con.execute("SELECT COUNT(DISTINCT user_id) FROM login_events WHERE date(logged_in_at)=date('now','localtime')").fetchone()[0]
        counts['login_events_today']=con.execute("SELECT COUNT(*) FROM login_events WHERE date(logged_in_at)=date('now','localtime')").fetchone()[0]
        counts['active_7d']=con.execute("SELECT COUNT(DISTINCT user_id) FROM login_events WHERE logged_in_at >= datetime('now','-7 days','localtime')").fetchone()[0]
        counts['never_logged_in']=con.execute("SELECT COUNT(*) FROM users WHERE role='user' AND last_login_at IS NULL").fetchone()[0]
    return render_template('admin_dashboard.html',counts=counts,username=session.get('user_name','Admin'))

@app.route('/admin/users')
@admin_required
def admin_users():
    query=request.args.get('q','').strip()
    with db_connect() as con:
        if query:
            like='%'+query+'%'
            users=con.execute("SELECT id,full_name,email,phone,role,created_at,last_login_at FROM users WHERE full_name LIKE ? OR email LIKE ? ORDER BY created_at DESC LIMIT 500",(like,like)).fetchall()
        else:
            users=con.execute("SELECT id,full_name,email,phone,role,created_at,last_login_at FROM users ORDER BY created_at DESC LIMIT 500").fetchall()
    return render_template('admin_users.html',users=[dict(u) for u in users],query=query)

@app.route('/admin/users/<int:user_id>/menu-rights',methods=['GET','POST'])
@admin_required
def admin_user_menu_rights(user_id):
    with db_connect() as con:
        user=con.execute('SELECT id,full_name,email,role FROM users WHERE id=?',(user_id,)).fetchone()
        if not user: abort(404)
        if user['role']=='admin':
            flash('Administrator accounts always have full menu access.','error')
            return redirect(url_for('admin_users'))
        if request.method=='POST':
            for key in USER_MENUS:
                allowed=1 if request.form.get('menu_'+key)=='on' else 0
                con.execute('INSERT INTO user_menu_permissions(user_id,menu_key,allowed,updated_at) VALUES(?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(user_id,menu_key) DO UPDATE SET allowed=excluded.allowed,updated_at=CURRENT_TIMESTAMP',(user_id,key,allowed))
            flash('Menu permissions updated for '+user['email']+'.','success')
            return redirect(url_for('admin_user_menu_rights',user_id=user_id))
        existing={r['menu_key']:bool(r['allowed']) for r in con.execute('SELECT menu_key,allowed FROM user_menu_permissions WHERE user_id=?',(user_id,)).fetchall()}
    rights={key:existing.get(key,True) for key in USER_MENUS}
    return render_template('admin_menu_rights.html',user=dict(user),menus=USER_MENUS,rights=rights)

@app.route('/admin/<kind>')
@admin_required
def admin_list(kind):
    if kind not in ENTITY_CONFIG: abort(404)
    cfg=ENTITY_CONFIG[kind]; records=[r for _,r in db_rows(kind)]
    query=request.args.get('q','').strip().lower()
    if query: records=[r for r in records if query in ' '.join(str(v) for v in r.values()).lower()]
    records=records[:500]
    return render_template('admin_list.html',kind=kind,cfg=cfg,records=records,query=query)

def validate_record(kind,form,existing=None):
    cfg=ENTITY_CONFIG[kind]; existing=existing or {}; data=dict(existing); errors=[]
    for field,label,typ,required in cfg['fields']:
        raw=form.get(field,'').strip()
        if required and not raw: errors.append(f'{label} is required.'); continue
        if typ=='number' and raw:
            try:
                value=float(raw)
                if not math.isfinite(value): raise ValueError()
                if field in ('estimated_daily_budget_bdt','estimated_visit_hours','estimated_visit_cost_bdt','price_per_night_bdt','star_class','hotel_rating','rating','rating_count') and value<0: errors.append(f'{label} cannot be negative.'); continue
                if field in ('estimated_visit_hours','estimated_daily_budget_bdt') and value<=0: errors.append(f'{label} must be greater than zero.'); continue
                if field in ('latitude',) and not -90<=value<=90: errors.append('Latitude must be between -90 and 90.'); continue
                if field in ('longitude',) and not -180<=value<=180: errors.append('Longitude must be between -180 and 180.'); continue
                data[field]=value
            except ValueError: errors.append(f'{label} must be a valid number.'); continue
        else:
            data[field]=raw
        if typ=='url' and raw and not re.match(r'^https?://',raw,re.I): errors.append(f'{label} must begin with http:// or https://.')
        if typ=='select_source' and raw not in ('Synthetic placeholder','Official BPC listing','Official district directory','Other source — unverified'):
            errors.append('Choose a valid hotel source type.')
        if typ=='select_verification' and raw not in ('Official source listed; details not independently verified','Verified by admin','Needs verification','Synthetic/demo data'):
            errors.append('Choose a valid verification status.')
    if kind=='destinations':
        data['content_text']=compute_content(data)
    if kind=='attractions':
        dest_ids=set(DEST['destination_id'].astype(str))
        if str(data.get('destination_id','')) not in dest_ids: errors.append('Select an existing destination.')
        data.setdefault('destination_name',str(DEST.loc[DEST['destination_id'].astype(str)==str(data.get('destination_id','')),'destination_name'].iloc[0]) if str(data.get('destination_id','')) in dest_ids else '')
        data.setdefault('destination_district',str(DEST.loc[DEST['destination_id'].astype(str)==str(data.get('destination_id','')),'district'].iloc[0]) if str(data.get('destination_id','')) in dest_ids else '')
        data.setdefault('rating_count',0); data.setdefault('source_type','Admin-added'); data.setdefault('location_match_method','Admin managed'); data.setdefault('source_location',data.get('destination_district',''))
        data.setdefault('estimate_note','Planning estimate; verify current access, hours and fees.')
    if kind=='hotels':
        dest=DEST[DEST['destination_id'].astype(str)==str(data.get('destination_id',''))]
        if dest.empty: errors.append('Select an existing destination.')
        else:
            data['district']=data.get('district') or str(dest.iloc[0]['district'])
        if data.get('source_type')=='Official BPC listing' and not data.get('data_source'): errors.append('Official listings must include their source.')
    return data,errors

@app.route('/admin/<kind>/new',methods=['GET','POST'])
@admin_required
def admin_new(kind):
    if kind not in ENTITY_CONFIG: abort(404)
    ensure_csrf(); cfg=ENTITY_CONFIG[kind]; data={}; errors=[]
    if request.method=='POST':
        data,errors=validate_record(kind,request.form)
        key=str(data.get(cfg['key'],'')).strip()
        if kind=='attractions' and not key: key='P'+secrets.token_hex(4).upper(); data[cfg['key']]=key
        if kind=='hotels' and not key: key=('BPC' if data.get('source_type')=='Official BPC listing' else 'H')+secrets.token_hex(4).upper(); data[cfg['key']]=key
        if kind=='destinations' and not key: errors.append('Destination ID is required.')
        if key and any(k==key for k,_ in db_rows(kind)): errors.append(f'{cfg["key"]} must be unique.')
        if not errors:
            if kind=='destinations': data['content_text']=compute_content(data)
            with db_connect() as con: con.execute('INSERT INTO records(entity,record_key,payload) VALUES(?,?,?)',(kind,key,json.dumps(data,ensure_ascii=False,default=str)))
            sync_entity_to_csv(kind); refresh_runtime_data(); flash(f'{cfg["label"][:-1] if cfg["label"].endswith("s") else cfg["label"]} added successfully. Public recommendations have been refreshed.','success')
            return redirect(url_for('admin_list',kind=kind))
    return render_template('admin_form.html',kind=kind,cfg=cfg,data=data,errors=errors,mode='Add',destinations=DEST.to_dict('records'))

@app.route('/admin/<kind>/<record_key>/edit',methods=['GET','POST'])
@admin_required
def admin_edit(kind,record_key):
    if kind not in ENTITY_CONFIG: abort(404)
    cfg=ENTITY_CONFIG[kind]; matches=dict(db_rows(kind))
    if record_key not in matches: abort(404)
    ensure_csrf(); original=matches[record_key]; data=dict(original); errors=[]
    if request.method=='POST':
        data,errors=validate_record(kind,request.form,original)
        new_key=str(data.get(cfg['key'],record_key)).strip()
        if not new_key: errors.append(f'{cfg["key"]} is required.')
        if new_key!=record_key: errors.append(f'{cfg["key"]} cannot be changed after the record is created.')
        if not errors:
            if kind=='destinations': data['content_text']=compute_content(data)
            with db_connect() as con:
                con.execute('DELETE FROM records WHERE entity=? AND record_key=?',(kind,record_key))
                con.execute('INSERT INTO records(entity,record_key,payload,updated_at) VALUES(?,?,?,CURRENT_TIMESTAMP)',(kind,new_key,json.dumps(data,ensure_ascii=False,default=str)))
            sync_entity_to_csv(kind); refresh_runtime_data(); flash('Record updated and public recommendation data refreshed.','success')
            return redirect(url_for('admin_list',kind=kind))
    return render_template('admin_form.html',kind=kind,cfg=cfg,data=data,errors=errors,mode='Edit',destinations=DEST.to_dict('records'))

@app.route('/admin/<kind>/<record_key>/delete',methods=['POST'])
@admin_required
def admin_delete(kind,record_key):
    if kind not in ENTITY_CONFIG: abort(404)
    cfg=ENTITY_CONFIG[kind]
    with db_connect() as con:
        row=con.execute('SELECT payload FROM records WHERE entity=? AND record_key=?',(kind,record_key)).fetchone()
        if not row: abort(404)
        payload=json.loads(row['payload'])
        if kind=='destinations':
            if len(db_rows('destinations')) <= 1:
                flash('At least one destination must remain in the recommendation dataset.','error')
                return redirect(url_for('admin_list',kind=kind))
            # Prevent deleting a destination that still owns attractions or hotels.
            linked=sum(1 for _,r in db_rows('attractions') if str(r.get('destination_id'))==record_key)+sum(1 for _,r in db_rows('hotels') if str(r.get('destination_id'))==record_key)
            if linked:
                flash(f'Cannot delete this destination while {linked} attraction/hotel records are linked to it. Reassign or delete them first.','error')
                return redirect(url_for('admin_list',kind=kind))
        con.execute('DELETE FROM records WHERE entity=? AND record_key=?',(kind,record_key))
    sync_entity_to_csv(kind); refresh_runtime_data(); flash('Record deleted and public data refreshed.','success')
    return redirect(url_for('admin_list',kind=kind))

@app.route('/admin/datasets')
@admin_required
def admin_datasets():
    datasets=[]
    for name in ['1_destinations.csv','destinations_cleaned.csv','itinerary_attractions.csv','attractions_cleaned.csv','hotels_cleaned.csv','bpc_hotels_cleaned.csv','2b_real_bpc_hotels.csv','reviews_cleaned.csv','BanTABSA.csv','bantabsa_city_summary.csv','transport_cleaned.csv','4_transport.csv']:
        path=DATA/name
        if not path.exists(): continue
        try:
            df=pd.read_csv(path,low_memory=False)
            missing=int(df.isna().sum().sum())
            total_cells=int(df.shape[0]*df.shape[1])
            sources=[]
            for col in ['source_type','data_source','source','Source']:
                if col in df.columns:
                    sources=[str(x) for x in df[col].dropna().astype(str).unique()[:5]]; break
            datasets.append({'name':name,'rows':len(df),'columns':len(df.columns),'missing':missing,'missing_pct':round(missing*100/total_cells,1) if total_cells else 0,'fields':', '.join(df.columns.tolist()),'sources':sources})
        except Exception as exc: datasets.append({'name':name,'error':str(exc),'rows':0,'columns':0,'missing':0,'missing_pct':0,'fields':'','sources':[]})
    return render_template('admin_datasets.html',datasets=datasets,counts={k:len(db_rows(k)) for k in ENTITY_CONFIG})

@app.route('/admin/refresh-model',methods=['POST'])
@admin_required
def admin_refresh_model():
    ensure_csrf(); refresh_runtime_data(); flash('Recommendation features rebuilt from the current destination data.','success')
    return redirect(url_for('admin_dashboard'))

# Initialize persistent admin store and refresh the public data on app startup.
init_admin_db()
# Create or promote the deployment admin using environment variables.
admin_email = os.environ.get('TRAVELBD_ADMIN_EMAIL', '').strip().lower()
admin_password = os.environ.get('TRAVELBD_ADMIN_PASSWORD', '')

if admin_email and admin_password:
    with db_connect() as con:
        existing = con.execute(
            'SELECT id FROM users WHERE email=?',
            (admin_email,)
        ).fetchone()

        if existing:
            con.execute(
                "UPDATE users SET role='admin' WHERE email=?",
                (admin_email,)
            )
        else:
            con.execute(
                """INSERT INTO users
                (full_name, email, password_hash, role)
                VALUES (?, ?, ?, 'admin')""",
                (
                    'TravelBD Administrator',
                    admin_email,
                    generate_password_hash(admin_password)
                )
            )

    print('[TravelBD] Deployment admin is ready.')
with db_connect() as _c:
    if not _c.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]:
        print('\n[TravelBD] No admin account exists yet. Create one from this terminal:  python manage.py create-admin\n')

if __name__=='__main__':
    import threading, webbrowser
    debug=os.environ.get('TRAVELBD_DEBUG')=='1'
    # Open the browser once (not in the debug reloader's parent process). Set TRAVELBD_NO_BROWSER=1 to disable.
    if not os.environ.get('TRAVELBD_NO_BROWSER') and (not debug or os.environ.get('WERKZEUG_RUN_MAIN')=='true'):
        threading.Timer(1.5,lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=debug)
