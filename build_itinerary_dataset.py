"""Build an itinerary-ready attraction table from TravelBD's supplied datasets.

The generated table distinguishes destination-listed places, geospatial tourism records,
and district-level chatbot suggestions. Cost and visit duration are planning estimates,
not verified opening hours, entry prices, or travel times.
Run: python build_itinerary_dataset.py
"""
from pathlib import Path
import json, math, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'itinerary_attractions.csv'


def norm(v):
    return re.sub(r'[^a-z0-9\u0980-\u09ff]+', ' ', str(v or '').lower()).strip()

def haversine(lat1, lon1, lat2, lon2):
    try:
        vals = [float(lat1), float(lon1), float(lat2), float(lon2)]
        if not all(math.isfinite(x) for x in vals): return 99999.0
        p1, p2 = math.radians(vals[0]), math.radians(vals[2])
        dp = math.radians(vals[2]-vals[0]); dl = math.radians(vals[3]-vals[1])
        a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 6371 * 2 * math.asin(min(1, math.sqrt(a)))
    except (ValueError, TypeError): return 99999.0

def duration_for(category, title=''):
    c = norm(category + ' ' + title)
    if any(k in c for k in ['national park','forest','hiking','waterfall','hill','tea garden','beach','island']): return 2.5
    if any(k in c for k in ['museum','fort','palace','archaeological','historical','landmark','temple','mosque']): return 1.5
    if any(k in c for k in ['market','bridge','river','viewpoint','lake','park','garden']): return 1.0
    return 1.25

def cost_for(category, title=''):
    c = norm(category + ' ' + title)
    if any(k in c for k in ['national park','safari','amusement','museum','boat','island']): return 300
    if any(k in c for k in ['fort','palace','archaeological','museum','historical place']): return 150
    if any(k in c for k in ['beach','river','bridge','market','viewpoint','park','garden','mosque','temple']): return 100
    return 150

def main():
    dest = pd.read_csv(DATA/'1_destinations.csv').fillna('')
    geo_path = DATA/'tourismdatasetbd_bn.json'
    geo = pd.DataFrame(json.loads(geo_path.read_text(encoding='utf-8'))) if geo_path.exists() else pd.DataFrame()
    chatbot_path = DATA/'chatbot_cleaned.csv'
    chatbot = pd.read_csv(chatbot_path).fillna('') if chatbot_path.exists() else pd.DataFrame()
    rows = []
    used_by_dest = {}
    geo_titles = geo.get('title', pd.Series(dtype=str)).astype(str).tolist()
    geo_title_norms = [norm(x) for x in geo_titles]
    exact_title_index = {}
    for i, title_key in enumerate(geo_title_norms):
        if title_key and title_key not in exact_title_index: exact_title_index[title_key] = i

    def add(d, name, category, lat, lon, rating, count, source, match_method, source_location='', force=False):
        name = str(name or '').strip()
        if not name: return
        did = str(d['destination_id'])
        key = norm(name)
        seen = used_by_dest.setdefault(did, set())
        if key in seen and not force: return
        seen.add(key)
        est_dur = duration_for(category, name)
        est_cost = cost_for(category, name)
        rows.append({
            'place_id': f"{did}-P{len(rows)+1:05d}", 'destination_id': did,
            'destination_name': d['destination_name'], 'destination_district': d['district'],
            'place_name': name, 'category': str(category or 'Tourist attraction'),
            'latitude': lat, 'longitude': lon,
            'rating': rating, 'rating_count': count,
            'estimated_visit_hours': est_dur, 'estimated_visit_cost_bdt': est_cost,
            'source_type': source, 'location_match_method': match_method,
            'source_location': source_location,
            'estimate_note': 'Visit duration and cost are planning estimates; confirm current access, hours and fees.'
        })

    # 1) Preserve the named highlights already curated for each destination. Use geodata
    # coordinates only when a reasonably strong title match exists; otherwise use the
    # destination coordinates and clearly flag that as approximate.
    for _, d in dest.iterrows():
        listed = [x.strip() for x in str(d.get('attractions','')).split(';') if x.strip()]
        for name in listed:
            lat, lon = d.get('latitude',''), d.get('longitude','')
            category = d.get('destination_type','Tourist attraction')
            rating, count = '', 0
            method = 'destination attraction list; destination coordinates approximate'
            if not geo.empty:
                q = norm(name)
                if q in exact_title_index:
                    i = exact_title_index[q]; g = geo.iloc[i]
                    lat, lon = g.get('latitude', lat), g.get('longitude', lon)
                    category = g.get('category', category) or category
                    rating, count = g.get('rating',''), g.get('ratingCount',0)
                    method = 'destination list matched exact tourism dataset title'
            add(d, name, category, lat, lon, rating, count, 'Destination attractions list', method, d.get('district',''))

    # 2) Attach geolocated tourism records to the nearest destination centre, but only
    # within 20 km. Vectorized distance calculation keeps this step quick.
    if not geo.empty:
        geo['latitude_num'] = pd.to_numeric(geo.get('latitude'), errors='coerce')
        geo['longitude_num'] = pd.to_numeric(geo.get('longitude'), errors='coerce')
        geo['rating_num'] = pd.to_numeric(geo.get('rating'), errors='coerce')
        geo['count_num'] = pd.to_numeric(geo.get('ratingCount'), errors='coerce').fillna(0)
        geo['title_key'] = geo.get('title', pd.Series('', index=geo.index)).astype(str).map(norm)
        dest_lat = pd.to_numeric(dest['latitude'], errors='coerce').to_numpy(dtype=float)
        dest_lon = pd.to_numeric(dest['longitude'], errors='coerce').to_numpy(dtype=float)
        dest_rows = [r for _, r in dest.iterrows()]
        for _, g in geo.iterrows():
            title = str(g.get('title','')).strip()
            if not title or norm(title) in ('bangladesh','unknown','nan'): continue
            cat = str(g.get('category','')).strip()
            if not cat or norm(cat) in ('bangladesh','unknown'): cat = 'Tourist attraction'
            try:
                lat, lon = float(g.get('latitude_num')), float(g.get('longitude_num'))
                if not (math.isfinite(lat) and math.isfinite(lon)): continue
            except (ValueError, TypeError): continue
            # Haversine distances from one point to all 78 destination centres.
            p1 = math.radians(lat); p2 = np.radians(dest_lat)
            dp = p2 - p1; dl = np.radians(dest_lon - lon)
            aa = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
            distances = 6371 * 2 * np.arcsin(np.minimum(1, np.sqrt(aa)))
            idx = int(np.nanargmin(distances)); best_dist = float(distances[idx])
            if best_dist > 5: continue
            best_d = dest_rows[idx]
            add(best_d, title, cat, lat, lon, g.get('rating_num',''), int(g.get('count_num',0)),
                'Bangladesh tourism geodata', f'nearest destination centre; approx {best_dist:.1f} km', g.get('upazila_name',''))

    # 3) Use the tourist chatbot dataset as a supplemental district-level source. These
    # entries are marked approximate because the chatbot data does not provide coordinates.
    if not chatbot.empty and 'district' in chatbot:
        seen_chat = set()
        for _, d in dest.iterrows():
            same = chatbot[chatbot['district'].astype(str).map(norm) == norm(d.get('district',''))]
            count_added = 0
            for _, c in same.iterrows():
                for col in ['tourist_spots','similar_spots']:
                    raw = str(c.get(col,'')).strip()
                    if not raw or norm(raw) in ('nan','none'): continue
                    for name in re.split(r'\s*(?:;|\||\n|&| and )\s*', raw):
                        name = name.strip(' ,.-')
                        key = (str(d['destination_id']), norm(name))
                        if not name or key in seen_chat: continue
                        if norm(name) in used_by_dest.get(str(d['destination_id']), set()): continue
                        # Keep chatbot suggestions only when they overlap with this
                        # destination's curated highlights/name; district-only guesses
                        # otherwise risk suggesting a distant place as a local stop.
                        known = norm(str(d.get('destination_name','')) + ' ' + str(d.get('attractions','')))
                        nt = set(norm(name).split()); kt = set(known.split())
                        overlap = len(nt & kt) / max(1, len(nt))
                        if overlap < 0.5: continue
                        add(d, name, 'District-level tourist suggestion', d.get('latitude',''), d.get('longitude',''), '', 0,
                            'Tourist chatbot dataset', 'district-level suggestion; location approximate', d.get('district',''))
                        seen_chat.add(key); count_added += 1
                        if count_added >= 3: break
                    if count_added >= 5: break
                if count_added >= 5: break

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=['place_id','destination_id','destination_name','destination_district','place_name','category','latitude','longitude','rating','rating_count','estimated_visit_hours','estimated_visit_cost_bdt','source_type','location_match_method','source_location','estimate_note'])
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(json.dumps({'output': str(OUT), 'attraction_rows': len(out), 'destinations_with_attractions': int(out['destination_id'].nunique()) if len(out) else 0,
                      'source_counts': out['source_type'].value_counts().to_dict() if len(out) else {}}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
