"""Prepare TravelBD datasets, integrate official BPC hotel listings and summarize BanTABSA.
Run: python train_model.py
"""
from pathlib import Path
import json, re, ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

def clean_text(x):
    if pd.isna(x) or x is None: return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def to_num(df, cols):
    for c in cols:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

def city_key(city):
    s=clean_text(city).lower().replace("’", "'")
    aliases={"kuakata":"patuakhali", "moulvibazar":"moulvibazar", "cox's bazar":"cox's bazar", "chittagong":"chattogram"}
    return aliases.get(s,s)

def main():
    # Build the itinerary attraction table from destination highlights, geolocated tourism
    # records and district-level chatbot suggestions before the app loads.
    # Keep the manually reviewed attraction file (it carries verification_* columns); rebuild only if it is missing or still the raw generated one.
    _att = DATA/'itinerary_attractions.csv'
    if not (_att.exists() and 'verification_status' in pd.read_csv(_att, nrows=1).columns):
        from build_itinerary_dataset import main as build_itinerary_dataset
        build_itinerary_dataset()

    destinations = pd.read_csv(DATA/'1_destinations.csv').fillna('')
    hotels = pd.read_csv(DATA/'2_hotels.csv').fillna('')
    bpc = pd.read_csv(DATA/'2b_real_bpc_hotels.csv').fillna('')
    reviews = pd.read_csv(DATA/'3_users_reviews.csv').fillna('')
    transport = pd.read_csv(DATA/'4_transport.csv').fillna('')

    for df, cols in [
        (destinations, ['destination_name','district','division','destination_type','attractions','best_season','activities','accessibility_level','accessibility_modes','accessibility_notes']),
        (hotels, ['hotel_name','district','facilities','room_types','address','data_source']),
        (bpc, ['hotel_name','district','facilities','room_types','address','booking_link','data_source']),
        (reviews, ['user_id','travel_style','preferred_facilities','home_origin','selected_destination_id','hotel_id','feedback','hotel_data_source']),
        (transport, ['origin','destination','destination_district','transport_mode','travel_season'])]:
        for c in cols:
            if c in df: df[c] = df[c].map(clean_text)

    to_num(destinations, ['estimated_daily_budget_bdt','latitude','longitude'])
    to_num(hotels, ['price_per_night_bdt','hotel_rating','star_class','latitude','longitude'])
    to_num(bpc, ['price_per_night_bdt','latitude','longitude'])
    to_num(reviews, ['nights','user_budget_bdt_per_person_per_day','group_size','price_paid_per_night_per_room_bdt','facilities_matched','total_estimated_trip_cost_bdt','rating','rooms_booked'])
    to_num(transport, ['travel_time_hrs','fare_bdt','distance_km'])

    # Ensure all BPC rows appear in the same table queried by the live hotel recommender.
    # Existing synthetic rows remain labelled; BPC rows retain their source and original IDs.
    hotels['source_type'] = hotels.get('data_source', pd.Series('', index=hotels.index)).map(
        lambda x: 'Official BPC listing' if 'BPC official' in str(x).lower() else 'Synthetic placeholder')
    bpc['source_type'] = 'Official BPC listing'
    bpc['star_class'] = ''
    bpc['hotel_rating'] = ''
    expected = list(hotels.columns)
    for c in expected:
        if c not in bpc.columns: bpc[c] = ''
    for c in bpc.columns:
        if c not in hotels.columns: hotels[c] = ''
    bpc_rows = bpc[~bpc['hotel_id'].astype(str).isin(set(hotels['hotel_id'].astype(str)))].copy()
    hotels = pd.concat([hotels, bpc_rows[hotels.columns]], ignore_index=True)
    hotels['source_type'] = hotels.apply(lambda r: 'Official BPC listing' if str(r.get('data_source','')).lower().find('bpc official')>=0 or str(r.get('hotel_id','')).startswith('BPC') else str(r.get('source_type') or 'Synthetic placeholder'), axis=1)

    # BanTABSA is aggregated by location only: hotel IDs are anonymized and cannot safely
    # be joined to named BPC listings. Parse aspect triplets for transparent city sentiment.
    bantabsa_path = DATA/'BanTABSA.csv'
    city_summary = pd.DataFrame(columns=['location_city','city_key','review_count','mean_rating','positive_triplets','neutral_triplets','negative_triplets','source_type'])
    hotel_summary = pd.DataFrame(columns=['hotel_id','location_city','review_count','mean_rating','source_type'])
    if bantabsa_path.exists():
        bt = pd.read_csv(bantabsa_path).fillna('')
        bt['rating_num'] = pd.to_numeric(bt.get('rating'), errors='coerce')
        def count_sentiments(raw):
            counts={'positive':0,'neutral':0,'negative':0}
            try:
                items=json.loads(raw) if isinstance(raw,str) else []
                for item in items:
                    if len(item)>=3 and str(item[2]).lower() in counts: counts[str(item[2]).lower()]+=1
            except Exception:
                pass
            return pd.Series(counts)
        if 'triplets' in bt:
            bt[['positive_triplets','neutral_triplets','negative_triplets']] = bt['triplets'].apply(count_sentiments)[['positive','neutral','negative']]
        else:
            bt['positive_triplets']=bt['neutral_triplets']=bt['negative_triplets']=0
        bt['city_key'] = bt['location_city'].map(city_key)
        city_summary = bt.groupby(['location_city','city_key'], dropna=False).agg(
            review_count=('review_id','count'), mean_rating=('rating_num','mean'),
            positive_triplets=('positive_triplets','sum'), neutral_triplets=('neutral_triplets','sum'),
            negative_triplets=('negative_triplets','sum')).reset_index()
        city_summary['source_type']='BanTABSA Bengali hotel review dataset; city-level aggregate, not hotel-specific match'
        hotel_summary = bt.groupby(['hotel_id','location_city'], dropna=False).agg(
            review_count=('review_id','count'), mean_rating=('rating_num','mean')).reset_index()
        hotel_summary['source_type']='BanTABSA anonymized hotel ID; do not join to named hotels without a verified mapping'
        city_summary.to_csv(DATA/'bantabsa_city_summary.csv', index=False, encoding='utf-8-sig')
        hotel_summary.to_csv(DATA/'bantabsa_anonymized_hotel_summary.csv', index=False, encoding='utf-8-sig')
    else:
        city_summary.to_csv(DATA/'bantabsa_city_summary.csv', index=False, encoding='utf-8-sig')
        hotel_summary.to_csv(DATA/'bantabsa_anonymized_hotel_summary.csv', index=False, encoding='utf-8-sig')

    destinations['content_text'] = destinations[['destination_name','district','division','destination_type','attractions','activities','best_season','accessibility_level','accessibility_modes','accessibility_notes']].astype(str).agg(' '.join, axis=1)
    reviews['rating'] = pd.to_numeric(reviews['rating'], errors='coerce')
    # Audited reviews: synthetic rows carry a low weight (0.05), unverified official rows 0. Weighted means; counts are effective (weighted) counts.
    _wc = 'production_training_weight' if 'production_training_weight' in reviews else 'recommended_training_weight'
    if _wc in reviews:
        reviews['_w'] = pd.to_numeric(reviews[_wc], errors='coerce').fillna(1.0)
        reviews['rating'] = reviews['rating'].where(reviews['_w'] > 0)
    else: reviews['_w'] = 1.0
    dest_stats = reviews.groupby('selected_destination_id').agg(review_mean_rating=('rating','mean'), review_count=('review_id','count'), median_user_daily_budget=('user_budget_bdt_per_person_per_day','median'), median_trip_cost=('total_estimated_trip_cost_bdt','median')).reset_index().rename(columns={'selected_destination_id':'destination_id'})
    hotel_stats = reviews.groupby('hotel_id').agg(review_mean_rating=('rating','mean'), review_count=('review_id','count'), median_paid_nightly_price=('price_paid_per_night_per_room_bdt','median'), facility_match_rate=('facilities_matched','mean')).reset_index()
    def _wstats(key):   # weighted mean rating + effective review count (sum of weights of rated rows)
        g = reviews.dropna(subset=['rating']).assign(_rw=lambda x: x['rating'] * x['_w']).groupby(key)
        return pd.DataFrame({'review_mean_w': g['_rw'].sum() / g['_w'].sum().replace(0, float('nan')), 'review_count_w': g['_w'].sum()})
    ws = _wstats('selected_destination_id'); k = dest_stats['destination_id']
    dest_stats['review_mean_rating'] = k.map(ws['review_mean_w']); dest_stats['review_count'] = k.map(ws['review_count_w']).fillna(0).round(2)
    ws = _wstats('hotel_id'); k = hotel_stats['hotel_id']
    hotel_stats['review_mean_rating'] = k.map(ws['review_mean_w']); hotel_stats['review_count'] = k.map(ws['review_count_w']).fillna(0).round(2)
    dest_stats.to_csv(DATA/'destination_review_summary.csv', index=False, encoding='utf-8-sig')
    hotel_stats.to_csv(DATA/'hotel_review_summary.csv', index=False, encoding='utf-8-sig')
    destinations.to_csv(DATA/'destinations_cleaned.csv', index=False, encoding='utf-8-sig')
    hotels.to_csv(DATA/'hotels_cleaned.csv', index=False, encoding='utf-8-sig')
    bpc.to_csv(DATA/'bpc_hotels_cleaned.csv', index=False, encoding='utf-8-sig')
    reviews.to_csv(DATA/'reviews_cleaned.csv', index=False, encoding='utf-8-sig')
    transport.to_csv(DATA/'transport_cleaned.csv', index=False, encoding='utf-8-sig')

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=12000, sublinear_tf=True, strip_accents='unicode')
    matrix = vectorizer.fit_transform(destinations['content_text'].fillna(''))
    import pickle
    with open(ARTIFACTS/'destination_content_model.pkl','wb') as f:
        pickle.dump({'vectorizer':vectorizer,'matrix':matrix,'destination_ids':destinations['destination_id'].tolist()},f)

    report = {
      'destinations': int(len(destinations)), 'districts': int(destinations['district'].nunique()),
      'hotel_rows_available_to_recommender': int(len(hotels)),
      'hotel_rows_marked_synthetic': int((hotels['source_type']=='Synthetic placeholder').sum()),
      'hotel_rows_marked_official_bpc': int((hotels['source_type']=='Official BPC listing').sum()),
      'separate_official_bpc_listings': int(len(bpc)), 'review_rows': int(len(reviews)),
      'unique_review_users': int(reviews['user_id'].nunique()), 'transport_rows': int(len(transport)),
      'transport_modes': int(transport['transport_mode'].nunique()), 'tfidf_features': int(len(vectorizer.get_feature_names_out())),
      'itinerary_attraction_rows': int(len(pd.read_csv(DATA/'itinerary_attractions.csv'))) if (DATA/'itinerary_attractions.csv').exists() else 0,
      'bantabsa_review_rows': int(sum(city_summary['review_count'])) if not city_summary.empty else 0,
      'bantabsa_locations': int(city_summary['location_city'].nunique()) if not city_summary.empty else 0,
      'note': 'BanTABSA ratings and sentiment are aggregated by location because hotel IDs are anonymized and not matched to named listings. Existing review/trip rows are mostly labelled synthetic placeholders. Destination budgets and transport prices are estimates, not guaranteed current quotes.'
    }
    (ARTIFACTS/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
def _warn_if_admin_db_overrides():
    import sqlite3
    db = ROOT/'instance'/'travelbd_admin.sqlite3'
    if not db.exists(): return
    try:
        n = sqlite3.connect(db).execute('SELECT COUNT(*) FROM records').fetchone()[0]
    except Exception:
        return
    if n:
        print(f"\nWARNING: instance/travelbd_admin.sqlite3 already holds {n} admin records. On the next app start they OVERWRITE the CSVs just generated.\n"
              "To use the freshly generated data instead, run:  python manage.py reseed   (makes a backup first).")

if __name__ == '__main__':
    main(); _warn_if_admin_db_overrides()
