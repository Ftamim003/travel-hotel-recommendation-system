"""Constraint-based day-wise itinerary planner for TravelBD demo."""
from pathlib import Path
import math, re
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ATTRACTIONS = pd.read_csv(DATA/'itinerary_attractions.csv').fillna('') if (DATA/'itinerary_attractions.csv').exists() else pd.DataFrame()
for _col in ['latitude','longitude','rating','rating_count','estimated_visit_hours','estimated_visit_cost_bdt']:
    if not ATTRACTIONS.empty and _col in ATTRACTIONS:
        ATTRACTIONS[_col] = pd.to_numeric(ATTRACTIONS[_col], errors='coerce')

STYLE_TERMS = {
    'adventure': 'hiking trekking hill mountain outdoor trail adventure waterfall',
    'nature': 'nature forest park garden waterfall river lake beach island wildlife scenic',
    'cultural': 'culture heritage historical museum temple mosque landmark archaeological market',
    'family': 'family park garden zoo museum beach playground amusement',
    'relaxation': 'beach garden river lake scenic peaceful park',
    'spiritual': 'temple mosque shrine religious worship monastery',
    'photography': 'scenic viewpoint river bridge mountain beach hill landscape',
    'food': 'market food local cuisine bazaar',
}

# Words that identify a "kind" of place; used to avoid planning five near-identical stops.
KIND_WORDS={'beach','fort','mosque','temple','park','museum','garden','lake','river','island','bridge','hill','falls','waterfall','zoo','market','tomb','palace','church','vihara'}
ROAD_FACTOR=1.3        # straight-line distance -> rough road distance
AVG_KMH=25.0           # rough average speed on Bangladeshi roads
MAX_HOP_KM=60.0        # do not chain stops further apart than this within a day

def kinds(name):
    return set(re.findall(r'[a-z]+', str(name or '').lower())) & KIND_WORDS

ACTIVITY_WORDS={'tasting','cycling','hiking','trekking','photography','walks','walk','boating','swimming','camping','fishing','shopping','sightseeing','rafting','kayaking','birdwatching','cruise','watching'}
EMOJI=re.compile('[\U0001F300-\U0001FAFF\u2600-\u27BF]')

def plausible_name(name):
    """Keep real places only: drop junk ('Pc'/'Mim'), emoji-decorated personal listings and activities ('Tea tasting', 'Cycling')."""
    n=str(name or '').strip()
    if not (len(n)>3 or (len(n)==3 and n.isupper())): return False
    if EMOJI.search(n): return False
    tokens=re.findall(r'[a-z]+', n.lower())
    if tokens and tokens[-1] in ACTIVITY_WORDS and len(tokens)<=3: return False
    return True

def norm(v):
    return re.sub(r'\s+', ' ', str(v or '')).strip().lower()

def num(v, default=0):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (ValueError, TypeError):
        return default

def haversine_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
        a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 6371 * 2 * math.asin(min(1, math.sqrt(a)))
    except (ValueError, TypeError):
        return 99999.0

def attraction_relevance(place, style, interests):
    text = norm(' '.join(str(place.get(k,'')) for k in ['place_name','category','source_location']))
    style = norm(style)
    terms = STYLE_TERMS.get(style, '')
    score = 0.0
    if terms:
        tokens = set(terms.split())
        score += min(1.0, len(tokens & set(text.split())) / 3.0)
    interest_tokens = set(re.findall(r'[a-z0-9]+', norm(interests)))
    if interest_tokens:
        score += min(1.0, len(interest_tokens & set(text.split())) / max(1, len(interest_tokens))) * 1.3
    if place.get('source_type') == 'Destination attractions list':
        score += 0.75
    rating = num(place.get('rating'), 0)
    reviews = num(place.get('rating_count'), 0)
    if rating > 0: score += (rating / 5) * 0.25
    if reviews > 0: score += min(0.4, math.log1p(reviews) / 25)   # well-known places (e.g. Lalbagh Fort) should not lose to obscure ones
    return score

def generate_itinerary(destination, days, daily_budget, group_size=2, travel_style='Any style', interests='', pace='Balanced'):
    if ATTRACTIONS.empty:
        return [], 'Attraction data is missing. Run python train_model.py first.'
    did = str(destination.get('destination_id', ''))
    zone_ids = {z for z in str(destination.get('zone_ids') or did).split(';') if z}   # a destination groups several areas
    candidates = ATTRACTIONS[ATTRACTIONS['destination_id'].astype(str).isin(zone_ids)].copy()
    if candidates.empty:
        return [], 'No attractions were found for this destination in the current dataset.'
    candidates = candidates.assign(_k=candidates['place_name'].map(norm)).drop_duplicates(subset=['_k']).drop(columns=['_k'])
    candidates = candidates[candidates['place_name'].map(plausible_name)]
    if candidates.empty:
        return [], 'No usable attractions were found for this destination in the current dataset.'
    candidates['relevance'] = candidates.apply(lambda r: attraction_relevance(r.to_dict(), travel_style, interests), axis=1)
    candidates['estimated_visit_hours'] = pd.to_numeric(candidates['estimated_visit_hours'], errors='coerce').fillna(1.5).clip(0.5, 4.0)
    candidates['estimated_visit_cost_bdt'] = pd.to_numeric(candidates['estimated_visit_cost_bdt'], errors='coerce').fillna(150).clip(0, 2000)
    candidates['rating_num'] = pd.to_numeric(candidates['rating'], errors='coerce').fillna(0)
    candidates['rating_count_num'] = pd.to_numeric(candidates['rating_count'], errors='coerce').fillna(0)
    candidates = candidates.sort_values(['relevance', 'rating_num', 'rating_count_num'], ascending=False)
    pool = [r.to_dict() for _, r in candidates.iterrows()]
    slots = [('Morning', '09:00', '12:00', 3.0), ('Afternoon', '13:00', '16:00', 3.0), ('Evening', '16:30', '19:00', 2.5)]
    if pace == 'Relaxed':
        slots = [('Morning', '09:30', '12:00', 2.5), ('Afternoon', '13:30', '16:00', 2.5)]
    elif pace == 'Packed':
        slots = [('Morning', '08:00', '11:30', 3.5), ('Afternoon', '12:30', '16:00', 3.5), ('Evening', '16:30', '19:30', 3.0)]
    itinerary = []
    used_kinds = set()
    max_days = max(1, min(int(days), 14))
    daily_budget = max(100, float(daily_budget))
    target_stops = min(len(pool), max_days * len(slots))
    stops_per_day = max(1, math.ceil(target_stops / max_days))
    for day_no in range(1, max_days + 1):
        slots_out, used_hours, day_cost, previous = [], 0.0, 0.0, None
        for slot_name, start, end, capacity in slots:
            if len(slots_out) >= stops_per_day:
                break
            possible = []
            for place in pool:
                duration = num(place.get('estimated_visit_hours'), 1.5)
                cost = num(place.get('estimated_visit_cost_bdt'), 150)
                if duration > capacity or used_hours + duration > sum(x[3] for x in slots) + 0.001:
                    continue
                if day_cost + cost > daily_budget and slots_out:
                    continue
                if previous is not None:
                    distance = haversine_km(previous.get('latitude'), previous.get('longitude'), place.get('latitude'), place.get('longitude'))
                    if distance < 99999 and distance > MAX_HOP_KM:
                        continue
                    proximity = 1 / (1 + distance / 5) if distance < 99999 else 0
                else:
                    proximity = 0
                repeat_penalty = min(0.4, 0.2 * len(kinds(place.get('place_name')) & used_kinds))
                score = num(place.get('relevance')) + proximity * 0.35 - repeat_penalty
                if place.get('source_type') == 'Destination attractions list': score += 0.15
                possible.append((score, place))
            if not possible:
                continue
            possible.sort(key=lambda x: x[0], reverse=True)
            selected = possible[0][1]
            pool.remove(selected)
            duration = num(selected.get('estimated_visit_hours'), 1.5)
            cost = num(selected.get('estimated_visit_cost_bdt'), 150)
            hop = haversine_km(previous.get('latitude'), previous.get('longitude'), selected.get('latitude'), selected.get('longitude')) if previous is not None else 99999.0
            travel_km = round(hop * ROAD_FACTOR, 1) if hop < 99999 else None
            travel_min = int(round(travel_km / AVG_KMH * 60 / 5.0) * 5) if travel_km is not None else None
            used_hours += duration
            day_cost += cost
            previous = selected
            used_kinds |= kinds(selected.get('place_name'))
            slots_out.append({'slot': slot_name, 'start': start, 'end': end, 'place': selected, 'duration': duration, 'cost': cost, 'travel_km': travel_km, 'travel_min': travel_min})
        if not slots_out and pool:
            selected = pool.pop(0)
            duration = num(selected.get('estimated_visit_hours'), 1.5)
            cost = num(selected.get('estimated_visit_cost_bdt'), 150)
            slots_out.append({'slot': 'Flexible visit', 'start': '09:00', 'end': '11:00', 'place': selected, 'duration': duration, 'cost': cost, 'travel_km': None, 'travel_min': None})
            day_cost = cost
            used_hours = duration
        if not slots_out:
            break
        itinerary.append({'day': day_no, 'items': slots_out, 'estimated_cost': day_cost, 'estimated_hours': used_hours})
        if not pool:
            break
    note = ('Costs and visit durations are planning estimates. Some coordinates are approximate or assigned to the nearest destination centre. '
            'Confirm attraction access, opening hours, entry fees and local travel before visiting. Local transport and meals are not included.')
    if len(itinerary) < max_days:
        note += f' Only {len(itinerary)} of the {max_days} requested day(s) could be filled with the attractions available for this destination.'
    return itinerary, note
