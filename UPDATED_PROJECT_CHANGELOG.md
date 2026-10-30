# TravelBD itinerary update

## Added
- Destination-specific **Plan my trip** button on destination details.
- Itinerary preferences: number of days, attraction budget per person per day, group size, travel style, interests, and planning pace.
- Day-wise schedule with suggested morning/afternoon/evening slots, estimated visit duration, estimated attraction expense, source labels, and approximate-location notes.
- `build_itinerary_dataset.py`, which builds `data/itinerary_attractions.csv` from curated destination highlights, supplied Bangladesh tourism geodata, and relevant district-level chatbot suggestions.
- `itinerary.py`, which ranks available stops by travel-style/interest relevance, destination-highlight priority, available rating signals, and proximity, then spreads stops across the requested days.
- Updated Model & Data page, README, and demo guide.

## Current generated dataset
- 1,573 attraction rows across 78 destinations.
- 319 rows from the destination highlight lists.
- 1,244 supplemental geolocated tourism records.
- 10 matching district-level suggestions from the tourist chatbot dataset.

## Important limitations
- Visit durations and attraction costs are heuristic demo estimates, not verified entry fees or visit times.
- Geolocated tourism records are associated with the nearest destination centre within 5 km; this association can still be approximate.
- Destination-listed attractions without exact coordinate matches use destination-centre coordinates marked as approximate.
- The planner does not check opening hours, traffic, live conditions, local transit schedules, weather, or bookings.
- The attraction-cost estimate excludes meals, hotel expenses, and local transport.
