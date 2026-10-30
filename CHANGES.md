# Cumulative patch (destination hierarchy + data quality)

Copy these files over your project (same paths). `data/`, `instance/` and `.venv/` are not overwritten.

1. python enrich_attractions.py --dry-run   (check), then python enrich_attractions.py   (adds ~83 real rated places to thin destinations)
2. python import_hotels.py --report          (see which districts still have no real hotel)
3. Fill data/real_hotels_template.csv with REAL hotels (with source_url), then: python import_hotels.py data/real_hotels_template.csv
4. python app.py

Do NOT run train_model.py (it can overwrite admin edits).

Changed: app.py, itinerary.py, evaluate.py, README.md, static/style.css, templates/{index,results,detail,itinerary_form,itinerary}.html
New: enrich_attractions.py, import_hotels.py, data/real_hotels_template.csv
