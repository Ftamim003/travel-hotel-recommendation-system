# Fixes from the code review

**Bugs**
- Itinerary page returned HTTP 500 for every plan (`day.items` resolved to `dict.items`). Now `day['items']`.
- Season matching: "Nov-Mar" matched nothing; "Oct-Mar" and "Jul-Oct (monsoon)" were treated as "shoulder". Seasons are now compared by month overlap.
- Origin no longer gives a ranking bonus to destinations in the traveller's own district (it still filters transport).

**Recommender**
- Budget score no longer rewards "cheapest"; affordable = full score, over budget decays to 0 at +25%.
- Content similarity is scaled so the best match is 1.0, and it now carries 50% of the score. Review-derived signals (mostly synthetic data) dropped from 26% to 15%.
- Hotel budget score uses the same affordability rule (room price split across up to 2 guests vs an assumed 40% lodging share).
- Profile interests prefill the search form for signed-in users.

**Itinerary**
- Shows approximate distance/time between stops; skips hops over 60 km; mild penalty for repeating the same kind of place; drops junk place names; notes when fewer days could be filled than requested.

**Architecture**
- Admin edits no longer overwrite the raw `data/1_destinations.csv`.
- Runtime data and the model are built first and swapped in together under a lock; scoring uses one consistent snapshot (tested with concurrent readers).
- The model pickle is written atomically and rebuilt automatically if missing or incompatible with the installed scikit-learn.
- `python manage.py reseed` (with automatic backup) and a warning from `train_model.py` address the two-sources-of-truth problem.

**Security**
- Real Gmail password, secret key, user database, `.venv` and caches removed from the package; `.gitignore` added.
- CSRF protection added to the search and itinerary forms.
- Session cookie: HttpOnly, SameSite=Lax, optional Secure (`TRAVELBD_HTTPS=1`).
- OTP emails rate-limited per IP; failed-login table bounded; backslash open-redirect closed.

**Not changed**
- The synthetic reviews/hotels and placeholder attraction costs are data, not code. They are still labelled and still contribute to ranking.
- `/register` still says when an email or phone is already registered (usability tradeoff; now rate-limited).
- Phone numbers are collected but not verified.
