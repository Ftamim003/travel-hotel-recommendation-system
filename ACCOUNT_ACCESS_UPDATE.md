# TravelBD account access update

Changes in this update:

- Public home page now explains the service and offers Sign up / Sign in before showing the personalized recommendation form.
- Personalized destination recommendation requires an account; destination browsing and Model & Data remain public.
- Day-wise itinerary generation requires sign-in. Signed-in users can save searches and itineraries.
- Admin dashboard now tracks registered users, unique logins today, login events today, active users over the last 7 days, and accounts that have never logged in.
- Admin User Management page lists accounts and last-login timestamps.
- Per-user Menu Rights page allows an admin to toggle Explore recommendations, itinerary planner, destination details, Model & Data, and profile. Menu visibility is backed by route checks, not just hidden links.
- SQLite schema migrates automatically to add login tracking and per-user permission tables.

To run: follow README.md, install requirements, then run `python app.py`. Create an admin from the terminal with `python manage.py create-admin`. The project archive intentionally omits the virtual environment, local database, session secret and private email configuration; the application recreates local schema on startup.
