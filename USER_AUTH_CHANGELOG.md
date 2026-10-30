# User authentication update

- Added user registration and sign-in using email and password.
- Passwords are hashed with Werkzeug's password hashing utilities.
- Added sign-out, protected profile page, profile editing, and CSRF checks for account forms.
- User profiles are stored in SQLite (`instance/travelbd_admin.sqlite3`).
- Saves destination search preferences/results for signed-in users.
- Saves generated itinerary plans and allows their owner to reopen them from their profile.
- Adds account links to the public navigation and registered-user count to the admin dashboard.
- Existing admin authentication and destination/attraction/hotel management are retained.

Prototype limitations: no email verification, password reset, or account deletion interface yet. Collect only the user information needed for the demo and keep the database private.
