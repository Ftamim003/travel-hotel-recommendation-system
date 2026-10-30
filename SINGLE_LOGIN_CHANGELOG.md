# Single login update
- One sign-in page (/login) for users and admins; accounts have a role (user/admin).
- Admins are created/promoted only from the terminal: python manage.py create-admin | promote | demote | reset-password | list-admins
- Public /register always creates normal users. /admin pages return 403 for non-admins; Admin link shows only for admins.
- Old separate admin table is migrated automatically on first start. Login locks for 5 minutes after 5 failed attempts.
- run.bat starts the site with waitress (debug off). Set TRAVELBD_DEBUG=1 for Flask debug mode.
- Registration now requires a mobile number and email OTP verification (see README).
- Added forgot-password flow with email OTP (/forgot-password, /reset-password).
