@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
echo.
echo TravelBD is starting at http://127.0.0.1:5000  (admin: /admin/login)
echo Press Ctrl+C to stop.
start "" http://127.0.0.1:5000
waitress-serve --listen=127.0.0.1:5000 app:app
