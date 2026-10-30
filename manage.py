"""TravelBD admin management. Run in a terminal on the server; never exposed on the web.

Everyone signs in on the same /login page. An account is an admin only if its role
is 'admin', and that can only be set here (the public /register page always makes
normal users).

    python manage.py create-admin [email]   create a new admin (or promote an existing user)
    python manage.py promote <email>        make an existing user an admin
    python manage.py demote <email>         make an admin a normal user again
    python manage.py reset-password <email>
    python manage.py list-admins
    python manage.py reseed                 back up the database, then reload records from the CSV files on next start
                                            (use after re-running train_model.py; discards admin edits to destinations/attractions/hotels)
"""
import sys, re, getpass
from werkzeug.security import generate_password_hash
import shutil, datetime
from app import db_connect, init_admin_db, DB_PATH

EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")

def ask_password():
    while True:
        pw = getpass.getpass('Password (min 10 chars): ')
        if len(pw) < 10: print('Too short.'); continue
        if pw != getpass.getpass('Confirm password: '): print('Passwords do not match.'); continue
        return pw

def main(argv):
    init_admin_db()
    cmd = argv[1] if len(argv) > 1 else ''
    arg = (argv[2] if len(argv) > 2 else '').strip().lower()
    if cmd == 'create-admin':
        email = arg or input('Admin email: ').strip().lower()
        if not EMAIL.fullmatch(email): sys.exit('Enter a valid email address.')
        with db_connect() as con:
            if con.execute('SELECT 1 FROM users WHERE email=?', (email,)).fetchone():
                con.execute("UPDATE users SET role='admin' WHERE email=?", (email,))
                print(f'{email} already existed and is now an admin (password unchanged).')
            else:
                name = input('Full name: ').strip() or 'Admin'
                con.execute("INSERT INTO users(full_name,email,password_hash,role) VALUES(?,?,?,'admin')",
                            (name, email, generate_password_hash(ask_password())))
                print(f'Admin {email} created. Sign in at /login.')
    elif cmd in ('promote', 'demote'):
        if not arg: sys.exit(f'Usage: python manage.py {cmd} <email>')
        role = 'admin' if cmd == 'promote' else 'user'
        with db_connect() as con:
            if role == 'user' and con.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0] <= 1:
                sys.exit('Refusing to demote the last admin.')
            cur = con.execute('UPDATE users SET role=? WHERE email=?', (role, arg))
        print('Done.' if cur.rowcount else 'No such user.')
    elif cmd == 'reset-password':
        if not arg: sys.exit('Usage: python manage.py reset-password <email>')
        with db_connect() as con:
            cur = con.execute('UPDATE users SET password_hash=? WHERE email=?', (generate_password_hash(ask_password()), arg))
        print('Password updated.' if cur.rowcount else 'No such user.')
    elif cmd == 'reseed':
        backup = DB_PATH.with_name(f"travelbd_admin.backup-{datetime.datetime.now():%Y%m%d-%H%M%S}.sqlite3")
        shutil.copy2(DB_PATH, backup)
        with db_connect() as con: con.execute('DELETE FROM records')
        print(f'Backup saved to {backup}. Destination/attraction/hotel records will be reloaded from data/*.csv on the next start.')
    elif cmd == 'list-admins':
        with db_connect() as con:
            for r in con.execute("SELECT email, full_name FROM users WHERE role='admin' ORDER BY id"): print(r['email'], '-', r['full_name'])
    else:
        print(__doc__)

if __name__ == '__main__': main(sys.argv)
