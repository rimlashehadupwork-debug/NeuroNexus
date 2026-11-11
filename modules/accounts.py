"""Local demo accounts and derived-session metadata. Not production authentication."""
import sqlite3, hashlib, secrets
from pathlib import Path
DB_PATH = Path(__file__).resolve().parent.parent / 'neuronexus_demo.sqlite3'

def connect():
    c=sqlite3.connect(DB_PATH); c.row_factory=sqlite3.Row; return c

def init_db():
    with connect() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, salt TEXT NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'individual', caregiver_email TEXT DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, summary TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id))''')
        if not c.execute("SELECT id FROM users WHERE email='admin@neuronexus.local'").fetchone():
            salt=secrets.token_hex(16); digest=hashlib.pbkdf2_hmac('sha256',b'ChangeMe123!',salt.encode(),180000).hex()
            c.execute('INSERT INTO users(name,email,salt,password_hash,role) VALUES(?,?,?,?,?)',('NeuroNexus Admin','admin@neuronexus.local',salt,digest,'admin'))

def register(name,email,password,role='individual',caregiver_email=''):
    salt=secrets.token_hex(16); digest=hashlib.pbkdf2_hmac('sha256',password.encode(),salt.encode(),180000).hex()
    with connect() as c:
        c.execute('INSERT INTO users(name,email,salt,password_hash,role,caregiver_email) VALUES(?,?,?,?,?,?)',(name.strip(),email.strip().lower(),salt,digest,role,caregiver_email.strip().lower()))

def authenticate(email,password):
    with connect() as c: row=c.execute('SELECT * FROM users WHERE email=?',(email.strip().lower(),)).fetchone()
    if not row: return None
    digest=hashlib.pbkdf2_hmac('sha256',password.encode(),row['salt'].encode(),180000).hex()
    return dict(row) if secrets.compare_digest(digest,row['password_hash']) else None

def update_profile(user_id,name,caregiver_email):
    with connect() as c: c.execute('UPDATE users SET name=?, caregiver_email=? WHERE id=?',(name.strip(),caregiver_email.strip().lower(),user_id))

def save_session(user_id,summary):
    import json
    with connect() as c: c.execute('INSERT INTO sessions(user_id,summary) VALUES(?,?)',(user_id,json.dumps(summary,allow_nan=False)))

def user_sessions(user_id):
    with connect() as c: return [dict(x) for x in c.execute('SELECT * FROM sessions WHERE user_id=? ORDER BY created_at DESC',(user_id,)).fetchall()]

def delete_user_sessions(user_id):
    with connect() as c:
        cursor = c.execute('DELETE FROM sessions WHERE user_id=?', (user_id,))
        return cursor.rowcount

def caregiver_sessions(email):
    with connect() as c:
        return [dict(x) for x in c.execute('SELECT s.*,u.name AS participant_name,u.email AS participant_email FROM sessions s JOIN users u ON s.user_id=u.id WHERE u.caregiver_email=? ORDER BY s.created_at DESC',(email.lower(),)).fetchall()]

def admin_overview():
    with connect() as c:
        return {'users':c.execute('SELECT COUNT(*) FROM users').fetchone()[0], 'individuals':c.execute("SELECT COUNT(*) FROM users WHERE role='individual'").fetchone()[0], 'sessions':c.execute('SELECT COUNT(*) FROM sessions').fetchone()[0], 'people': [dict(x) for x in c.execute('SELECT id,name,email,role,created_at FROM users ORDER BY created_at DESC').fetchall()], 'recent': [dict(x) for x in c.execute('SELECT s.id,s.created_at,u.name,u.email FROM sessions s JOIN users u ON u.id=s.user_id ORDER BY s.created_at DESC LIMIT 30').fetchall()]}
