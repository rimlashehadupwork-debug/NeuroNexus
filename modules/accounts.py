"""Account, finalized-report, and local-draft persistence."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
from urllib.parse import parse_qs, urlsplit

from .demo_mode import build_synthetic_history


DB_PATH = Path(__file__).resolve().parent.parent / "neuronexus_demo.sqlite3"
ROLES = {"individual", "caregiver", "admin"}
DEMO_ACCOUNTS = {
    "individual@neuronexus.local": ("NeuroNexus Individual", "individual"),
    "caregiver@neuronexus.local": ("NeuroNexus Caregiver", "caregiver"),
}
DEMO_PASSWORD = "ChangeMe123!"
LEGACY_DEMO_ADMIN_EMAIL = "admin@neuronexus.local"


def _configured_value(name):
    value = os.getenv(name)
    if value:
        return value.strip()

    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return None

    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def get_database_url():
    """Return PostgreSQL configuration from the environment or Streamlit secrets."""
    database_url = _configured_value("DATABASE_URL")
    if database_url and not database_url.startswith(("postgresql://", "postgres://")):
        raise ValueError("DATABASE_URL must be a PostgreSQL connection string.")
    if database_url and database_url.startswith("postgres://"):
        return "postgresql://" + database_url[len("postgres://"):]
    return database_url


def storage_backend():
    return "postgresql" if get_database_url() else "sqlite"


def _postgres_sslmode(database_url):
    sslmodes = [
        mode.casefold()
        for mode in parse_qs(urlsplit(database_url).query).get("sslmode", [])
    ]
    if not sslmodes:
        return "require"
    if len(sslmodes) != 1 or sslmodes[0] not in {
        "require",
        "verify-ca",
        "verify-full",
    }:
        raise ValueError(
            "PostgreSQL connections require sslmode=require or stronger."
        )
    return sslmodes[0]


@contextmanager
def _sqlite_connection(path):
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


@contextmanager
def connect():
    """Connect to persistent profile/report storage, falling back to SQLite."""
    database_url = get_database_url()
    if not database_url:
        with _sqlite_connection(DB_PATH) as connection:
            yield connection
        return

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL is configured, but psycopg is unavailable. "
            "Install the project requirements."
        ) from exc

    sslmode = _postgres_sslmode(database_url)
    with psycopg.connect(
        database_url,
        row_factory=dict_row,
        sslmode=sslmode,
    ) as connection:
        yield connection


def _execute(connection, query, parameters=()):
    if isinstance(connection, sqlite3.Connection):
        return connection.execute(query, parameters)
    return connection.execute(query.replace("?", "%s"), parameters)


def _normalize_record(row):
    record = dict(row)
    created_at = record.get("created_at")
    if isinstance(created_at, datetime):
        record["created_at"] = created_at.isoformat()
    if "consent_flags" in record:
        flags = record["consent_flags"]
        record["consent_flags"] = json.loads(flags or "{}") if isinstance(flags, str) else flags
        if not isinstance(record["consent_flags"], dict):
            raise ValueError("Stored consent flags must be a JSON object.")
    return record


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _ensure_demo_accounts(connection):
    for email, (name, role) in DEMO_ACCOUNTS.items():
        if _execute(
            connection, "SELECT id FROM users WHERE email=?", (email,)
        ).fetchone():
            continue
        salt = secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", DEMO_PASSWORD.encode(), salt.encode(), 180000
        ).hex()
        _execute(
            connection,
            """INSERT INTO users(name,email,salt,password_hash,role)
               VALUES(?,?,?,?,?) ON CONFLICT(email) DO NOTHING""",
            (name, email, salt, digest, role),
        )


def _ensure_demo_history(connection):
    """Seed labelled synthetic history for the legacy individual/caregiver demos."""
    for email in (
        "individual@neuronexus.local",
        "caregiver@neuronexus.local",
    ):
        user = _execute(
            connection, "SELECT id FROM users WHERE email=?", (email,)
        ).fetchone()
        if not user:
            continue
        user_id = user["id"]
        existing_demo = _execute(
            connection,
            "SELECT id FROM sessions WHERE user_id=? AND summary LIKE ?",
            (user_id, '%"demo_history": true%'),
        ).fetchone()
        if existing_demo:
            continue

        for sample in build_synthetic_history():
            _execute(
                connection,
                "INSERT INTO sessions(user_id,created_at,summary) VALUES(?,?,?)",
                (
                    user_id,
                    sample["created_at"],
                    json.dumps(sample["summary"], allow_nan=False),
                ),
            )


def demo_account(email):
    """Ensure and return one of the public demo accounts for quick login."""
    email = email.strip().lower()
    if email not in DEMO_ACCOUNTS:
        raise ValueError("Only configured demo accounts can use quick login.")
    with connect() as connection:
        _ensure_demo_accounts(connection)
        _ensure_demo_history(connection)
        row = _execute(
            connection, "SELECT * FROM users WHERE email=?", (email,)
        ).fetchone()
    if not row:
        raise RuntimeError(f"Unable to initialize the demo account for {email}.")
    return _normalize_record(row)


def init_db():
    """Initialize persistent account/report tables and the local draft table."""
    with _sqlite_connection(DB_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS assessment_drafts(
                user_id INTEGER PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )

    database_url = get_database_url()
    with connect() as connection:
        if database_url:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS users(
                    id BIGSERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'individual',
                    caregiver_email TEXT NOT NULL DEFAULT '',
                    consent_flags TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS sessions(
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS wellness_plans(
                    user_id BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS consent_flags TEXT NOT NULL DEFAULT '{}'"
            )
            connection.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS caregiver_email TEXT NOT NULL DEFAULT ''"
            )
        else:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS users(
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'individual',
                    caregiver_email TEXT DEFAULT '',
                    consent_flags TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "consent_flags" not in columns:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN consent_flags TEXT NOT NULL DEFAULT '{}'"
                )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS sessions(
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS wellness_plans(
                    user_id INTEGER PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )"""
            )
        _ensure_demo_accounts(connection)
        _ensure_demo_history(connection)

    admin_email = _configured_value("NEURONEXUS_ADMIN_EMAIL")
    admin_password = _configured_value("NEURONEXUS_ADMIN_PASSWORD")
    if database_url and bool(admin_email) != bool(admin_password):
        raise ValueError(
            "Configure both NEURONEXUS_ADMIN_EMAIL and "
            "NEURONEXUS_ADMIN_PASSWORD to provision the initial cloud admin."
        )
    if database_url and admin_email and admin_password:
        with connect() as connection:
            if not _execute(
                connection,
                "SELECT id FROM users WHERE email=?",
                (admin_email.lower(),),
            ).fetchone():
                salt = secrets.token_hex(16)
                digest = hashlib.pbkdf2_hmac(
                    "sha256", admin_password.encode(), salt.encode(), 180000
                ).hex()
                _execute(
                    connection,
                    """INSERT INTO users(
                        name,email,salt,password_hash,role,consent_flags
                    ) VALUES(?,?,?,?,?,?)""",
                    (
                        "NeuroNexus Admin",
                        admin_email.lower(),
                        salt,
                        digest,
                        "admin",
                        json.dumps({"admin_provisioned": True}),
                    ),
                )


def register(name, email, password, role="individual", caregiver_email="", consent_flags=None):
    if role not in {"individual", "caregiver"}:
        raise ValueError("New accounts must use the individual or caregiver role.")
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 180000
    ).hex()
    flags = json.dumps(consent_flags or {}, allow_nan=False)
    with connect() as connection:
        _execute(
            connection,
            """INSERT INTO users(
                name,email,salt,password_hash,role,caregiver_email,consent_flags
            ) VALUES(?,?,?,?,?,?,?)""",
            (
                name.strip(),
                email.strip().lower(),
                salt,
                digest,
                role,
                caregiver_email.strip().lower(),
                flags,
            ),
        )


def authenticate(email, password):
    normalized_email = email.strip().lower()
    if (
        normalized_email == LEGACY_DEMO_ADMIN_EMAIL
        and secrets.compare_digest(password, DEMO_PASSWORD)
    ):
        return None
    with connect() as connection:
        row = _execute(
            connection, "SELECT * FROM users WHERE email=?", (normalized_email,)
        ).fetchone()
    if not row:
        return None
    record = _normalize_record(row)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), record["salt"].encode(), 180000
    ).hex()
    return record if secrets.compare_digest(digest, record["password_hash"]) else None


def update_profile(user_id, name, caregiver_email):
    with connect() as connection:
        cursor = _execute(
            connection,
            "UPDATE users SET name=?, caregiver_email=? WHERE id=?",
            (name.strip(), caregiver_email.strip().lower(), user_id),
        )
        if cursor.rowcount == 0:
            raise LookupError("The account to update does not exist.")


def update_consent_flag(user_id, flag, granted=True):
    if not flag or not isinstance(granted, bool):
        raise ValueError("A consent flag name and boolean value are required.")
    with connect() as connection:
        row = _execute(
            connection, "SELECT consent_flags FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if not row:
            raise LookupError("The account for this consent record does not exist.")
        stored_flags = row["consent_flags"]
        flags = json.loads(stored_flags or "{}") if isinstance(stored_flags, str) else stored_flags
        if not isinstance(flags, dict):
            raise ValueError("Stored consent flags must be a JSON object.")
        flags[flag] = granted
        _execute(
            connection,
            "UPDATE users SET consent_flags=? WHERE id=?",
            (json.dumps(flags, allow_nan=False), user_id),
        )
    return flags


def update_user_role(user_id, role):
    if role not in ROLES:
        raise ValueError(f"Role must be one of: {', '.join(sorted(ROLES))}.")
    with connect() as connection:
        cursor = _execute(
            connection, "UPDATE users SET role=? WHERE id=?", (role, user_id)
        )
        if cursor.rowcount == 0:
            raise LookupError("The account to update does not exist.")


def is_duplicate_email_error(error):
    if isinstance(error, sqlite3.IntegrityError):
        return "users.email" in str(error).casefold()
    try:
        import psycopg
    except ImportError:
        return False
    return isinstance(error, psycopg.errors.UniqueViolation)


def save_session(user_id, summary, created_at=None):
    serialized = json.dumps(summary, allow_nan=False, default=_json_default)
    with connect() as connection:
        if isinstance(connection, sqlite3.Connection):
            if created_at is None:
                cursor = _execute(
                    connection,
                    "INSERT INTO sessions(user_id,summary) VALUES(?,?)",
                    (user_id, serialized),
                )
            else:
                cursor = _execute(
                    connection,
                    "INSERT INTO sessions(user_id,created_at,summary) VALUES(?,?,?)",
                    (user_id, created_at, serialized),
                )
            return cursor.lastrowid
        if created_at is None:
            cursor = _execute(
                connection,
                "INSERT INTO sessions(user_id,summary) VALUES(?,?) RETURNING id",
                (user_id, serialized),
            )
        else:
            cursor = _execute(
                connection,
                "INSERT INTO sessions(user_id,created_at,summary) VALUES(?,?,?) RETURNING id",
                (user_id, created_at, serialized),
            )
        return cursor.fetchone()["id"]


def user_sessions(user_id):
    with connect() as connection:
        rows = _execute(
            connection,
            "SELECT * FROM sessions WHERE user_id=? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [_normalize_record(row) for row in rows]


def save_wellness_plan(user_id, payload):
    """Persist an explicitly saved wellness plan for one account."""
    if user_id is None:
        raise ValueError("A persisted account ID is required to save a wellness plan.")
    if not isinstance(payload, dict):
        raise ValueError("A wellness plan must be an object.")
    serialized = json.dumps(
        payload,
        allow_nan=False,
        default=_json_default,
        sort_keys=True,
    )
    with connect() as connection:
        _execute(
            connection,
            """INSERT INTO wellness_plans(user_id,payload) VALUES(?,?)
               ON CONFLICT(user_id) DO UPDATE SET
               payload=excluded.payload,updated_at=CURRENT_TIMESTAMP""",
            (user_id, serialized),
        )
    return True


def load_wellness_plan(user_id):
    """Load one account's saved wellness plan, if it exists."""
    if user_id is None:
        raise ValueError("A persisted account ID is required to load a wellness plan.")
    with connect() as connection:
        row = _execute(
            connection,
            "SELECT payload FROM wellness_plans WHERE user_id=?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    payload = json.loads(row["payload"])
    if not isinstance(payload, dict):
        raise ValueError("The saved wellness plan is not a JSON object.")
    return payload


def delete_wellness_plan(user_id):
    """Delete one account's saved wellness plan."""
    if user_id is None:
        raise ValueError("A persisted account ID is required to delete a wellness plan.")
    with connect() as connection:
        cursor = _execute(
            connection,
            "DELETE FROM wellness_plans WHERE user_id=?",
            (user_id,),
        )
    return cursor.rowcount > 0


def delete_user_sessions(user_id):
    with connect() as connection:
        cursor = _execute(
            connection, "DELETE FROM sessions WHERE user_id=?", (user_id,)
        )
        return cursor.rowcount


def caregiver_sessions(email):
    with connect() as connection:
        rows = _execute(
            connection,
            """SELECT s.*,u.name AS participant_name,u.email AS participant_email
               FROM sessions s JOIN users u ON s.user_id=u.id
               WHERE u.caregiver_email=? ORDER BY s.created_at DESC""",
            (email.strip().lower(),),
        ).fetchall()
    return [_normalize_record(row) for row in rows]


def admin_overview():
    with connect() as connection:
        users_count = _execute(connection, "SELECT COUNT(*) AS count FROM users").fetchone()
        individual_count = _execute(
            connection,
            "SELECT COUNT(*) AS count FROM users WHERE role='individual'",
        ).fetchone()
        sessions_count = _execute(connection, "SELECT COUNT(*) AS count FROM sessions").fetchone()
        people = _execute(
            connection,
            "SELECT id,name,email,role,created_at FROM users ORDER BY created_at DESC",
        ).fetchall()
        recent = _execute(
            connection,
            """SELECT s.id,s.created_at,u.name,u.email FROM sessions s
               JOIN users u ON u.id=s.user_id ORDER BY s.created_at DESC LIMIT 30""",
        ).fetchall()

    def count_value(row):
        return row["count"] if isinstance(row, dict) else row[0]

    return {
        "users": count_value(users_count),
        "individuals": count_value(individual_count),
        "sessions": count_value(sessions_count),
        "people": [_normalize_record(row) for row in people],
        "recent": [_normalize_record(row) for row in recent],
    }


def save_draft(user_id, payload):
    """Persist an unfinished assessment locally, never in the cloud database."""
    serialized = json.dumps(payload, allow_nan=False, default=_json_default, sort_keys=True)
    with _sqlite_connection(DB_PATH) as connection:
        row = connection.execute(
            "SELECT payload FROM assessment_drafts WHERE user_id=?", (user_id,)
        ).fetchone()
        if row and row["payload"] == serialized:
            return False
        connection.execute(
            """INSERT INTO assessment_drafts(user_id,payload) VALUES(?,?)
               ON CONFLICT(user_id) DO UPDATE SET
               payload=excluded.payload,updated_at=CURRENT_TIMESTAMP""",
            (user_id, serialized),
        )
    return True


def load_draft(user_id):
    with _sqlite_connection(DB_PATH) as connection:
        row = connection.execute(
            "SELECT payload,updated_at FROM assessment_drafts WHERE user_id=?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    payload = json.loads(row["payload"])
    if not isinstance(payload, dict):
        raise ValueError("The saved assessment draft is not a JSON object.")
    return payload


def delete_draft(user_id):
    with _sqlite_connection(DB_PATH) as connection:
        cursor = connection.execute(
            "DELETE FROM assessment_drafts WHERE user_id=?", (user_id,)
        )
        return cursor.rowcount > 0
