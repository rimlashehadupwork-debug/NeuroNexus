from modules import accounts


def test_sqlite_fallback_persists_consent_reports_and_local_drafts(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "accounts.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    accounts.register(
        "Participant",
        "participant@example.test",
        "long-password",
        consent_flags={"registration_acknowledged": True},
    )
    user = accounts.authenticate("participant@example.test", "long-password")

    assert accounts.storage_backend() == "sqlite"
    assert user["consent_flags"] == {"registration_acknowledged": True}
    flags = accounts.update_consent_flag(user["id"], "assessment", True)
    assert flags["assessment"] is True
    accounts.update_user_role(user["id"], "caregiver")
    assert accounts.authenticate("participant@example.test", "long-password")["role"] == "caregiver"

    report_id = accounts.save_session(user["id"], {"consent_flags": flags})
    assert report_id is not None
    assert len(accounts.user_sessions(user["id"])) == 1

    draft = {"audio_result": {"duration_seconds": 4.0}}
    assert accounts.save_draft(user["id"], draft)
    assert accounts.load_draft(user["id"]) == draft
    assert accounts.delete_draft(user["id"])
    assert accounts.load_draft(user["id"]) is None


def test_database_url_environment_selects_postgresql_without_connecting(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgres://example.invalid:5432/neuronexus",
    )

    assert accounts.get_database_url() == "postgresql://example.invalid:5432/neuronexus"
    assert accounts.storage_backend() == "postgresql"


def test_init_db_seeds_all_demo_accounts_idempotently(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "accounts.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    accounts.init_db()
    accounts.init_db()

    for email, expected_role in (
        ("individual@neuronexus.local", "individual"),
        ("caregiver@neuronexus.local", "caregiver"),
        ("admin@neuronexus.local", "admin"),
    ):
        user = accounts.authenticate(f"  {email}  ", "ChangeMe123!")
        assert user["role"] == expected_role

    with accounts.connect() as connection:
        rows = connection.execute(
            "SELECT email FROM users WHERE email LIKE '%@neuronexus.local'"
        ).fetchall()
    assert len(rows) == 3


def test_quick_demo_account_recreates_missing_account(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "accounts.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()

    with accounts.connect() as connection:
        connection.execute(
            "DELETE FROM users WHERE email=?",
            ("caregiver@neuronexus.local",),
        )

    user = accounts.demo_account("caregiver@neuronexus.local")

    assert user["email"] == "caregiver@neuronexus.local"
    assert user["role"] == "caregiver"
    assert accounts.authenticate(user["email"], "ChangeMe123!")["id"] == user["id"]


def test_delete_user_sessions_removes_only_the_selected_users_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "accounts.sqlite3")
    accounts.init_db()
    accounts.register("Participant One", "one@example.test", "long-password-one")
    accounts.register("Participant Two", "two@example.test", "long-password-two")
    first = accounts.authenticate("one@example.test", "long-password-one")
    second = accounts.authenticate("two@example.test", "long-password-two")
    accounts.save_session(first["id"], {"voice": {"duration_seconds": 4}})
    accounts.save_session(second["id"], {"voice": {"duration_seconds": 5}})

    deleted = accounts.delete_user_sessions(first["id"])

    assert deleted == 1
    assert accounts.user_sessions(first["id"]) == []
    assert len(accounts.user_sessions(second["id"])) == 1
