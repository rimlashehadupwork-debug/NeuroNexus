from modules import accounts


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
