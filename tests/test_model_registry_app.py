from pathlib import Path

from streamlit.testing.v1 import AppTest

from modules import accounts


def test_model_registry_page_renders_darwin_status():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30)
    app.session_state["user"] = {
        "id": 1,
        "name": "Research admin",
        "role": "admin",
    }
    app.session_state["workspace_page"] = "Model registry"

    app.run()

    assert not app.exception
    assert any(element.value == "Alzheimer’s pathway" for element in app.subheader)
    assert any(
        "darwin" in element.value.casefold()
        for element in app.caption
        if isinstance(element.value, str)
    )


def test_my_reports_page_renders_plain_language_guide():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30)
    app.session_state["user"] = {
        "id": -1,
        "name": "Report reader",
        "role": "individual",
    }
    app.session_state["workspace_page"] = "My reports"

    app.run()

    assert not app.exception
    assert any(element.value == "How to read your results" for element in app.subheader)


def test_saved_report_page_displays_measurements_and_download_actions(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "reports.sqlite3")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    accounts.init_db()
    accounts.register("Report reader", "report@example.test", "long-password")
    user = accounts.authenticate("report@example.test", "long-password")
    accounts.save_session(user["id"], {
        "voice": {"duration_seconds": 4.2, "pitch_median_hz": 182.0},
        "cognitive_tasks": {
            "memory_check": {
                "completed": True,
                "items_presented": 3,
                "items_recalled": 2,
                "recall_latency_seconds": 5.6,
                "completion_status": "completed",
            },
        },
    })

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45)
    app.session_state["user"] = user
    app.session_state["workspace_page"] = "My reports"

    app.run()

    assert not app.exception
    measurement_table = next(
        element.value
        for element in app.dataframe
        if "Measurement" in element.value.columns
    )
    assert "Items recalled" in measurement_table["Measurement"].values
    assert any(
        element.label == "Download PDF"
        for element in app.download_button
    )
    assert any(
        element.label == "View full structured report"
        for element in app.expander
    )
