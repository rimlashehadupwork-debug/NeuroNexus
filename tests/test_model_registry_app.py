from pathlib import Path

from streamlit.testing.v1 import AppTest


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
