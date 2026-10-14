"""Проверка восстановления диалога в реальном Streamlit test runner."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_ui_loads_and_creates_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    app = AppTest.from_file(Path("app.py").resolve()).run(timeout=20)
    assert not app.exception
    assert len(app.selectbox[0].options) == 1
    app.button[0].click().run()
    assert not app.exception
    assert len(app.selectbox[0].options) == 2
