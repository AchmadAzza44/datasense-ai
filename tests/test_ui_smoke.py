"""Smoke test wiring Streamlit DataSense AI."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _fresh_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception
    return app


def test_page_renders_with_six_tabs():
    assert len(_fresh_app().tabs) == 6


def test_loads_sample_dataset():
    app = _fresh_app()
    sample = next(s for s in app.selectbox if s.key == "ds_sample_choice")
    sample.set_value("sales_regression.csv")
    next(b for b in app.button if b.label == "Muat dataset").click().run()
    assert not app.exception
