"""Дымовые тесты интерфейса: каждая страница отрисовывается без исключений на ДФО и на всей России."""

from __future__ import annotations

import os

import pytest

from src.io import PROCESSED, ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "Home.py",
    "pages/1_Данные_и_качество.py",
    "pages/2_Показатели.py",
    "pages/3_Сеть.py",
    "pages/4_Кластеры.py",
    "pages/5_Динамика.py",
    "pages/6_Конвергенция.py",
]
needs_data = pytest.mark.skipif(not (PROCESSED / "indicators_wide.parquet").exists(), reason="данные не собраны")
SAMPLES = ["ДФО"] + (["Россия"] if os.environ.get("APP_TEST_RUSSIA") else [])


@needs_data
@pytest.mark.parametrize("sample", SAMPLES)
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, sample):
    at = AppTest.from_file(str(ROOT / "app" / page), default_timeout=900)
    at.session_state["sample_kind"] = sample
    at.session_state["years_sel"] = (2021, 2024)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
