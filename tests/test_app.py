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
    "pages/7_Сводные_таблицы.py",
    "pages/8_Обновление_данных.py",
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


@needs_data
def test_clusters_pooled_ward_kmeans_with_dendrogram():
    """Страница «Кластеры»: режим pooled, «Уорд + k-means» по умолчанию, дендрограмма, подписи K1…Kn."""
    at = AppTest.from_file(str(ROOT / "app" / "pages/4_Кластеры.py"), default_timeout=900)
    at.session_state["sample_kind"] = "ДФО"
    at.session_state["years_sel"] = (2021, 2024)
    at.run()
    next(s for s in at.selectbox if s.label == "Режим").set_value("pooled").run()
    assert not at.exception, [e.value for e in at.exception]
    assert next(s for s in at.selectbox if s.label == "Метод").value == "ward_kmeans"
    assert any(e.label.startswith("Дендрограмма") for e in at.expander)
    assert any("K1" in c.value for c in at.caption)
    assert any(m.value.startswith("**Локоть: k =") for m in at.markdown)  # метод локтя в выборе k
