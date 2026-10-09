"""Дымовые тесты интерфейса: каждая страница отрисовывается без исключений на ДФО и на всей России."""

from __future__ import annotations

import os

import pytest

from src.io import PROCESSED, ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [
    "Home.py",
    "hub.py",
    "overview.py",
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
    assert any(e.label.startswith("Дерево Уорда") for e in at.expander)
    assert any("K1" in c.value for c in at.caption)
    assert any(m.value.startswith("**Локоть: k =") for m in at.markdown)  # метод локтя в выборе k


@needs_data
def test_clusters_of_a_year_do_not_depend_on_window():
    """Окно «Годы» — только фильтр показа: модель и нормировка строятся на полной панели, поэтому кластеры 2014 г.
    при окнах 2014–2015 и 2014–2024 одинаковы и совпадают на страницах «Кластеры» и «Динамика»."""
    import json
    import re

    def sankey_2014(at):
        spec = next(json.loads(c.proto.spec) for c in at.get("plotly_chart") if "sankey" in c.proto.spec)
        cd = spec["data"][0]["node"]["customdata"]
        return {re.search(r": (K\d+), (\d+)", x).group(1): x for x in cd if x.startswith("2014")}

    out = []
    for win in [(2014, 2015), (2014, 2024)]:
        at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=900)
        at.session_state["sample_kind"] = "ДФО"
        at.session_state["years_sel"] = win
        at.run()
        at.switch_page("pages/5_Динамика.py").run()
        assert not at.exception, [e.value for e in at.exception]
        out.append(sankey_2014(at))
    assert out[0] == out[1]


def test_hub_documents_present_and_page_renders():
    """Стартовая страница: оба PDF и все отрисованные страницы на месте, страница открывается без ошибок."""
    docs = ROOT / "app" / "static" / "docs"
    for name in ("presentation", "report"):
        assert (docs / f"{name}.pdf").stat().st_size > 100_000
        pages = sorted((docs / name).glob("[0-9][0-9].jpg"))
        assert pages and pages[0].name == "01.jpg" and len(pages) == int(pages[-1].stem)
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "app/static/docs/presentation/01.jpg" in html and "app/static/docs/report.pdf" in html
    next(b for b in at.button if b.label == "▶").click().run()
    assert at.session_state["hub_slide"] == 2
