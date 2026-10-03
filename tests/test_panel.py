"""Тесты этапа 2: справочник, привязка ОКТМО, годовая панель."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bdmo import is_summary_oktmo
from src.dictionary import load_dict, map_to_territory, norm_oktmo
from src.io import PROCESSED
from src.regional import okved2_letter, region_code_from_name

needs_panel = pytest.mark.skipif(
    not (PROCESSED / "panel_long.parquet").exists(), reason="панель не собрана (scripts/build_panel.py)"
)


def test_norm_oktmo():
    s = pd.Series(["79-701-000-000", "7548000", "05602000"])
    assert norm_oktmo(s).tolist() == ["79701000", "07548000", "05602000"]


def test_summary_codes():
    s = pd.Series(["05600000", "05700000", "05602000", "05602101"])
    assert is_summary_oktmo(s).tolist() == [True, True, False, True]


def test_okved_letters_cyrillic_lookalikes():
    assert okved2_letter("Раздел Н Транспортировка и хранение") == "H"
    assert okved2_letter("Раздел В Добыча полезных ископаемых") == "B"
    assert okved2_letter("Раздел C Обрабатывающие производства") == "C"
    assert okved2_letter("Всего по обследуемым видам экономической деятельности") is None


def test_region_names():
    assert region_code_from_name("Тюменская область (без Ханты-Мансийского авт.округа-Югра и Ямало-Ненецкого)") == 72
    assert region_code_from_name("Тюменская область") is None
    assert region_code_from_name("в т.ч. Ханты-Мансийский автономный округ") == 86
    assert region_code_from_name("Еврейская автономная  область") == 79
    assert region_code_from_name("Дальневосточный федеральный округ") is None


def test_map_by_year_and_peer():
    d = load_dict()
    # Каларский район: 76615000 до 2021 г., с 2021 г. — округ 76515000, territory_id тот же
    tid = d.loc[d.oktmo8.eq("76615000"), "territory_id"].iloc[0]
    df = pd.DataFrame(
        {
            "oktmo8": ["76615000", "76515000", "99999000"],
            "oktmo_stable8": ["76515000", "76515000", "99999000"],
            "year": [2019, 2023, 2020],
        }
    )
    m = map_to_territory(df)
    assert m["territory_id"].iloc[0] == tid and m["match"].iloc[0] == "by_year"
    assert m["territory_id"].iloc[1] == tid
    assert pd.isna(m["territory_id"].iloc[2]) and m["match"].iloc[2] == "нет в справочнике"


@needs_panel
def test_no_duplicate_keys():
    long = pd.read_parquet(PROCESSED / "panel_long.parquet")
    assert not long.duplicated(["territory_id", "year", "indicator"]).any()


@needs_panel
def test_territories_in_dictionary():
    long = pd.read_parquet(PROCESSED / "panel_long.parquet", columns=["territory_id"])
    assert long["territory_id"].isin(load_dict()["territory_id"]).all()


@needs_panel
def test_population_positive():
    long = pd.read_parquet(PROCESSED / "panel_long.parquet")
    assert (long.loc[long["indicator"].eq("pop"), "value"] > 0).all()


@needs_panel
def test_spend_shares():
    long = pd.read_parquet(PROCESSED / "panel_long.parquet")
    s = long[long["indicator"].str.startswith("spend_share")].pivot_table(
        index=["territory_id", "year"], columns="indicator", values="value"
    )
    assert np.allclose(s.sum(axis=1), 1.0)
    assert (s["spend_share_other"] >= -1e-9).all()  # сумма пяти категорий не больше итога


@needs_panel
def test_wide_matches_long():
    long = pd.read_parquet(PROCESSED / "panel_long.parquet")
    wide = pd.read_parquet(PROCESSED / "panel_wide.parquet")
    assert not wide.duplicated(["territory_id", "year"]).any()
    p = long[long["indicator"].eq("pop")].set_index(["territory_id", "year"])["value"]
    w = wide.set_index(["territory_id", "year"])["pop"].dropna()
    assert np.allclose(w.sort_index(), p.reindex(w.index).sort_index())


@needs_panel
def test_no_double_counting_on_type_change():
    """Ольский округ (Магаданская обл.): в 2015 г. значение стояло под старым и новым кодом."""
    long = pd.read_parquet(PROCESSED / "panel_long.parquet")
    v = long[(long["territory_id"] == 1431) & (long["indicator"] == "pop_avg") & (long["year"] == 2015)]["value"]
    assert len(v) == 1 and v.iloc[0] < 15000
