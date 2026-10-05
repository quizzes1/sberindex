"""Пересчёт денежных показателей в цены базового года (src/prices.py, этап 3 доработки)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import clustering, network, prices
from src import indicators as ind
from src.io import PROCESSED

needs_prices = pytest.mark.skipif(not prices.LEVELS_PARQUET.exists(), reason="нет price_levels.parquet")
needs_ind = pytest.mark.skipif(not (PROCESSED / "indicators_wide.parquet").exists(), reason="нет показателей")


# ---------------------------------------------------------------- реестр
def test_every_monetary_indicator_has_deflator():
    for r in ind.registry():
        assert "money" not in r, f"{r['code']}: устаревшее поле money — нужно monetary + deflator"
        if r.get("monetary"):
            assert r.get("deflator") in prices.DEFLATORS, f"{r['code']}: нет дефлятора или он неизвестен"
        else:
            assert "deflator" not in r, f"{r['code']}: дефлятор у неденежного показателя"


def test_ruble_units_are_monetary():
    """Показатель в рублях без флага monetary смешал бы номинал с реальными значениями."""
    for r in ind.registry():
        if str(r["unit"]).startswith("руб"):
            assert r.get("monetary"), r["code"]


# ---------------------------------------------------------------- цепной индекс: ручная проверка
def test_annual_avg_level_by_hand():
    # 2020: цены неизменны; 2021: +1% каждый месяц; 2022: снова без изменений
    m = pd.DataFrame([[100.0] * 12, [101.0] * 12, [100.0] * 12], index=[2020, 2021, 2022], columns=range(1, 13))
    lv = prices.annual_avg_level(m)
    assert lv[2020] == pytest.approx(1.0)
    assert lv[2021] == pytest.approx(np.mean(1.01 ** np.arange(1, 13)))
    assert lv[2022] == pytest.approx(1.01**12)
    # индекс «в среднем за год к предыдущему»: 2022/2021 = 1.01^12 / mean(1.01^1..12)
    assert lv[2022] / lv[2021] == pytest.approx(1.01**12 / np.mean(1.01 ** np.arange(1, 13)))


def test_chain_annual_and_real_value_by_hand():
    # индексы «к предыдущему году»: 2021 — 110%, 2022 — 105%, 2023 — 120%
    lv = prices.chain_annual(pd.Series({2021: 110.0, 2022: 105.0, 2023: 120.0}))
    assert lv.to_dict() == pytest.approx({2020: 1.0, 2021: 1.1, 2022: 1.155, 2023: 1.386})
    # 1000 руб. 2021 г. в ценах 2023 г. = 1000 × 1.386 / 1.1 = 1260
    assert 1000 * lv[2023] / lv[2021] == pytest.approx(1260.0)
    # 2000 руб. 2023 г. в ценах 2020 г. = 2000 / 1.386
    assert 2000 * lv[2020] / lv[2023] == pytest.approx(2000 / 1.386)


def test_dec_dec_average_is_geometric_mean():
    lv = prices.dec_dec_avg_level(pd.Series({2021: 110.0, 2022: 120.0}))
    assert lv[2021] == pytest.approx(np.sqrt(1.0 * 1.1))
    assert lv[2022] == pytest.approx(np.sqrt(1.1 * 1.32))


def test_rosstat_sentinels_break_chain():
    """Служебные коды Росстата (−777777) не перемножаются: цепочка начинается после разрыва."""
    s = prices.chain_annual(pd.Series({2018: 104.0, 2019: -777777.77, 2020: 103.0, 2021: 105.0}))
    assert list(s.index) == [2019, 2020, 2021]
    assert s[2021] == pytest.approx(1.03 * 1.05)


@needs_prices
def test_cpi_annual_average_matches_rosstat():
    """ИПЦ в среднем за год к предыдущему (Росстат): 2021 — 106,69%; 2022 — 113,75%; 2023 — 105,94%."""
    lv = prices.levels()
    cpi = lv[lv["region_code"].eq(prices.RUSSIA) & lv["deflator"].eq("cpi")].set_index("year")["level"]
    for y, ref in {2021: 106.69, 2022: 113.75, 2023: 105.94}.items():
        assert cpi[y] / cpi[y - 1] * 100 == pytest.approx(ref, abs=0.1), y


# ---------------------------------------------------------------- базовый год
@needs_prices
@pytest.mark.parametrize("scope", ["national", "regional"])
def test_base_year_real_equals_nominal(scope):
    df = pd.DataFrame(
        {
            "region_code": [25, 27, 77, 25],
            "year": [2023, 2023, 2023, 2017],
            "wage": [80000.0, 90000.0, 120000.0, 40000.0],
            "gmp_pc": [1e6, 2e6, 3e6, 5e5],
        }
    )
    r = prices.to_real(df, ["wage", "gmp_pc"], 2023, scope)
    base = df["year"].eq(2023)
    assert np.allclose(r.loc[base, ["wage", "gmp_pc"]], df.loc[base, ["wage", "gmp_pc"]])
    assert not np.allclose(r.loc[~base, "wage"], df.loc[~base, "wage"])  # другие годы — пересчитаны


@needs_prices
def test_non_monetary_untouched():
    df = pd.DataFrame({"region_code": [25], "year": [2017], "density": [12.5], "hhi_emp": [0.2]})
    r = prices.to_real(df, ["density", "hhi_emp"], 2023, "regional", spatial=True)
    pd.testing.assert_frame_equal(r, df)


@needs_prices
@needs_ind
def test_national_base_year_change_keeps_features_and_labels():
    """По России коэффициент L_b/L_t одинаков для всех МО года: смена базового года умножает все реальные
    значения на одну константу, поэтому нормированные признаки и метки кластеров не меняются."""
    p = network.default_params()
    p["sample"]["federal_districts"] = ["ДФО"]
    a = network.merge_params(p, {"prices": {"values": "real", "base_year": 2017, "deflator_scope": "national"}})
    b = network.merge_params(p, {"prices": {"values": "real", "base_year": 2023, "deflator_scope": "national"}})
    assert network.config_hash(a) != network.config_hash(b)  # параметры цен входят в хэш
    fa, fb = network.features(a), network.features(b)
    pd.testing.assert_frame_equal(fa, fb, check_exact=False, atol=1e-10)
    X = fa.xs(2022, level="year").dropna()
    la = clustering.fit(X, None, {"method": "kmeans", "k": 5, "seed": 42})
    lb = clustering.fit(fb.xs(2022, level="year").dropna(), None, {"method": "kmeans", "k": 5, "seed": 42})
    assert (np.asarray(la) == np.asarray(lb)).all()


@needs_prices
@needs_ind
def test_nominal_differs_from_real():
    p = network.default_params()
    p["sample"]["federal_districts"] = ["ДФО"]
    real = network.features(p)
    nom = network.features(network.merge_params(p, {"prices": {"values": "nominal"}}))
    assert not np.allclose(real["gmp_pc"].dropna(), nom["gmp_pc"].dropna())
    pd.testing.assert_series_equal(real["density"], nom["density"])
