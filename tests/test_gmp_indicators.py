"""Тесты этапа 3: ВМП, показатели, нормировка."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import gmp
from src import indicators as ind
from src.io import PROCESSED, load_yaml
from src.normalize import normalize, prepare

needs_gmp = pytest.mark.skipif(not (PROCESSED / "gmp.parquet").exists(), reason="ВМП не посчитан")


@pytest.fixture(scope="module")
def data():
    if not (PROCESSED / "panel_wide.parquet").exists():
        pytest.skip("панель не собрана")
    return gmp.GMPInputs.load(
        pd.read_parquet(PROCESSED / "panel_wide.parquet"), pd.read_parquet(PROCESSED / "mo.parquet")
    )


def _params():
    c = load_yaml("gmp.yaml")
    return {**c["sectoral"], "imputation_weight": c["imputation_weight"]}


@pytest.mark.parametrize("method", ["basic", "sectoral"])
@pytest.mark.parametrize("year", [2017, 2020, 2024])
def test_gmp_sums_to_grp(data, method, year):
    """Сумма ВМП по МО субъекта = ВРП субъекта (метод 2 — ВДС «всего», совпадает с ВРП до округления Росстата)."""
    d = gmp.compute(data, method, [year], _params())
    s = d.groupby("region_code")["gmp"].sum()
    if method == "basic":
        ref = data.grp[data.grp["year"].eq(year)].set_index("region_code")["grp"] * 1000
        tol = 1e-9
    else:
        ref = data.gva[data.gva["year"].eq(year) & data.gva["section"].eq("TOTAL")].set_index("region_code")["gva"]
        tol = 1e-9
    assert np.allclose(s / ref.reindex(s.index), 1, atol=tol)


def test_section_shares_sum_to_one(data):
    """Доли распределения s_ik по каждой отрасли субъекта в сумме дают 1: Σ_i вклад_ik = ВДС_rk."""
    d = gmp.compute(data, "sectoral", [2022], _params())
    gva = data.gva[data.gva["year"].eq(2022)].pivot_table(index="region_code", columns="section", values="gva")
    for k in gmp.GVA_SECTIONS:
        s = d.groupby("region_code")[f"sec_{k}"].sum()
        ref = gva[k].reindex(s.index)
        ok = ref.ne(0)
        assert np.allclose(s[ok], ref[ok], rtol=1e-9), k


def test_imputed_share_in_unit_interval(data):
    d = gmp.compute(data, "sectoral", [2021], _params())
    sh = d["gmp_imputed"] / d["gmp"]
    assert sh.between(-1e-12, 1 + 1e-12).all()


@needs_gmp
def test_gmp_parquet_columns():
    g = pd.read_parquet(PROCESSED / "gmp.parquet")
    for c in ["territory_id", "year", "method", "gmp", "gmp_pc", "gmp_pc_real", "gmp_imputed_share"]:
        assert c in g
    assert not g.duplicated(["territory_id", "year", "method"]).any()


def test_registry_has_functions():
    for r in ind.registry():
        assert callable(getattr(ind, r["code"], None)), r["code"]
        for f in ("name", "formula", "source", "unit", "block", "direction"):
            assert f in r, (r["code"], f)


def test_minmax_unit_interval():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(200, 3)) * [1, 100, 1e6], columns=list("abc"))
    z = normalize(df, "minmax")
    assert np.allclose(z.min(), 0) and np.allclose(z.max(), 1)


def test_zscore_panel_and_year():
    rng = np.random.default_rng(42)
    idx = pd.MultiIndex.from_product([range(50), [2020, 2021]], names=["territory_id", "year"])
    df = pd.DataFrame({"x": rng.normal(5, 3, 100), "y": rng.lognormal(size=100)}, index=idx)
    z = normalize(df, "zscore", scope="panel")
    assert np.allclose(z.mean(), 0) and np.allclose(z.std(ddof=0), 1)
    zy = normalize(df, "zscore", scope="year")
    for _, g in zy.groupby(level="year"):
        assert np.allclose(g.mean(), 0) and np.allclose(g.std(ddof=0), 1)


def test_prepare_keeps_missing():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 100.0, 1000.0]})
    z = prepare(df, ["a", "b"], log_columns=["b"])
    assert z["a"].isna().sum() == 1 and np.allclose(z["b"].diff().dropna(), 0.5, atol=0.01)
