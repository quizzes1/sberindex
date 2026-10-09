"""Тесты: динамика кластеров и конвергенция на синтетических данных с известным ответом."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import convergence, dynamics


def _labels(year, lab):
    return pd.DataFrame({"territory_id": np.arange(len(lab)), "year": year, "label": lab})


def test_matching_survives_relabeling():
    base = np.repeat([0, 1, 2], 10)
    perm = {0: 2, 1: 0, 2: 1}
    nxt = np.array([perm[v] for v in base])
    nxt[0] = perm[1]  # один МО действительно сменил тип 0 → 1
    th = dynamics.match_years(pd.concat([_labels(2020, base), _labels(2021, nxt)]), 0.2)
    t20 = th[th.year == 2020].set_index("territory_id")["through"]
    t21 = th[th.year == 2021].set_index("territory_id")["through"]
    assert (t20.iloc[1:] == t21.iloc[1:]).all()  # остальные сохранили сквозной номер
    assert t20.iloc[0] == 0 and t21.iloc[0] == 1
    mig = dynamics.migrants(th)
    assert len(mig) == 1 and mig.iloc[0]["territory_id"] == 0
    tr = dynamics.transitions(th)
    assert tr["n"].sum() == 30


def test_new_type_below_jaccard_threshold():
    a = np.repeat([0, 1], 10)
    b = np.r_[np.zeros(10, int), np.tile([1, 2], 5)]  # кластер 1 распался пополам
    th = dynamics.match_years(pd.concat([_labels(1, a), _labels(2, b)]), 0.6)
    # обе половины (Жаккар 0,5 < 0,6) — новые типы 2 и 3; тип 1 исчез
    assert set(th[th.year == 2]["through"]) == {0, 2, 3}


def test_year_summary_ari_one_for_identical():
    a = np.repeat([0, 1, 2], 5)
    s = dynamics.year_summary(dynamics.match_years(pd.concat([_labels(1, a), _labels(2, a)]), 0.2))
    assert s["ARI"].iloc[0] == pytest.approx(1.0) and s["доля сменивших тип"].iloc[0] == 0


def synthetic_panel(beta: float, n: int = 200, T: int = 12, seed: int = 42) -> pd.DataFrame:
    """ln y_t − ln y_{t−1} = g + beta·(ln y_{t−1} − mean) + шум: beta < 0 — конвергенция."""
    rng = np.random.default_rng(seed)
    ly = rng.normal(11, 0.6, n)
    rows = []
    for t in range(T):
        rows.append(pd.DataFrame({"territory_id": np.arange(n), "year": 2013 + t, "value": np.exp(ly)}))
        ly = ly + 0.02 + beta * (ly - ly.mean()) + rng.normal(0, 0.02, n)
    return pd.concat(rows, ignore_index=True)


def test_beta_absolute_detects_convergence():
    r = convergence.beta_absolute(synthetic_panel(-0.08))
    assert r["b"] < 0 and r["p"] < 0.01 and r["half_life"] > 0


def test_beta_absolute_no_convergence():
    r = convergence.beta_absolute(synthetic_panel(0.0))
    assert abs(r["b"]) < 0.01


def test_sigma_decreases_under_convergence():
    _, tr = convergence.sigma(synthetic_panel(-0.08))
    assert tr["slope"] < 0 and tr["p"] < 0.01
    _, tr2 = convergence.sigma(synthetic_panel(0.05))
    assert tr2["slope"] > 0


def test_beta_panel_negative():
    d = synthetic_panel(-0.08)
    region = pd.Series(np.arange(200) % 10, index=np.arange(200))
    r = convergence.beta_panel(d, region)
    assert r["b"] < 0 and r["p"] < 0.01


def test_log_t_convergence_vs_divergence():
    conv = convergence.log_t(convergence._log_panel(synthetic_panel(-0.15, T=20)))
    div = convergence.log_t(convergence._log_panel(synthetic_panel(0.1, T=20)))
    assert conv["t"] > -1.65 and div["t"] < -1.65


def test_per_year_rank_numbering_matches_cluster_page():
    """Режим «каждый год отдельно»: номер типа в динамике = место кластера по показателю внутри года (как на
    страницах «Кластеры» и «Сводные таблицы»), а не номер по Жаккару — один кластер везде называется одинаково."""
    import pandas as pd

    from src import dynamics

    lab = pd.DataFrame(
        {"territory_id": [1, 2, 3, 4, 1, 2, 3, 4], "year": [2020] * 4 + [2021] * 4, "label": [0, 0, 1, 1, 1, 1, 0, 0]}
    )
    th = dynamics.through_labels(lab, "per_year", order=False)
    assert (th["through"] == th["label"]).all() and th["through"].max() == 1  # ровно k типов
    tj = dynamics.through_labels(lab, "per_year", order=False, numbering="jaccard")
    # по Жаккару кластеры 2021 г. сопоставлены с 2020 г. — номера отличаются от меток года
    assert not (tj["through"] == tj["label"]).all()
