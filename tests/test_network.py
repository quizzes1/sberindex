"""Тесты этапа 4: расстояния и сеть."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import connected_components

from src import network
from src.io import PROCESSED

needs_ind = pytest.mark.skipif(not (PROCESSED / "indicators_wide.parquet").exists(), reason="нет показателей")


@pytest.fixture
def toy():
    rng = np.random.default_rng(42)
    # три далёкие друг от друга группы → kNN-граф несвязен, MST его сшивает
    pts = np.vstack([rng.normal(c, 0.01, size=(15, 2)) for c in (0.0, 0.5, 1.0)])
    return pd.DataFrame(pts, columns=["a", "b"], index=pd.Index(range(45), name="territory_id"))


def test_distance_symmetric_zero_diag(toy):
    for metric in ("weighted_l1", "euclidean"):
        D = network.econ_distance(toy, {"a": 1, "b": 2}, metric)
        assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0) and (D >= 0).all()


def test_single_feature_distance_is_abs_diff(toy):
    D = network.econ_distance(toy, {"a": 1}, "weighted_l1")
    assert np.allclose(D, np.abs(toy["a"].to_numpy()[:, None] - toy["a"].to_numpy()[None, :]))


@pytest.mark.parametrize("kind", ["gaussian", "inverse"])
def test_weights_in_unit_interval(toy, kind):
    D = network.econ_distance(toy, {"a": 1, "b": 1})
    W = network.to_weight(D, kind)
    off = ~np.eye(len(D), dtype=bool)
    assert (W[off] > 0).all() and (W <= 1).all()


def test_knn_mst_connected(toy):
    D = network.econ_distance(toy, {"a": 1, "b": 1})
    W = network.to_weight(D)
    A_knn = network.sparsify(D, W, {"method": "knn", "k": 3})
    assert connected_components(A_knn, directed=False)[0] > 1  # проверка, что пример содержательный
    A = network.sparsify(D, W, {"method": "knn_mst", "k": 3})
    assert connected_components(A, directed=False)[0] == 1
    assert (A == A.T).all() and not A.diagonal().any()


def test_mutual_subset_of_symmetric(toy):
    D = network.econ_distance(toy, {"a": 1, "b": 1})
    m = network.knn_adjacency(D, 5, "mutual")
    s = network.knn_adjacency(D, 5, "symmetric")
    assert (m <= s).all()


def test_config_hash_stable():
    p = network.default_params()
    assert network.config_hash(p) == network.config_hash(dict(reversed(list(p.items()))))
    q = network.merge_params(p, {"sparsify": {"k": 5}})
    assert network.config_hash(p) != network.config_hash(q) and q["sparsify"]["method"] == p["sparsify"]["method"]


@needs_ind
def test_default_network_connected_and_valid():
    p = network.default_params()
    nets = network.build(p, years=[2022])
    net = nets[2022]
    assert net.stats()["components"] == 1
    assert np.allclose(net.D, net.D.T) and np.allclose(np.diag(net.D), 0)
    e = net.edges()
    assert (e["weight"] > 0).all() and (e["weight"] <= 1).all()


@needs_ind
def test_geo_component_in_unit_interval():
    p = network.default_params()
    ids = network.sample_rows(p).query("year == 2022")["territory_id"].to_numpy()[:60]
    Dg, _ = network.geo_distance(
        np.sort(ids), {"weights": {"road_km": 1, "line_km": 1, "adjacency": 1}, "no_road": "line"}
    )
    assert np.nanmin(Dg) >= 0 and np.nanmax(Dg) <= 1 and np.allclose(Dg, Dg.T, equal_nan=True)


@needs_ind
def test_network_2014_without_specialization():
    """В 2014–2016 гг. специализации занятости (hhi_emp) по МО нет: сеть строится по остальным признакам."""
    p = network.default_params()
    p["sample"]["federal_districts"] = ["ДФО"]
    net = network.build(p, years=[2014])[2014]
    assert "hhi_emp" not in net.X.columns and len(net.ids) > 150
    assert net.stats()["components"] == 1
