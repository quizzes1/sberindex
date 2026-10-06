"""Тесты индексов качества (этап 5) на игрушечных данных с известным ответом и сверка с эталонами."""

from __future__ import annotations

import sys

import networkx as nx
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from src import clustering, icvi
from src.io import ROOT


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
    return X, y


def two_cliques(n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Две несвязанные клики по n узлов."""
    A = np.zeros((2 * n, 2 * n))
    A[:n, :n] = 1
    A[n:, n:] = 1
    np.fill_diagonal(A, 0)
    return A, np.repeat([0, 1], n)


def test_network_indices_perfect_partition():
    A, lab = two_cliques()
    assert icvi.avi(A, lab) == pytest.approx(1.0)
    assert icvi.avu(A, lab) == pytest.approx(0.0)
    assert icvi.anui(A, lab) == pytest.approx(1.0)
    assert icvi.mq(A, lab) == pytest.approx(0.5)


def test_mq_matches_networkx():
    g = nx.karate_club_graph()
    A = nx.to_numpy_array(g, weight="weight")
    comms = nx.community.greedy_modularity_communities(g, weight="weight")
    lab = np.zeros(len(A), int)
    for c, nodes in enumerate(comms):
        lab[list(nodes)] = c
    assert icvi.mq(A, lab) == pytest.approx(nx.community.modularity(g, comms, weight="weight"))


def test_network_indices_match_reference_pattern():
    """Сверка AVI/AVU/ANUI/модулярности с эталонной реализацией Sorooshi/Pattern (коммит e5ff50b)."""
    path = ROOT / "external" / "Pattern"
    if not path.exists():
        pytest.skip("external/Pattern не склонирован (python scripts/fetch_external.py --only Pattern)")
    sys.path.insert(0, str(path / "metrics"))
    try:
        import clustering_metrics as ref
    except ImportError as e:  # зависимости эталона
        pytest.skip(str(e))
    rng = np.random.default_rng(42)
    for _ in range(5):
        n = 40
        A = rng.random((n, n)) * (rng.random((n, n)) < 0.2)
        A = np.triu(A, 1)
        A = A + A.T
        lab = rng.integers(0, 4, n)
        r = ref.AdjacencyClusteringMetrics().get_metric(A, lab)
        assert icvi.avi(A, lab) == pytest.approx(r["AVI"])
        assert icvi.avu(A, lab) == pytest.approx(r["AVU"])
        assert icvi.anui(A, lab) == pytest.approx(r["ANUI"])
        assert icvi.mq(A, lab) == pytest.approx(r["modularity"])


def test_feature_indices_pick_true_k(blobs):
    X, _ = blobs
    res = {}
    for k in range(2, 8):
        lab = clustering.fit(X, None, {"method": "kmeans", "k": k, "seed": 42})
        res[k] = icvi.compute_all(X, None, lab)
    assert max(res, key=lambda k: res[k]["SW"]) == 4
    assert max(res, key=lambda k: res[k]["CH"]) == 4
    assert min(res, key=lambda k: res[k]["S_Dbw"]) == 4


def test_sdbw_matches_reference_package(blobs):
    """S_Dbw: при sigma="std" совпадает с пакетом s_dbw (method='Halkidi'); по умолчанию — как в статье
    (дисперсии), значение другое, но выбор числа кластеров тот же (см. test_feature_indices_pick_true_k)."""
    s_dbw = pytest.importorskip("s_dbw")
    X, y = blobs
    for lab in (y, (y + (X[:, 0] > 0)) % 4):
        ref = s_dbw.S_Dbw(X, lab, method="Halkidi", centr="mean", nearest_centr=False)
        assert icvi.s_dbw(X, lab, sigma="std") == pytest.approx(ref, rel=1e-6)


def test_sw_bounds_and_random_partition(blobs):
    X, y = blobs
    good = icvi.sw(X, y)
    rnd = icvi.sw(X, np.random.default_rng(42).integers(0, 4, len(X)))
    assert -1 <= rnd < good <= 1 and good > 0.6


def test_directions_defined():
    for k in (*icvi.FEATURE_INDICES, *icvi.NETWORK_INDICES):
        assert icvi.BETTER[k] in (-1, 1)


# ---------------------------------------------------------------- метод локтя
def test_wcss_equals_kmeans_inertia():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=150, centers=3, random_state=0)
    km = KMeans(n_clusters=3, n_init=5, random_state=0).fit(X)
    assert icvi.wcss(X, km.labels_) == pytest.approx(km.inertia_)
    lab = km.labels_.copy()
    lab[:10] = -1  # без метки — не учитывается
    assert icvi.wcss(X, lab) < km.inertia_


def test_elbow_finds_true_k():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=7)
    ks = list(range(2, 11))
    w = [KMeans(n_clusters=k, n_init=5, random_state=0).fit(X).inertia_ for k in ks]
    assert icvi.elbow(ks, w) == 4


def test_elbow_degenerate():
    assert icvi.elbow([2, 3], [10.0, 5.0]) is None
    assert icvi.elbow([2, 3, 4], [1.0, 1.0, 1.0]) is None
    assert icvi.elbow([2, 3, 4, 5], [100.0, 20.0, 15.0, 12.0]) == 3
