"""Тесты этапа 5: единый интерфейс методов кластеризации."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from src import clustering, network


@pytest.fixture(scope="module")
def toy():
    X, y = make_blobs(n_samples=90, centers=3, cluster_std=0.5, random_state=42)
    import pandas as pd

    df = pd.DataFrame(X, columns=["a", "b"])
    D = network.econ_distance(df, {"a": 1, "b": 1})
    W = network.to_weight(D)
    A = network.sparsify(D, W, {"method": "knn_mst", "k": 7})
    return X, W * A, y


METHODS = ["kmeans", "ward", "ward_kmeans", "gmm", "leiden", "spectral", "kefrin", "canus"]


@pytest.mark.parametrize("method", METHODS)
def test_method_recovers_blobs(toy, method):
    ok, why = clustering.available(method)
    if not ok:
        pytest.skip(why)
    X, W, y = toy
    p = {**clustering.default_params()["methods"][method], "method": method, "k": 3, "seed": 42}
    if method == "canus":
        p["epochs"] = 100
    lab = clustering.fit(X, W, p)
    assert lab.shape == (len(X),) and set(lab) == {0, 1, 2}
    assert adjusted_rand_score(y, lab) > 0.8


@pytest.mark.parametrize("method", ["kmeans", "gmm", "leiden", "spectral", "kefrin"])
def test_deterministic_with_seed(toy, method):
    ok, why = clustering.available(method)
    if not ok:
        pytest.skip(why)
    X, W, _ = toy
    p = {**clustering.default_params()["methods"][method], "method": method, "k": 3, "seed": 42}
    assert (clustering.fit(X, W, p) == clustering.fit(X, W, p)).all()


def test_relabel_order():
    assert clustering.relabel(np.array([5, 5, 2, 9, 2])).tolist() == [0, 0, 1, 2, 1]


def test_graph_method_needs_graph(toy):
    X, _, _ = toy
    with pytest.raises(ValueError):
        clustering.fit(X, None, {"method": "leiden", "k": 3})


# ---------------------------------------------------------------- Уорд + k-means (этап 4 доработки)
def test_ward_kmeans_starts_from_ward_centroids(toy):
    """k-means запускается из центроидов Уорда и не ухудшает их: инерция ≤ инерции разбиения Уорда."""
    from sklearn.cluster import KMeans

    X, _, _ = toy
    C = clustering.ward_centroids(X, 3)
    lab = clustering.fit(X, None, {"method": "ward_kmeans", "k": 3, "seed": 42})
    km = KMeans(n_clusters=3, init=C, n_init=1).fit(X)
    assert adjusted_rand_score(km.labels_, lab) == 1.0
    ward = clustering.fit(X, None, {"method": "ward", "k": 3, "seed": 42})
    inertia = lambda lb: sum(((X[lb == c] - X[lb == c].mean(0)) ** 2).sum() for c in np.unique(lb))  # noqa: E731
    assert inertia(lab) <= inertia(ward) + 1e-9


def test_ward_subsample_for_large_n():
    X, _ = make_blobs(n_samples=3000, centers=4, cluster_std=0.4, random_state=1)
    Z, idx = clustering.ward_linkage(X, max_n=500)
    assert len(idx) == 500 and len(Z) == 499
    lab = clustering.fit(X, None, {"method": "ward_kmeans", "k": 4, "seed": 42, "ward_max_n": 500})
    assert set(lab) == {0, 1, 2, 3}


def test_ward_jumps_pick_true_k():
    X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.3, random_state=3)
    Z, _ = clustering.ward_linkage(X)
    jt = clustering.ward_jumps(Z, (2, 8))
    assert jt[jt["k"].ge(3)].sort_values("скачок, раз", ascending=False)["k"].iloc[0] == 4


def test_pooled_ward_kmeans_same_labels_for_same_point():
    import pandas as pd

    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.3, random_state=0)
    idx = pd.MultiIndex.from_product([range(30), [2020, 2021]], names=["territory_id", "year"])
    Xp = pd.DataFrame(X, index=idx)
    Xp.loc[(slice(None), 2021), :] = Xp.xs(2020, level="year").to_numpy()  # показатели не изменились
    lab = clustering.fit_pooled(Xp, {"method": "ward_kmeans", "k": 3, "seed": 42})
    assert (lab.xs(2020, level="year") == lab.xs(2021, level="year")).all()  # → тип не меняется


def test_order_by_value_k1_is_highest():
    lab = np.array([0, 0, 1, 1, 2, 2, 3])
    val = np.array([10, 12, 100, 90, 50, 55, np.nan])
    out = clustering.order_by_value(lab, val)
    assert list(out) == [2, 2, 0, 0, 1, 1, 3]  # K1 — кластер 1 (медиана 95), неизвестный — последним
    assert clustering.code(0) == "K1"


def test_pooled_with_structurally_missing_feature():
    """Признак отсутствует у всех МО в части лет (специализация до 2017 г.): модель учится на полных строках,
    остальные относятся к ближайшему центру; если признака нет во всём окне — он не участвует."""
    import pandas as pd

    X, _ = make_blobs(n_samples=120, centers=3, n_features=3, cluster_std=0.3, random_state=0)
    idx = pd.MultiIndex.from_product([range(40), [2015, 2016, 2017]], names=["territory_id", "year"])
    Xp = pd.DataFrame(X, index=idx, columns=["a", "b", "hhi"])
    Xp.loc[(slice(None), [2015, 2016]), "hhi"] = np.nan
    lab = clustering.fit_pooled(Xp, {"method": "ward_kmeans", "k": 3, "seed": 42})
    assert lab.ge(0).all() and set(lab) == {0, 1, 2}
    only_old = Xp.loc[(slice(None), [2015, 2016]), :]
    lab2 = clustering.fit_pooled(only_old, {"method": "kmeans", "k": 3, "seed": 42})
    assert lab2.ge(0).all() and len(lab2) == len(only_old)
