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


METHODS = ["kmeans", "ward", "gmm", "leiden", "spectral", "kefrin", "canus"]


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
