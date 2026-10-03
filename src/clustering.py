"""Методы кластеризации с единым интерфейсом (этап 5).

    fit(features, graph, params) -> labels

features — нормированные признаки (DataFrame n × p или ndarray), graph — матрица весов рёбер
разреженной сети (n × n, симметричная, нули на диагонали; для методов по атрибутам может быть
None), params — {"method": ..., "k": ..., "seed": ..., + параметры метода}. Порядок узлов в features
и graph одинаковый. Возвращает целочисленные метки 0..K−1.

Добавить метод: функция _fit_<имя>(X, W, params) -> labels + запись в configs/clustering.yaml.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from src.io import ROOT, load_yaml

EXTERNAL = ROOT / "external"


def default_params() -> dict:
    return load_yaml("clustering.yaml")


def relabel(labels: np.ndarray) -> np.ndarray:
    """Метки 0..K−1 в порядке первого появления (детерминированно)."""
    _, first = np.unique(labels, return_index=True)
    order = np.unique(labels)[np.argsort(first)]
    m = {v: i for i, v in enumerate(order)}
    return np.array([m[v] for v in labels], dtype=int)


# ============================================================================ по атрибутам
def _fit_kmeans(X, W, p):
    return KMeans(n_clusters=p["k"], n_init=p.get("n_init", 20), random_state=p["seed"]).fit_predict(X)


def _fit_ward(X, W, p):
    return AgglomerativeClustering(n_clusters=p["k"], linkage="ward").fit_predict(X)


def _fit_gmm(X, W, p):
    g = GaussianMixture(
        n_components=p["k"],
        covariance_type=p.get("covariance_type", "diag"),
        n_init=p.get("n_init", 5),
        random_state=p["seed"],
        reg_covar=1e-5,
    )
    return g.fit(X).predict(X)


# ============================================================================ по графу
def _leiden_partition(W: np.ndarray, resolution: float, seed: int) -> np.ndarray:
    import igraph as ig
    import leidenalg

    i, j = np.where(np.triu(W, 1) > 0)
    g = ig.Graph(n=len(W), edges=list(zip(i.tolist(), j.tolist())), directed=False)
    g.es["weight"] = W[i, j].tolist()
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=-1,
    )
    return np.array(part.membership)


def _fit_leiden(X, W, p):
    """Leiden (оптимизация модулярности с параметром разрешения γ).

    Число сообществ задаётся не прямо, а через γ: при resolution_search подбирается γ
    (двоичный поиск по log γ), при котором число сообществ равно k; если точное k не
    достигается — берётся ближайшее, фактическое K возвращают индексы.
    """
    if not p.get("resolution_search", True):
        return _leiden_partition(W, p.get("resolution", 1.0), p["seed"])
    lo, hi = -6.0, 4.0
    best, best_gap = None, np.inf
    for _ in range(40):
        mid = (lo + hi) / 2
        lab = _leiden_partition(W, 10**mid, p["seed"])
        K = len(np.unique(lab))
        if abs(K - p["k"]) < best_gap:
            best, best_gap = lab, abs(K - p["k"])
        if K == p["k"]:
            break
        if K < p["k"]:
            lo = mid
        else:
            hi = mid
    return best


def _fit_spectral(X, W, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = SpectralClustering(
            n_clusters=p["k"], affinity="precomputed", random_state=p["seed"], assign_labels="cluster_qr"
        )
        return sc.fit_predict(W + 1e-12 * (W > 0))


# ============================================================================ атрибутированные сети (внешний код)
def _external(name: str, module: str):
    path = EXTERNAL / name
    if not path.exists():
        raise ImportError(f"{name} не склонирован: python scripts/fetch_external.py --only {name}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module(module)


def _fit_kefrin(X, W, p):
    """KEFRiN (Shalileh, Mirkin 2022): расширенный k-means по признакам и строкам матрицы сети.

    Вызывается из локальной копии репозитория (external/KEFRiN), код не копируется.
    Предобработка признаков отключена: X уже нормирован нашим конвейером.
    """
    kf = _external("KEFRiN", "kefrin")
    metric = {
        "euclidean": kf.DistanceMetric.EUCLIDEAN,
        "cosine": kf.DistanceMetric.COSINE,
        "manhattan": kf.DistanceMetric.MANHATTAN,
    }[p.get("metric", "euclidean")]
    cfg = kf.KEFRiNConfig(
        n_clusters=p["k"],
        rho=p.get("rho", 1.0),
        xi=p.get("xi", 1.0),
        distance_metric=metric,
        random_state=p["seed"],
        n_init=p.get("n_init", 10),
        preprocessing_y=kf.PreprocessingMethod("none"),
        preprocessing_p=kf.PreprocessingMethod("none"),
    )
    import logging

    logging.getLogger("kefrin").setLevel(logging.WARNING)
    return kf.KEFRiN(cfg).fit_predict(np.asarray(X, float), np.asarray(W, float))


def _fit_canus(X, W, p):
    """CANUS (Shalileh 2025): кластеризация атрибутированной сети фильтрованным градиентным спуском.

    Вызывается из external/CANUS; нужен PyTorch (pip install torch).
    """
    cn = _external("CANUS", "canus")
    m = cn.CANUSClusterer(
        n_clusters=p["k"],
        epochs=p.get("epochs", 300),
        rho=p.get("rho", 1.0),
        zeta=p.get("zeta", 1.0),
        seed=p["seed"],
        device="cpu",
        attribute_distance=p.get("attribute_distance", "minkowski"),
        network_distance=p.get("network_distance", "cosine"),
    )
    m.fit(np.asarray(X, np.float32), np.asarray(W, np.float32))
    return np.asarray(m.y_pred)


FAMILIES = {
    "kmeans": "attributes",
    "ward": "attributes",
    "gmm": "attributes",
    "leiden": "graph",
    "spectral": "graph",
    "kefrin": "attributed_network",
    "canus": "attributed_network",
}


def available(method: str) -> tuple[bool, str]:
    """Можно ли запустить метод (для внешних — склонирован ли репозиторий и есть ли зависимости)."""
    try:
        if method == "kefrin":
            _external("KEFRiN", "kefrin")
        elif method == "canus":
            importlib.import_module("torch")
            _external("CANUS", "canus")
        elif method == "leiden":
            importlib.import_module("leidenalg")
        return True, ""
    except ImportError as e:
        return False, str(e)


def fit(features: pd.DataFrame | np.ndarray, graph: np.ndarray | None, params: dict) -> np.ndarray:
    """Единый интерфейс: метки кластеров для узлов (порядок как в features)."""
    method = params["method"]
    f = globals().get(f"_fit_{method}")
    if f is None:
        raise ValueError(f"Неизвестный метод: {method}")
    X = np.asarray(features, dtype=float)
    p = {"seed": 42, **params}
    if FAMILIES.get(method) != "attributes" and graph is None:
        raise ValueError(f"{method}: нужен граф")
    return relabel(np.asarray(f(X, graph, p)))


def fit_pooled(X_panel: pd.DataFrame, params: dict) -> pd.Series:
    """Режим «по всей панели»: одна модель на все МО-годы сразу (только методы по атрибутам).

    X_panel — нормированные признаки с индексом (territory_id, year) (нормировка по всей
    панели). Типы одинаковы для всех лет по построению, поэтому смена типа МО означает
    изменение его показателей, а не перекластеризацию года. Возвращает метки с тем же индексом.
    """
    if FAMILIES.get(params["method"]) != "attributes":
        raise ValueError("Режим pooled доступен только для методов по атрибутам (kmeans, ward, gmm)")
    lab = fit(X_panel.to_numpy(), None, params)
    return pd.Series(lab, index=X_panel.index, name="label")
