"""Методы кластеризации с единым интерфейсом.

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
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from src.io import ROOT, load_yaml

EXTERNAL = ROOT / "external"


def default_params() -> dict:
    """Параметры по умолчанию из соответствующего YAML в configs/."""
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


def _ward_sample(n: int, max_n: int | None, seed: int) -> np.ndarray:
    """Индексы строк для дерева Уорда: все, если n ≤ max_n; иначе случайная подвыборка (память O(n²))."""
    if not max_n or n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=int(max_n), replace=False))


def ward_linkage(X, max_n: int | None = None, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Дерево Уорда (матрица слияний scipy) и индексы строк, по которым оно построено."""
    from scipy.cluster.hierarchy import linkage

    X = np.asarray(X, dtype=float)
    idx = _ward_sample(len(X), max_n, seed)
    return linkage(X[idx], method="ward"), idx


def ward_jumps(Z: np.ndarray, k_range=(2, 10)) -> pd.DataFrame:
    """Расстояние слияния при переходе от k к k−1 кластерам и его скачок.

    Высота слияния, после которого остаётся k−1 кластер, — Z[n−k, 2]. Большой скачок между высотами
    соседних слияний означает, что дальше объединяются далёкие друг от друга группы, — разумно
    остановиться на k кластерах. Рекомендуемое k — с наибольшим относительным скачком.
    """
    h = Z[:, 2]
    n = len(h) + 1
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        if k >= n:
            break
        up, down = h[n - k], h[n - k - 1]  # слияние k → k−1 и предыдущее (k+1 → k)
        rows.append({"k": k, "высота слияния k→k−1": up, "скачок": up - down, "скачок, раз": up / down})
    return pd.DataFrame(rows)


def ward_centroids(X, k: int, max_n: int | None = None, seed: int = 42) -> np.ndarray:
    """Центроиды k кластеров Уорда (по подвыборке, если строк больше max_n)."""
    from scipy.cluster.hierarchy import fcluster

    X = np.asarray(X, dtype=float)
    Z, idx = ward_linkage(X, max_n, seed)
    lab = fcluster(Z, t=k, criterion="maxclust")
    return np.vstack([X[idx][lab == c].mean(axis=0) for c in np.unique(lab)])


def _fit_ward_kmeans(X, W, p):
    """Уорд → центроиды → k-means (один запуск из центров Уорда, без случайных стартов)."""
    C = ward_centroids(X, p["k"], p.get("ward_max_n"), p["seed"])
    return KMeans(n_clusters=len(C), init=C, n_init=1, random_state=p["seed"]).fit_predict(X)


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


def _kefrin_n_init(X, p) -> int:
    """Число случайных стартов KEFRiN: на больших выборках (вся Россия, ~2 200 МО) один старт
    стоит ~4 с, поэтому берём n_init_large вместо n_init (см. configs/clustering.yaml)."""
    if len(X) > p.get("large_n", float("inf")):
        return int(p.get("n_init_large", p.get("n_init", 10)))
    return int(p.get("n_init", 10))


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
        n_init=_kefrin_n_init(X, p),
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
    "ward_kmeans": "attributes",
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
    панели). Строки с пропусками (только структурными — network.usable_rows) в обучении не участвуют
    и относятся к ближайшему центру кластера по имеющимся признакам. Типы одинаковы для всех лет по построению, поэтому смена типа МО означает
    изменение его показателей, а не перекластеризацию года. Возвращает метки с тем же индексом.
    """
    if FAMILIES.get(params["method"]) != "attributes":
        raise ValueError("Режим pooled доступен только для методов по атрибутам (kmeans, ward, ward_kmeans, gmm)")
    # признак, которого нет во всём окне (например, специализации при окне 2014–2016), в модели не участвует
    X = X_panel.loc[:, X_panel.notna().any()].to_numpy(dtype=float)
    full = ~np.isnan(X).any(axis=1)
    if not full.any():
        raise ValueError("Нет МО-лет с полным набором признаков — выберите другие признаки или окно лет")
    lab = np.full(len(X), -1, dtype=int)
    lab[full] = fit(X[full], None, params)
    if (~full).any():
        # строки со структурно отсутствующим признаком (например, специализации до 2017 г. нет ни у кого):
        # ближайший центр кластера по имеющимся признакам (центры — по полным строкам)
        C = np.vstack([X[full][lab[full] == c].mean(axis=0) for c in range(lab[full].max() + 1)])
        P = X[~full]
        m = ~np.isnan(P)
        d = np.stack(
            [np.where(m, (np.nan_to_num(P) - C[c]) ** 2, 0.0).sum(axis=1) / m.sum(axis=1) for c in range(len(C))], 1
        )
        lab[~full] = d.argmin(axis=1)
    return pd.Series(lab, index=X_panel.index, name="label")


# ============================================================================ нумерация K1…Kn
def order_by_value(labels, values, descending: bool = True) -> np.ndarray:
    """Перенумеровать кластеры по медиане показателя: 0 — самые высокие значения (подпись K1).

    labels и values — одинаковой длины (values — например, ВМП на душу в ценах базового года).
    Кластеры, где показатель неизвестен у всех МО, — в конце (в порядке исходных номеров).
    """
    lab = np.asarray(labels)
    med = pd.Series(np.asarray(values, dtype=float)).groupby(lab).median()
    med = med.reindex(np.unique(lab))
    key = -med if descending else med
    order = sorted(med.index, key=lambda c: (np.isnan(key[c]), key[c] if np.isfinite(key[c]) else 0, c))
    m = {c: i for i, c in enumerate(order)}
    return np.array([m[c] for c in lab], dtype=int)


def code(label: int) -> str:
    """Подпись кластера: K1, K2, … (label 0 → K1)."""
    return f"K{int(label) + 1}"


@lru_cache(maxsize=8)
def _order_values(ind: str, base_year: int, scope: str, spatial: bool) -> pd.Series:
    """Показатель упорядочения кластеров в ценах base_year, индекс (territory_id, year) — кэш на процесс."""
    from src import prices
    from src.io import PROCESSED

    w = pd.read_parquet(PROCESSED / "indicators_wide.parquet", columns=["territory_id", "year", "region_code", ind])
    w = prices.to_real(w, [ind], base_year, scope, spatial)
    return w.set_index(["territory_id", "year"])[ind]


def order_labels(
    labels: pd.Series, price_params: dict | None = None, year: int | None = None, indicator: str | None = None
) -> pd.Series:
    """Метки, перенумерованные по показателю (по умолчанию ВМП на душу в ценах базового года): 0 → K1.

    labels — индекс (territory_id, year) (режим pooled / сквозные типы) или territory_id и тогда нужен year.
    Показатель — configs/clustering.yaml → order_by; денежный пересчитывается в цены base_year (src/prices.py).
    """
    from src import prices

    ind = indicator or default_params().get("order_by", "gmp_pc")
    pp = prices.price_params({**(price_params or {}), "values": "real"})
    v = _order_values(ind, int(pp["base_year"]), pp["deflator_scope"], bool(pp["spatial_price_adjustment"]))
    if isinstance(labels.index, pd.MultiIndex):
        vals = v.reindex(labels.index).to_numpy()
    else:
        vals = v.reindex(pd.MultiIndex.from_arrays([labels.index, np.full(len(labels), year)])).to_numpy()
    return pd.Series(order_by_value(labels.to_numpy(), vals), index=labels.index, name=labels.name)
