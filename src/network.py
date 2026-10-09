"""Расстояния между МО и сеть на каждый год.

Шаги для года Y:
1. признаки МО выборки (indicators_wide, только действующие в году МО) → логарифм по реестру →
   нормировка (по умолчанию min-max по всем годам выборки сразу);
2. экономическое расстояние по показателю k: D_k(i, j) = |x_ik − x_jk|;
   комбинированное: D_econ = Σ a_k D_k / Σ a_k (или евклидово: sqrt(Σ a_k D_k² / Σ a_k));
3. географическое: дороги / ж-д / по прямой (км, нормированы в [0, 1]) и смежность (0 — общая
   граница, 1 — нет); D_geo — взвешенная сумма компонентов;
4. D = α·D_econ + (1 − α)·D_geo (при α < 1 обе части приводятся к [0, 1] делением на максимум года);
5. вес ребра: w = exp(−D²/(2σ²)), σ — медиана ненулевых D, или w = 1/(1 + D);
6. прореживание: kNN (симметризованный или взаимный), порог по весу, kNN + минимальное остовное дерево.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from src.io import GEO, NETWORKS, PROCESSED, RAW, load_yaml
from src.normalize import prepare

EARTH_R = 6371.0


# ============================================================================ параметры
def default_params() -> dict:
    """configs/network.yaml + параметры цен (configs/indicators.yaml → params.prices, если в network.yaml их нет).

    Параметры цен входят в хэш сети: смена базового года, охвата дефляторов или межрегиональной
    поправки даёт другую сеть и другой каталог data/networks/{hash}/."""
    from src.prices import price_params

    p = load_yaml("network.yaml")
    p["prices"] = price_params(p.get("prices"))
    return p


def merge_params(base: dict, override: dict | None) -> dict:
    """Глубокое слияние словарей параметров (override поверх base)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "features":
            out[k] = merge_params(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(params: dict) -> str:
    """Короткий хэш набора параметров (детерминированный JSON с сортировкой ключей)."""
    s = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# ============================================================================ признаки
@lru_cache(maxsize=1)
def _indicators() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "indicators_wide.parquet")


@lru_cache(maxsize=1)
def _registry() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "indicator_registry.parquet")


def sample_rows(params: dict, wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """Строки indicators_wide выборки: действующие МО, годы, ФО/субъекты, исключение смен границ."""
    w = _indicators() if wide is None else wide
    s = params["sample"]
    y0, y1 = s["years"]
    m = w["valid_in_year"] & w["year"].between(y0, y1)
    if s.get("regions"):
        m &= w["region_code"].isin(s["regions"])
    elif s.get("federal_districts"):
        m &= w["federal_district"].isin(s["federal_districts"])
    if s.get("exclude_boundary_change"):
        m &= ~w["boundary_change"]
    return w[m]


def features(params: dict, wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """Нормированные признаки выборки, индекс (territory_id, year). Пропуски сохраняются.

    Денежные признаки сначала пересчитываются в цены базового года (params.prices, src/prices.py)."""
    rows = sample_rows(params, wide).set_index(["territory_id", "year"])
    cols = list(params["features"])
    missing = [c for c in cols if c not in rows]
    if missing:
        raise KeyError(f"Нет показателей: {missing}")
    pr = params.get("prices") or {}
    if pr.get("values", "real") == "real":
        from src.prices import to_real

        rows = to_real(
            rows, cols, pr["base_year"], pr.get("deflator_scope", "national"), bool(pr.get("spatial_price_adjustment"))
        )
    pp = params["preprocess"]
    logc = []
    if pp.get("log") == "registry":
        reg = _registry().set_index("code")
        logc = [c for c in cols if c in reg.index and bool(reg.at[c, "log"])]
        if pr.get("values", "real") == "real":
            # ln(1 + x) — в рублях опорного года: признаки не зависят от выбора базового года (src/prices.py)
            from src.prices import log_unit_scale

            rows = rows.copy()
            for c, u in log_unit_scale([c for c in logc], pr["base_year"]).items():
                rows[c] = rows[c] / u
    return prepare(
        rows,
        cols,
        log_columns=logc,
        method=pp["method"],
        scope=pp["scope"],
        winsor=tuple(pp["winsor"]) if pp.get("winsor") else None,
    )


# ============================================================================ расстояния
def pairwise_abs(x: np.ndarray) -> np.ndarray:
    """Матрица |x_i − x_j| для одного признака (D_k)."""
    return np.abs(x[:, None] - x[None, :])


def econ_distance(x: pd.DataFrame, weights: dict, metric: str = "weighted_l1") -> np.ndarray:
    """Комбинированное экономическое расстояние по нормированным признакам.

    weighted_l1: D = Σ_k a_k |x_ik − x_jk| / Σ a_k  (по признакам, известным у обоих МО);
    euclidean:   D = sqrt(Σ_k a_k (x_ik − x_jk)² / Σ a_k).
    Если у пары нет ни одного общего признака — NaN.
    """
    num = np.zeros((len(x), len(x)))
    den = np.zeros((len(x), len(x)))
    for c, a in weights.items():
        if a == 0:
            continue
        v = x[c].to_numpy(dtype=float)
        d = pairwise_abs(v)
        ok = ~np.isnan(d)
        num[ok] += a * (d[ok] if metric == "weighted_l1" else d[ok] ** 2)
        den[ok] += a
    with np.errstate(invalid="ignore", divide="ignore"):
        D = num / den
    if metric == "euclidean":
        D = np.sqrt(D)
    elif metric != "weighted_l1":
        raise ValueError(metric)
    np.fill_diagonal(D, 0.0)
    return D


@lru_cache(maxsize=4)
def _pairs(kind: str) -> pd.DataFrame:
    if kind == "road_km":
        c = pd.read_parquet(RAW / "sber/hackathonlicence/connection.parquet")
        return c[c["type"].eq("highway")][["territory_id_x", "territory_id_y", "distance"]]
    if kind == "rail_km":
        return pd.read_parquet(RAW / "sber/t_pairs_distance_railway/t_pairs_distance_railway.parquet")[
            ["territory_id_x", "territory_id_y", "distance"]
        ]
    if kind == "adjacency":
        return pd.read_parquet(GEO / "adjacency.parquet").assign(distance=0.0)
    raise ValueError(kind)


def _matrix_from_pairs(ids: np.ndarray, kind: str) -> np.ndarray:
    """Симметричная матрица по таблице пар (NaN — пары нет)."""
    pos = pd.Series(np.arange(len(ids)), index=ids)
    p = _pairs(kind)
    p = p[p["territory_id_x"].isin(pos.index) & p["territory_id_y"].isin(pos.index)]
    M = np.full((len(ids), len(ids)), np.nan)
    i, j = pos[p["territory_id_x"]].to_numpy(), pos[p["territory_id_y"]].to_numpy()
    M[i, j] = p["distance"].to_numpy()
    M[j, i] = p["distance"].to_numpy()
    np.fill_diagonal(M, 0.0)
    return M


def line_km(ids: np.ndarray) -> np.ndarray:
    """Расстояние по прямой (по дуге большого круга) между центрами МО, км."""
    mo = pd.read_parquet(PROCESSED / "mo.parquet").set_index("territory_id").loc[ids]
    lat, lon = np.radians(mo["lat"].to_numpy()), np.radians(mo["lon"].to_numpy())
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geo_distance(ids: np.ndarray, geo: dict) -> tuple[np.ndarray, dict]:
    """Географическое расстояние в [0, 1] и отчёт о правиле для МО без связи."""
    ws = {k: v for k, v in geo["weights"].items() if v}
    tot = sum(ws.values())
    D = np.zeros((len(ids), len(ids)))
    report = {}
    line = None
    adj = None
    for kind, a in ws.items():
        if kind == "line_km":
            M = line = line_km(ids) if line is None else line
        elif kind == "adjacency":
            M = _matrix_from_pairs(ids, "adjacency")
            M = np.where(np.isnan(M), 1.0, 0.0)
            np.fill_diagonal(M, 0.0)
            adj = M
        else:
            M = _matrix_from_pairs(ids, kind)
            miss = np.isnan(M)
            no_link = ids[miss.sum(axis=1) == len(ids) - 1]
            report[f"{kind}: МО без связи"] = [int(t) for t in no_link]
            rule = geo.get("no_road", "line")
            if rule == "line":
                line = line_km(ids) if line is None else line
                # приводим прямую к масштабу дорог: медиана отношения дорога/прямая
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.nanmedian((M / line)[(~miss) & (line > 0)])
                M = np.where(miss, line * ratio, M)
            elif rule == "adjacency":
                if adj is None:
                    adj = np.where(np.isnan(_matrix_from_pairs(ids, "adjacency")), 1.0, 0.0)
                    np.fill_diagonal(adj, 0.0)
                M = np.where(miss, adj * np.nanmax(M), M)
        mx = np.nanmax(M)
        M = M / mx if mx > 0 else M
        D += a * M
    D /= tot  # при no_road = exclude пары без связи остаются NaN → в combine берётся только D_econ
    return D, report


def combine(D_econ: np.ndarray, D_geo: np.ndarray | None, alpha: float) -> np.ndarray:
    """D = α·D_econ + (1 − α)·D_geo; где D_geo нет (правило exclude) — только D_econ."""
    if D_geo is None or alpha >= 1:
        return D_econ
    e = D_econ / np.nanmax(D_econ) if np.nanmax(D_econ) > 0 else D_econ
    out = alpha * e + (1 - alpha) * D_geo
    return np.where(np.isnan(D_geo), e, out)


def to_weight(D: np.ndarray, kind: str = "gaussian") -> np.ndarray:
    """Расстояние → вес ребра в (0, 1]."""
    if kind == "gaussian":
        nz = D[(D > 0) & np.isfinite(D)]
        sigma = np.median(nz) if len(nz) else 1.0
        return np.exp(-(D**2) / (2 * sigma**2))
    if kind == "inverse":
        return 1.0 / (1.0 + D)
    raise ValueError(kind)


# ============================================================================ прореживание
def knn_adjacency(D: np.ndarray, k: int, mode: str = "symmetric") -> np.ndarray:
    """Булева матрица рёбер kNN по расстоянию (без петель)."""
    n = len(D)
    k = min(k, n - 1)
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    Dm = np.where(np.isnan(Dm), np.inf, Dm)
    nn = np.argsort(Dm, axis=1, kind="stable")[:, :k]
    A = np.zeros((n, n), dtype=bool)
    A[np.repeat(np.arange(n), k), nn.ravel()] = True
    A &= np.isfinite(Dm)
    return (A | A.T) if mode == "symmetric" else (A & A.T)


def mst_adjacency(D: np.ndarray) -> np.ndarray:
    """Рёбра минимального остовного дерева (лес, если граф несвязен по конечным расстояниям)."""
    Dm = np.where(np.isfinite(D), D, 0.0)
    Dm = np.where((Dm == 0) & ~np.eye(len(D), dtype=bool) & np.isfinite(D), 1e-12, Dm)
    T = minimum_spanning_tree(Dm).toarray() > 0
    return T | T.T


def sparsify(D: np.ndarray, W: np.ndarray, sp: dict) -> np.ndarray:
    """Прореживание полной матрицы весов: kNN, порог веса или kNN + минимальное остовное дерево."""
    method = sp.get("method", "knn_mst")
    if method == "knn":
        return knn_adjacency(D, sp.get("k", 7), sp.get("knn_mode", "symmetric"))
    if method == "knn_mst":
        return knn_adjacency(D, sp.get("k", 7), sp.get("knn_mode", "symmetric")) | mst_adjacency(D)
    if method == "threshold":
        A = W >= sp.get("threshold", 0.5)
        np.fill_diagonal(A, False)
        return A
    raise ValueError(method)


# ============================================================================ сеть года
@dataclass
class YearNetwork:
    """Сеть одного года: узлы, матрица расстояний, рёбра и статистика."""

    year: int
    ids: np.ndarray
    X: pd.DataFrame
    D: np.ndarray
    W: np.ndarray
    A: np.ndarray
    dropped: list = field(default_factory=list)
    geo_report: dict = field(default_factory=dict)

    def edges(self) -> pd.DataFrame:
        """Рёбра сети: source, target, вес (сходство) и расстояние."""
        i, j = np.where(np.triu(self.A, 1))
        return pd.DataFrame(
            {"source": self.ids[i], "target": self.ids[j], "weight": self.W[i, j], "distance": self.D[i, j]}
        )

    def graph(self) -> nx.Graph:
        """Сеть как граф networkx (узлы — territory_id, у рёбер атрибуты weight и distance)."""
        g = nx.Graph()
        g.add_nodes_from(int(t) for t in self.ids)
        for r in self.edges().itertuples():
            g.add_edge(int(r.source), int(r.target), weight=float(r.weight), distance=float(r.distance))
        return g

    def stats(self) -> dict:
        """Сводка по сети года: узлы, рёбра, компоненты, степени."""
        n = len(self.ids)
        ncomp, lab = connected_components(self.A, directed=False)
        deg = self.A.sum(axis=1)
        sizes = np.bincount(lab)
        return {
            "year": int(self.year),
            "nodes": int(n),
            "edges": int(np.triu(self.A, 1).sum()),
            "components": int(ncomp),
            "largest_component": int(sizes.max()) if n else 0,
            "isolated": int((deg == 0).sum()),
            "degree_min": int(deg.min()) if n else 0,
            "degree_mean": float(deg.mean()) if n else 0.0,
            "degree_max": int(deg.max()) if n else 0,
            "dropped_missing": [int(t) for t in self.dropped],
            **self.geo_report,
        }


def structural_missing(X_year: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    """Признаки, которых в году нет у большинства МО (меньше threshold с данными) — «структурный» пропуск:
    например, специализации занятости по ОКВЭД2 до 2017 г. нет ни у одного МО. Такие признаки в этом году
    не участвуют в расстоянии (иначе из сети года выпали бы все МО)."""
    if not len(X_year):
        return []
    cov = X_year.notna().mean()
    return [c for c in X_year.columns if cov[c] < threshold]


def usable_rows(X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Строки (territory_id, year), где есть все признаки, кроме структурно отсутствующих в этом году.

    Для режима pooled: признаки, отсутствующие в году у всех МО, остаются NaN — такие МО-годы относятся к
    кластеру по имеющимся признакам (clustering.fit_pooled)."""
    keep = []
    for _, g in X.groupby(level="year", sort=False):
        skip = structural_missing(g, threshold)
        rest = [c for c in g.columns if c not in skip]
        keep.append(g[g[rest].notna().all(axis=1)] if rest else g.iloc[:0])
    return pd.concat(keep).reindex(columns=X.columns) if keep else X.iloc[:0]


def build_year(X_all: pd.DataFrame, year: int, params: dict) -> YearNetwork:
    """Сеть года по заранее нормированным признакам X_all (индекс territory_id, year)."""
    X = X_all.xs(year, level="year")
    skip = structural_missing(X, params.get("structural_missing", 0.5))
    if skip:
        X = X.drop(columns=skip)
        params = {**params, "features": {f: a for f, a in params["features"].items() if f not in skip}}
    dropped = []
    if params.get("missing", "drop") == "drop":
        bad = X.isna().any(axis=1)
        dropped = list(X.index[bad])
        X = X[~bad]
    X = X.sort_index()
    ids = X.index.to_numpy()
    De = econ_distance(X, params["features"], params.get("metric", "weighted_l1"))
    geo = params.get("geo", {})
    Dg, rep = (None, {})
    if geo.get("alpha", 1.0) < 1:
        Dg, rep = geo_distance(ids, geo)
    D = combine(De, Dg, geo.get("alpha", 1.0))
    D = np.where(np.isnan(D), np.inf, D)  # пары без общих признаков — не соседи
    np.fill_diagonal(D, 0.0)
    W = to_weight(np.where(np.isfinite(D), D, np.nan), params.get("edge_weight", "gaussian"))
    W = np.nan_to_num(W, nan=0.0)
    A = sparsify(D, W, params["sparsify"])
    return YearNetwork(year=year, ids=ids, X=X, D=D, W=W, A=A, dropped=dropped, geo_report=rep)


def build(params: dict | None = None, years: list[int] | None = None) -> dict[int, YearNetwork]:
    """Сети за все годы выборки."""
    p = params or default_params()
    X = features(p)
    ys = years or sorted(X.index.get_level_values("year").unique())
    return {y: build_year(X, y, p) for y in ys}


def save(nets: dict[int, YearNetwork], params: dict) -> str:
    """Сохраняет рёбра и параметры в data/networks/{hash}/; возвращает хэш."""
    h = config_hash(params)
    out = NETWORKS / h
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for y, net in nets.items():
        net.edges().to_parquet(out / f"edges_{y}.parquet", index=False)
        pd.DataFrame({"territory_id": net.ids}).to_parquet(out / f"nodes_{y}.parquet", index=False)
        stats.append(net.stats())
    (out / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return h


def load_edges(h: str, year: int) -> pd.DataFrame:
    """Рёбра сохранённой сети года: source, target, weight, distance."""
    return pd.read_parquet(NETWORKS / h / f"edges_{year}.parquet")
