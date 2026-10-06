"""Индексы качества кластеризации (ICVI): SW, CH, S_Dbw — в пространстве признаков;
AVI, AVU, ANUI, MQ — в пространстве сети.

Обозначения: X — нормированные признаки (n × p), labels — номера кластеров, A — симметричная
матрица весов рёбер сети (n × n, нули на диагонали), K — число кластеров.
Для сетевых индексов S_kl = Σ_{i∈k, j∈l} A_ij — сумма весов между кластерами k и l
(внутренние рёбра входят в S_kk дважды — матрица симметрична).

Направление «лучше» — в BETTER: +1 — чем больше, тем лучше; −1 — чем меньше, тем лучше.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

BETTER = {"SW": +1, "CH": +1, "DBI": -1, "S_Dbw": -1, "AVI": +1, "AVU": -1, "ANUI": +1, "MQ": +1}
FEATURE_INDICES = ("SW", "CH", "DBI", "S_Dbw")
NETWORK_INDICES = ("AVI", "AVU", "ANUI", "MQ")


def _k(labels: np.ndarray) -> int:
    return len(np.unique(labels))


# ============================================================================ признаки
def sw(X: np.ndarray, labels: np.ndarray) -> float:
    """Силуэт (Silhouette Width), больше — лучше, ∈ [−1, 1].

    s(i) = (b_i − a_i) / max(a_i, b_i), a_i — среднее расстояние до своего кластера,
    b_i — минимальное по чужим кластерам среднее расстояние; SW = среднее s(i).
    Источник: Rousseeuw P.J. Silhouettes: a graphical aid to the interpretation and validation of
    cluster analysis. J. Comput. Appl. Math. 20 (1987) 53–65. Реализация — scikit-learn (евклидова метрика).
    """
    return float(silhouette_score(X, labels)) if 1 < _k(labels) < len(X) else np.nan


def ch(X: np.ndarray, labels: np.ndarray) -> float:
    """Индекс Калински–Харабаша, больше — лучше.

    CH = [tr(B) / (K − 1)] / [tr(W) / (n − K)], B и W — межгрупповой и внутригрупповой разброс.
    Источник: Caliński T., Harabasz J. A dendrite method for cluster analysis.
    Communications in Statistics 3 (1974) 1–27. Реализация — scikit-learn.
    """
    return float(calinski_harabasz_score(X, labels)) if 1 < _k(labels) < len(X) else np.nan


def dbi(X: np.ndarray, labels: np.ndarray) -> float:
    """Индекс Дэвиса–Боулдина (дополнительно), меньше — лучше.

    Источник: Davies D.L., Bouldin D.W. A cluster separation measure. IEEE TPAMI 1(2) (1979) 224–227.
    """
    return float(davies_bouldin_score(X, labels)) if 1 < _k(labels) < len(X) else np.nan


def s_dbw(X: np.ndarray, labels: np.ndarray, sigma: str = "variance") -> float:
    """S_Dbw = Scat + Dens_bw, меньше — лучше.

    Scat — средняя «рассеянность» кластеров относительно всей выборки:
        Scat = (1/K) Σ_k ‖σ(c_k)‖ / ‖σ(X)‖, σ — вектор дисперсий признаков.
    Dens_bw — плотность «между» кластерами относительно плотности в их центрах:
        stdev = (1/K) · sqrt(Σ_k ‖σ(c_k)‖),
        dens(u) = число точек кластеров k и l в шаре радиуса stdev вокруг u,
        Dens_bw = 1/(K(K−1)) Σ_{k≠l} dens(u_kl) / max(dens(v_k), dens(v_l)),
        v_k — центр кластера, u_kl — середина отрезка между центрами.
    Источник: Halkidi M., Vazirgiannis M. Clustering validity assessment: finding the optimal
    partitioning of a data set. Proc. IEEE ICDM 2001, 187–194.

    sigma="variance" — как в статье (σ — вектор дисперсий; тогда радиус stdev имеет размерность
    расстояния). sigma="std" — вариант пакета s_dbw (PyPI, method="Halkidi"), где вместо дисперсий
    берутся стандартные отклонения; используется в тесте для сверки остальной логики.
    """
    labs = np.unique(labels)
    K = len(labs)
    if not 1 < K < len(X):
        return np.nan
    X = np.asarray(X, dtype=float)
    norm = lambda v: float(np.sqrt(v @ v))  # noqa: E731
    spread = np.var if sigma == "variance" else np.std
    sig_all = norm(spread(X, axis=0))
    centers, sig = [], []
    for c in labs:
        P = X[labels == c]
        centers.append(P.mean(axis=0))
        sig.append(norm(spread(P, axis=0)))
    scat = float(np.mean(sig)) / sig_all if sig_all > 0 else np.nan
    stdev = np.sqrt(np.sum(sig)) / K

    def dens(points: np.ndarray, u: np.ndarray) -> int:
        return int((np.linalg.norm(points - u, axis=1) <= stdev).sum())

    total = 0.0
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            P = X[(labels == labs[a]) | (labels == labs[b])]
            u = (centers[a] + centers[b]) / 2
            m = max(dens(P, centers[a]), dens(P, centers[b]))
            total += dens(P, u) / m if m > 0 else 0.0
    return scat + total / (K * (K - 1))


# ============================================================================ сеть
def block_sums(A: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Матрица S (K × K): суммы весов рёбер между кластерами (порядок — np.unique(labels))."""
    labs, inv = np.unique(labels, return_inverse=True)
    K = len(labs)
    M = np.zeros((len(labels), K))
    M[np.arange(len(labels)), inv] = 1.0
    return M.T @ A @ M


def avi(A: np.ndarray, labels: np.ndarray) -> float:
    """Average Isolability, больше — лучше, ∈ [0, 1].

    Изолированность кластера k: I_k = S_kk / Σ_l S_kl — доля весов рёбер кластера, которые
    остаются внутри него; AVI = (1/K) Σ_k I_k.
    Источник: определение по статье «Defining quality metrics for graph clustering
    evaluation», Expert Systems with Applications 71 (2017) 1–17; формула сверена с эталонной
    реализацией Shalileh S., github.com/Sorooshi/Pattern (metrics/clustering_metrics.py,
    коммит e5ff50b), см. tests/test_icvi.py.
    """
    S = block_sums(A, labels)
    row = S.sum(axis=1)
    iso = np.divide(np.diag(S), row, out=np.zeros(len(S)), where=row != 0)
    return float(iso.mean())


def avu(A: np.ndarray, labels: np.ndarray) -> float:
    """Average Unifiability, меньше — лучше.

    Склонность кластеров k и l к слиянию: U_kl = S_kl / (out_k + in_l − S_kl), где
    out_k = Σ_m S_km − S_kk (внешние связи k), in_l = Σ_m S_ml − S_ll (внешние связи l);
    AVU = (1/K) Σ_k Σ_{l≠k} U_kl. Источник и сверка — как у avi().
    """
    S = block_sums(A, labels)
    K = len(S)
    ext_out = S.sum(axis=1) - np.diag(S)
    ext_in = S.sum(axis=0) - np.diag(S)
    tot = 0.0
    for k in range(K):
        for m in range(K):
            if k == m:
                continue
            den = ext_out[k] + ext_in[m] - S[k, m]
            tot += S[k, m] / den if den != 0 else 0.0
    return tot / K


def anui(A: np.ndarray, labels: np.ndarray) -> float:
    """ANUI = 1 / (AVU + 1/AVI) — сводный индекс изолированности и неслияния, больше — лучше."""
    a, u = avi(A, labels), avu(A, labels)
    if a == 0:
        return 0.0
    return 1.0 / (u + 1.0 / a)


def mq(A: np.ndarray, labels: np.ndarray) -> float:
    """Модулярность Ньюмана (MQ) для взвешенного графа, больше — лучше, ∈ [−1/2, 1].

    Q = Σ_k [ S_kk / (2m) − (d_k / (2m))² ], 2m = Σ_ij A_ij, d_k — сумма взвешенных степеней кластера.
    Источник: Newman M.E.J., Girvan M. Finding and evaluating community structure in networks.
    Phys. Rev. E 69 (2004) 026113.
    """
    S = block_sums(A, labels)
    two_m = S.sum()
    if two_m == 0:
        return np.nan
    d = S.sum(axis=1)
    return float(np.sum(np.diag(S) / two_m - (d / two_m) ** 2))


# ============================================================================ всё сразу
def wcss(X: np.ndarray, labels: np.ndarray) -> float:
    """Внутрикластерная сумма квадратов (WCSS, инерция): Σ_k Σ_{i∈k} ‖x_i − c_k‖² по нормированным признакам.

    Для метода локтя: с ростом k всегда убывает; «локоть» — k, после которого убывание резко замедляется.
    Метки < 0 (нет метки) не учитываются. В BETTER не входит: само по себе всегда «лучше» большее k.
    """
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    m = labels >= 0
    X, labels = X[m], labels[m]
    return float(sum(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels)))


def elbow(ks, values) -> int | None:
    """k «локтя» кривой WCSS(k): точка, наиболее удалённая от прямой между первой и последней точкой кривой
    (обе оси приведены к [0, 1] — как в методе Kneedle). None — если точек меньше трёх или кривая плоская."""
    k = np.asarray(ks, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    k, v = k[ok], v[ok]
    if len(k) < 3 or np.ptp(v) == 0 or np.ptp(k) == 0:
        return None
    x = (k - k.min()) / np.ptp(k)
    y = (v - v.min()) / np.ptp(v)
    # расстояние до хорды (0, y0)–(1, y1): для убывающей выпуклой кривой точки лежат под хордой
    y0, y1 = y[0], y[-1]
    d = np.abs((y1 - y0) * x - y + y0) / np.hypot(y1 - y0, 1.0)
    return int(k[int(np.argmax(d))])


def compute_all(X: np.ndarray, A: np.ndarray | None, labels: np.ndarray) -> dict:
    """Все индексы для одного разбиения. A — веса рёбер разреженной сети (или None). WCSS — для метода локтя."""
    labels = np.asarray(labels)
    out = {"K": _k(labels), "SW": sw(X, labels), "CH": ch(X, labels), "DBI": dbi(X, labels), "S_Dbw": s_dbw(X, labels)}
    out["WCSS"] = wcss(X, labels)
    if A is not None:
        out.update({"AVI": avi(A, labels), "AVU": avu(A, labels), "ANUI": anui(A, labels), "MQ": mq(A, labels)})
    return out
