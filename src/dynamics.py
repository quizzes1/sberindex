"""Динамика кластеров во времени.

- сопоставление кластеров соседних лет: матрица пересечений по составу МО, мера Жаккара,
  оптимальное сопоставление венгерским алгоритмом → «сквозные» номера кластеров;
- матрица переходов год → год, доля МО, сменивших тип, список «мигрантов»;
- устойчивость: ARI между разбиениями соседних лет и бутстрэп по узлам.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def overlap(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Матрица пересечений: число общих МО у кластера a_i (строки) и b_j (столбцы)."""
    common = a.index.intersection(b.index)
    return pd.crosstab(a.loc[common], b.loc[common])


def jaccard(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Мера Жаккара |A ∩ B| / |A ∪ B| для всех пар кластеров (по всем МО каждого года)."""
    ov = overlap(a, b)
    sa = a.value_counts()
    sb = b.value_counts()
    J = ov.copy().astype(float)
    for i in ov.index:
        for j in ov.columns:
            J.loc[i, j] = ov.loc[i, j] / (sa[i] + sb[j] - ov.loc[i, j])
    return J


def match_years(labels: pd.DataFrame, min_jaccard: float = 0.2) -> pd.DataFrame:
    """Сквозные номера кластеров.

    labels: territory_id, year, label. Первый год: сквозной номер = метка. Для каждого
    следующего года кластеры сопоставляются с кластерами предыдущего года венгерским
    алгоритмом (максимум суммарной меры Жаккара); пара принимается, если Жаккар ≥ min_jaccard,
    иначе кластер получает новый сквозной номер («новый тип»).
    Возвращает labels + колонки through (сквозной номер) и jaccard (с кем сопоставлен).
    """
    years = sorted(labels["year"].unique())
    out = []
    first = labels[labels["year"].eq(years[0])].copy()
    first["through"] = first["label"]
    first["jaccard"] = np.nan
    out.append(first)
    next_id = int(first["through"].max()) + 1
    prev = first.set_index("territory_id")["through"]
    for y in years[1:]:
        cur = labels[labels["year"].eq(y)].copy()
        c = cur.set_index("territory_id")["label"]
        J = jaccard(c, prev)  # строки — текущие кластеры, столбцы — сквозные номера прошлого года
        r, k = linear_sum_assignment(-J.to_numpy())
        mapping, jac = {}, {}
        for i, j in zip(r, k):
            if J.iat[i, j] >= min_jaccard:
                mapping[J.index[i]] = J.columns[j]
                jac[J.index[i]] = J.iat[i, j]
        for lab in sorted(c.unique()):
            if lab not in mapping:
                mapping[lab] = next_id
                jac[lab] = np.nan
                next_id += 1
        cur["through"] = cur["label"].map(mapping)
        cur["jaccard"] = cur["label"].map(jac)
        out.append(cur)
        prev = cur.set_index("territory_id")["through"]
    return pd.concat(out, ignore_index=True)


def through_labels(
    labels: pd.DataFrame,
    mode: str,
    min_jaccard: float = 0.2,
    price_params: dict | None = None,
    order: bool = True,
    numbering: str = "rank",
) -> pd.DataFrame:
    """Сквозные типы K1…Kn: territory_id, year, label, through (0 → K1), jaccard.

    pooled — одна модель на все годы: метка уже сквозная (through = label), номера — по медиане показателя
    упорядочения (configs/clustering.yaml → order_by, по умолчанию ВМП на душу в ценах базового года) по всей панели.
    per_year, numbering = rank (по умолчанию) — номер кластера в каждом году — его место по тому же показателю
    внутри года (K1 — самый высокий), как на страницах «Кластеры» и «Сводные таблицы»: один и тот же кластер везде
    называется одинаково, типов ровно k; переход K2 → K3 — переход к кластеру с более низким ВМП.
    per_year, numbering = jaccard — сопоставление соседних лет по мере Жаккара (match_years): номер сохраняется,
    если кластер следующего года похож на кластер прошлого; могут появляться новые номера (больше k).
    """
    from src.clustering import order_labels

    if mode == "pooled":
        th = labels.assign(through=labels["label"], jaccard=np.nan)
    elif numbering == "rank":
        th = labels.assign(through=labels["label"], jaccard=np.nan)
        if order:
            parts = []
            for y, g in th.groupby("year", sort=True):
                s = g.set_index("territory_id")["through"]
                parts.append(g.assign(through=order_labels(s, price_params, int(y)).to_numpy()))
            return pd.concat(parts).sort_index().reset_index(drop=True)
        return th.reset_index(drop=True)
    else:
        th = match_years(labels, min_jaccard)
    if order:
        s = th.set_index(["territory_id", "year"])["through"]
        th["through"] = order_labels(s, price_params).to_numpy()
    return th.reset_index(drop=True)


def transitions(through: pd.DataFrame) -> pd.DataFrame:
    """Переходы между соседними годами: year_from, year_to, from, to, n (число МО)."""
    years = sorted(through["year"].unique())
    rows = []
    s = through.set_index(["year", "territory_id"])["through"]
    for y0, y1 in zip(years[:-1], years[1:]):
        a, b = s.loc[y0], s.loc[y1]
        common = a.index.intersection(b.index)
        t = pd.crosstab(a.loc[common], b.loc[common])
        for i in t.index:
            for j in t.columns:
                if t.loc[i, j]:
                    rows.append({"year_from": y0, "year_to": y1, "from": int(i), "to": int(j), "n": int(t.loc[i, j])})
    return pd.DataFrame(rows)


def migrants(through: pd.DataFrame) -> pd.DataFrame:
    """МО, сменившие сквозной тип между соседними годами: territory_id, year_from, year_to, from, to."""
    t = through.sort_values(["territory_id", "year"])
    t["prev"] = t.groupby("territory_id")["through"].shift(1)
    t["prev_year"] = t.groupby("territory_id")["year"].shift(1)
    m = t[t["prev"].notna() & t["prev"].ne(t["through"]) & (t["year"] - t["prev_year"]).eq(1)]
    return m.rename(columns={"prev_year": "year_from", "year": "year_to", "prev": "from", "through": "to"})[
        ["territory_id", "year_from", "year_to", "from", "to"]
    ].astype({"year_from": int, "from": int, "to": int})


def year_summary(through: pd.DataFrame) -> pd.DataFrame:
    """По парам соседних лет: ARI, доля МО, сменивших тип, число новых и исчезнувших типов."""
    years = sorted(through["year"].unique())
    s = through.set_index(["year", "territory_id"])
    rows = []
    for y0, y1 in zip(years[:-1], years[1:]):
        a, b = s.loc[y0], s.loc[y1]
        common = a.index.intersection(b.index)
        ta, tb = a.loc[common, "through"], b.loc[common, "through"]
        rows.append(
            {
                "year_from": y0,
                "year_to": y1,
                "МО в обоих годах": len(common),
                "ARI": adjusted_rand_score(a.loc[common, "label"], b.loc[common, "label"]),
                "доля сменивших тип": float((ta != tb).mean()),
                "новых типов": len(set(tb) - set(a["through"])),
                "исчезло типов": len(set(ta) - set(b["through"])),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_stability(
    fit_fn, X: np.ndarray, W: np.ndarray | None, base: np.ndarray, n: int = 50, frac: float = 1.0, seed: int = 42
) -> np.ndarray:
    """Устойчивость разбиения к составу выборки: n раз берём узлы с возвращением, заново
    кластеризуем и считаем ARI с исходным разбиением на попавших в выборку (уникальных) узлах.

    fit_fn(X_sub, W_sub) -> labels. Возвращает массив ARI длины n.
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    out = []
    for _ in range(n):
        idx = np.unique(rng.choice(N, size=int(frac * N), replace=True))
        Wsub = W[np.ix_(idx, idx)] if W is not None else None
        lab = fit_fn(X[idx], Wsub)
        out.append(adjusted_rand_score(base[idx], lab))
    return np.asarray(out)
