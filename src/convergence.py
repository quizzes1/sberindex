"""Конвергенция муниципалитетов (этап 6).

Вход — «длинный» ряд показателя y_it > 0 в реальном выражении (МО i, год t).

- σ-конвергенция: разброс ln y по МО в каждом году (стандартное отклонение и коэффициент
  вариации) и его тренд: регрессия σ_t = a + c·t (c < 0 — разброс сокращается).
- β-конвергенция абсолютная: (ln y_iT − ln y_i0)/T = a + b·ln y_i0 + ε; b < 0 — бедные растут
  быстрее; скорость λ = −ln(1 + bT)/T, период полусхождения ln 2 / λ.
- β-конвергенция условная (панельная): Δln y_it = α_i + δ_t + b·ln y_i,t−1 + ε_it с фиксированными
  эффектами МО и года, стандартные ошибки кластеризованы по субъекту. На коротких панелях
  оценка b смещена вниз (смещение Никелла) — сравнивать лучше знаки и порядки, чем точные значения.
- клубная конвергенция (Phillips, Sul 2007): тест log t и алгоритм выделения «клубов».
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _log_panel(df: pd.DataFrame) -> pd.DataFrame:
    """df: territory_id, year, value → таблица ln y (строки — МО, столбцы — годы), y ≤ 0 → NaN."""
    p = df.pivot_table(index="territory_id", columns="year", values="value")
    return np.log(p.where(p > 0))


def sigma(df: pd.DataFrame, balanced: bool = False) -> tuple[pd.DataFrame, dict]:
    """σ-конвергенция. Возвращает (по годам: n, sd_log, cv; тренд σ_t: наклон, ст. ошибка, p-value).

    balanced=True — только МО с данными во все годы (иначе состав выборки может «рисовать» тренд).
    """
    L = _log_panel(df)
    if balanced:
        L = L.dropna()
    y = np.exp(L)
    t = pd.DataFrame({"n": L.notna().sum(), "sd_log": L.std(ddof=1), "cv": y.std(ddof=1) / y.mean()})
    t.index.name = "year"
    s = t["sd_log"].dropna()
    trend = {"slope": np.nan, "se": np.nan, "p": np.nan, "years": len(s)}
    if len(s) >= 3:
        X = sm.add_constant(s.index.to_numpy(dtype=float))
        r = sm.OLS(s.to_numpy(), X).fit()
        trend = {
            "slope": float(r.params[1]),
            "se": float(r.bse[1]),
            "p": float(r.pvalues[1]),
            "years": len(s),
            "sd_first": float(s.iloc[0]),
            "sd_last": float(s.iloc[-1]),
        }
    return t.reset_index(), trend


def beta_absolute(df: pd.DataFrame, y0: int | None = None, y1: int | None = None) -> dict:
    """Абсолютная β-конвергенция за окно [y0, y1] по МО с данными в оба года (робастные ошибки HC1)."""
    L = _log_panel(df)
    y0 = y0 or int(L.columns.min())
    y1 = y1 or int(L.columns.max())
    T = y1 - y0
    d = L[[y0, y1]].dropna()
    out = {"y0": y0, "y1": y1, "T": T, "n": len(d)}
    if len(d) < 5 or T <= 0:
        return out | {"b": np.nan}
    g = (d[y1] - d[y0]) / T
    r = sm.OLS(g.to_numpy(), sm.add_constant(d[y0].to_numpy())).fit(cov_type="HC1")
    b = float(r.params[1])
    lam = -np.log(1 + b * T) / T if 1 + b * T > 0 else np.nan
    return out | {
        "a": float(r.params[0]),
        "b": b,
        "se": float(r.bse[1]),
        "p": float(r.pvalues[1]),
        "r2": float(r.rsquared),
        "lambda": lam,
        "half_life": float(np.log(2) / lam) if lam and lam > 0 else np.nan,
        "growth": g,
        "level0": d[y0],
    }


def beta_panel(df: pd.DataFrame, region: pd.Series) -> dict:
    """Условная β-конвергенция: Δln y_it на ln y_i,t−1 с эффектами МО и года (within-оценка),
    стандартные ошибки кластеризованы по субъекту (region: territory_id → region_code)."""
    L = _log_panel(df)
    long = L.stack().rename("ly").reset_index().dropna(subset=["ly"])  # pandas 3: stack сохраняет NaN
    long = long.sort_values(["territory_id", "year"])
    long["lag"] = long.groupby("territory_id")["ly"].shift(1)
    long["lag_year"] = long.groupby("territory_id")["year"].shift(1)
    long = long[(long["year"] - long["lag_year"]).eq(1)].dropna(subset=["lag"])
    long["dly"] = long["ly"] - long["lag"]
    if long["territory_id"].nunique() < 5 or long["year"].nunique() < 2:
        return {"b": np.nan, "n_obs": len(long)}
    # двусторонняя within-трансформация (несбалансированная панель → итеративное снятие средних)
    y = long["dly"].to_numpy(float).copy()
    x = long["lag"].to_numpy(float).copy()
    gi = long["territory_id"].to_numpy()
    gt = long["year"].to_numpy()
    for _ in range(100):
        y0, x0 = y.copy(), x.copy()
        for g in (gi, gt):
            y -= pd.Series(y).groupby(g).transform("mean").to_numpy()
            x -= pd.Series(x).groupby(g).transform("mean").to_numpy()
        if np.abs(y - y0).max() < 1e-10 and np.abs(x - x0).max() < 1e-10:
            break
    cl = long["territory_id"].map(region).fillna(-1).to_numpy()
    r = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(cl)[0]})
    b = float(r.params[0])
    return {
        "b": b,
        "se": float(r.bse[0]),
        "p": float(r.pvalues[0]),
        "n_obs": len(long),
        "n_mo": int(long["territory_id"].nunique()),
        "n_clusters": int(len(np.unique(cl))),
        "lambda": float(-np.log(1 + b)) if 1 + b > 0 else np.nan,
    }


# ============================================================================ Phillips–Sul
def log_t(L: pd.DataFrame, trim: float = 0.3, hp_lambda: float = 400) -> dict:
    """Тест log t (Phillips, Sul 2007, Econometrica 75(6)).

    h_it = X_it / mean_i X_it — относительный путь (X — трендовая компонента ln y после фильтра
    Ходрика–Прескотта); H_t = mean_i (h_it − 1)²; регрессия
    log(H_1/H_t) − 2·log(log t) = a + b·log t по t = [rT]…T, ошибки HAC.
    Нулевая гипотеза — конвергенция; отвергается, если t-статистика b < −1,65.
    b/2 — скорость сближения (b ≥ 2 — сходимость уровней).
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter

    L = L.dropna()
    if len(L) < 3 or L.shape[1] < 5:
        return {"b": np.nan, "t": np.nan, "n": len(L)}
    X = np.vstack([hpfilter(row, lamb=hp_lambda)[1] for row in L.to_numpy()])
    h = X / X.mean(axis=0)
    H = ((h - 1) ** 2).mean(axis=0)
    T = X.shape[1]
    t = np.arange(1, T + 1)
    start = max(int(np.floor(trim * T)), 1)
    tt = t[start:]
    with np.errstate(divide="ignore", invalid="ignore"):
        yv = np.log(H[0] / H[start:]) - 2 * np.log(np.log(tt))
    ok = np.isfinite(yv)
    if ok.sum() < 3:
        return {"b": np.nan, "t": np.nan, "n": len(L)}
    r = sm.OLS(yv[ok], sm.add_constant(np.log(tt[ok]))).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return {"b": float(r.params[1]), "t": float(r.tvalues[1]), "n": len(L), "points": int(ok.sum())}


def clubs(L: pd.DataFrame, trim: float = 0.3, hp_lambda: float = 400, crit: float = -1.65) -> pd.Series:
    """Клубы конвергенции по алгоритму Phillips–Sul (упрощённая версия без слияния клубов).

    1) МО сортируются по последнему году по убыванию; 2) ядро — первые k МО с максимальной
    t-статистикой log t при t > crit (k ≥ 2); 3) к ядру добавляются МО, с которыми log t остаётся
    > 0 (консервативное правило); 4) остаток обрабатывается так же; МО, не вошедшие ни в один клуб,
    получают −1 (расходящиеся). Возвращает номер клуба (1, 2, …) для каждого МО.
    """
    L = L.dropna()
    order = L.iloc[:, -1].sort_values(ascending=False).index.tolist()
    club = pd.Series(-1, index=L.index)
    cid = 1
    rest = order
    while len(rest) >= 2:
        best_k, best_t = None, -np.inf
        for k in range(2, len(rest) + 1):
            r = log_t(L.loc[rest[:k]], trim, hp_lambda)
            if np.isfinite(r["t"]) and r["t"] > crit and r["t"] > best_t:
                best_k, best_t = k, r["t"]
            elif best_k is not None and np.isfinite(r["t"]) and r["t"] < crit:
                break
        if best_k is None:
            rest = rest[1:]  # первый МО не сходится ни с кем — расходящийся
            continue
        core = rest[:best_k]
        members = list(core)
        for i in rest[best_k:]:
            r = log_t(L.loc[core + [i]], trim, hp_lambda)
            if np.isfinite(r["t"]) and r["t"] > 0:
                members.append(i)
        club.loc[members] = cid
        cid += 1
        rest = [i for i in rest if i not in members]
    return club
