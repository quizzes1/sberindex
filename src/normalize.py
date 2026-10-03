"""Предобработка признаков перед расстояниями и кластеризацией.

Порядок: (1) логарифм для скошенных показателей (по реестру), (2) винзоризация (опция,
по умолчанию выключена — крупные города не «обрезаем»), (3) нормировка min-max в [0, 1]
или z-score.

Область нормировки (scope):
  "panel" (по умолчанию) — параметры (min/max или среднее/σ) считаются по всем годам сразу;
           значения разных лет сопоставимы, и движение МО между кластерами отражает изменение
           самих показателей, а не смену шкалы;
  "year"  — параметры считаются отдельно для каждого года: каждый год сравнивается сам с собой,
           общий рост (например, номинальной зарплаты) исчезает, и «динамика кластеров»
           становится в значительной мере артефактом нормировки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("minmax", "zscore")
SCOPES = ("panel", "year")


def log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """ln(1 + x) для неотрицательных столбцов; при отрицательных значениях — знаковый ln(1 + |x|)."""
    out = df.copy()
    for c in columns:
        if c in out:
            x = out[c].astype(float)
            out[c] = np.sign(x) * np.log1p(np.abs(x))
    return out


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99, by: pd.Series | None = None) -> pd.DataFrame:
    """Обрезка хвостов по квантилям lower/upper (по всей выборке или внутри групп by)."""
    out = df.copy()
    num = out.select_dtypes("number").columns
    if by is None:
        lo, hi = out[num].quantile(lower), out[num].quantile(upper)
        out[num] = out[num].clip(lo, hi, axis=1)
        return out
    for _, idx in out.groupby(by).groups.items():
        part = out.loc[idx, num]
        out.loc[idx, num] = part.clip(part.quantile(lower), part.quantile(upper), axis=1)
    return out


def _scale(x: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "minmax":
        lo, hi = x.min(), x.max()
        rng = (hi - lo).replace(0, np.nan)
        return ((x - lo) / rng).fillna(0.0).where(x.notna())
    if method == "zscore":
        sd = x.std(ddof=0).replace(0, np.nan)
        return ((x - x.mean()) / sd).fillna(0.0).where(x.notna())
    raise ValueError(f"method должен быть одним из {METHODS}")


def normalize(
    df: pd.DataFrame, method: str = "minmax", scope: str = "panel", year: pd.Series | None = None
) -> pd.DataFrame:
    """Нормировка числовых столбцов.

    method: "minmax" → [0, 1]; "zscore" → среднее 0, σ = 1 (на области нормировки).
    scope: "panel" — по всем строкам сразу; "year" — внутри каждого года (нужен year).
    Пропуски остаются пропусками. Постоянный столбец превращается в 0.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope должен быть одним из {SCOPES}")
    num = df.select_dtypes("number").columns
    out = df.copy()
    if scope == "panel":
        out[num] = _scale(df[num].astype(float), method)
        return out
    if year is None:
        year = df.index.get_level_values("year") if "year" in df.index.names else df["year"]
    year = pd.Series(np.asarray(year), index=df.index)
    for _, idx in df.groupby(year).groups.items():
        out.loc[idx, num] = _scale(df.loc[idx, num].astype(float), method)
    return out


def prepare(
    df: pd.DataFrame,
    columns: list[str],
    log_columns: list[str] | None = None,
    method: str = "minmax",
    scope: str = "panel",
    winsor: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Полная предобработка: выбор столбцов → логарифм → (винзоризация) → нормировка.

    df индексирован (territory_id, year). Возвращает таблицу тех же строк и columns.
    """
    x = df[columns].astype(float)
    x = log_transform(x, [c for c in (log_columns or []) if c in columns])
    if winsor:
        x = winsorize(x, *winsor)
    return normalize(x, method=method, scope=scope)
