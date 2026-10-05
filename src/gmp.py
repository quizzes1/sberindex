"""Валовой муниципальный продукт (ВМП) — расчётная оценка команды.

Идея: валовая добавленная стоимость (ВДС) субъекта по каждой отрасли распределяется между
его МО пропорционально «индикатору распределения» этой отрасли и суммируется по отраслям.
Сумма ВМП по МО субъекта равна ВРП субъекта по построению (по МО, где есть ФОТ).

Все методы имеют один интерфейс:
    method(data: GMPInputs, year: int, params: dict) -> DataFrame
        territory_id, region_code, year, gmp, gmp_imputed, [sec_<k> — вклад раздела k]
Единицы: gmp — тыс. руб. (как ВДС у Росстата), gmp_pc — руб. на жителя.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.regional import grp, gva_okved2

GVA_SECTIONS = [*"ABCDEFGHIJK", "L1", "L2", *"MNOPQRST"]


@dataclass
class GMPInputs:
    """Входные данные: панель МО (только действующие в году строки) и региональные ряды."""

    mo: (
        pd.DataFrame
    )  # territory_id, region_code, year, pop, payroll, payroll__K, shipped__K, agri_output, anomaly_sector_sum
    grp: pd.DataFrame  # region_code, year, grp (млн руб.)
    gva: pd.DataFrame  # region_code, year, section, gva (тыс. руб.)

    @classmethod
    def load(cls, wide: pd.DataFrame, mo_table: pd.DataFrame) -> GMPInputs:
        """Собирает входы из panel_wide и mo.parquet."""
        w = wide[wide["valid_in_year"]].merge(mo_table[["territory_id", "region_code"]], on="territory_id")
        # ФОТ: если ряда нет, но есть работники и зарплата — L × W × 12 (тыс. руб.)
        lw = w["workers"] * w["wage"] * 12 / 1000
        w["payroll_src"] = np.where(w["payroll"].notna(), "payroll", np.where(lw.notna(), "L*W*12", None))
        w["payroll"] = w["payroll"].fillna(lw)
        return cls(mo=w, grp=grp(), gva=gva_okved2())


def _payroll_frame(g: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Итоговый ФОТ и ФОТ по разделам ОКВЭД2 для МО региона (МО-годы с аномалией — без разделов)."""
    total = g["payroll"]
    sec = pd.DataFrame({k: g.get(f"payroll__{k}") for k in "ABCDEFGHIJKLMNOPQRS"}, index=g.index)
    sec.loc[g["anomaly_sector_sum"].fillna(False).astype(bool)] = np.nan
    return total, sec


def basic(data: GMPInputs, year: int, params: dict | None = None) -> pd.DataFrame:
    """Метод 1: ВМП_i = ВРП_r × ФОТ_i / Σ_j ФОТ_j (по МО с известным ФОТ)."""
    m = data.mo[data.mo["year"].eq(year) & data.mo["payroll"].gt(0)].copy()
    g = data.grp[data.grp["year"].eq(year)].set_index("region_code")["grp"] * 1000  # тыс. руб.
    m["gmp"] = m["region_code"].map(g) * m["payroll"] / m.groupby("region_code")["payroll"].transform("sum")
    m["gmp_imputed"] = 0.0
    return m[["territory_id", "region_code", "year", "gmp", "gmp_imputed"]].dropna(subset=["gmp"])


def _indicator(g: pd.DataFrame, kind: str, section: str, payroll_sec: pd.DataFrame) -> pd.Series:
    letter = "L" if section in ("L1", "L2") else section
    if kind == "payroll":
        return payroll_sec[letter] if letter in payroll_sec else pd.Series(np.nan, index=g.index)
    if kind == "shipped":
        return g.get(f"shipped__{letter}", pd.Series(np.nan, index=g.index))
    if kind == "agri":
        return g["agri_output"]
    if kind == "pop":
        return g["pop"]
    raise ValueError(kind)


def sectoral(data: GMPInputs, year: int, params: dict) -> pd.DataFrame:
    """Метод 2 (основной): ВМП_i = Σ_k ВДС_{r,k} × s_ik с учётом скрытых значений.

    Для раздела k региона r:
      φ_r       — доля ФОТ региона, не разложенная по опубликованным разделам;
      (1−φ_r)·ВДС_rk распределяется по известным X_ik (индикатор из params['by_section']);
      φ_r·ВДС_rk — по весу imputation_weight (по умолчанию «остаточный» ФОТ МО:
                   итог минус сумма опубликованных разделов) → это и есть gmp_imputed.
    Для индикатора pop (нет скрытия) φ = 0. Если у индикатора в регионе нет данных ни у одного
    МО — fallback (payroll); если нет и его — весь ВДС раздела распределяется по весу
    imputation_weight и целиком считается импутированным.
    """
    by = {**params.get("by_section", {})}
    default, fallback = params.get("default", "payroll"), params.get("fallback", "payroll")
    wmode = params.get("imputation_weight", "residual_payroll")
    m = data.mo[data.mo["year"].eq(year) & data.mo["payroll"].gt(0)]
    gva = data.gva[data.gva["year"].eq(year)].pivot_table(index="region_code", columns="section", values="gva")
    out = []
    for r, g in m.groupby("region_code"):
        if r not in gva.index:
            continue
        g = g.set_index("territory_id")
        total, psec = _payroll_frame(g)
        known = psec.sum(axis=1, min_count=1).fillna(0)
        resid = (total - known).clip(lower=0)
        phi = float(resid.sum() / total.sum()) if total.sum() > 0 else 0.0
        weight = {"residual_payroll": resid, "payroll": total, "pop": g["pop"]}[wmode].fillna(0)
        if weight.sum() <= 0:
            weight, phi = total, 0.0
        res = pd.DataFrame(index=g.index)
        imputed = pd.Series(0.0, index=g.index)
        for k in GVA_SECTIONS:
            v = gva.at[r, k] if k in gva.columns else np.nan
            if not np.isfinite(v) or v == 0:
                res[f"sec_{k}"] = 0.0
                continue
            kind = by.get(k, default)
            x = _indicator(g, kind, k, psec).fillna(0).clip(lower=0)
            if x.sum() <= 0 and kind != fallback:
                kind = fallback
                x = _indicator(g, kind, k, psec).fillna(0).clip(lower=0)
            if x.sum() <= 0:
                part_known, part_imp = 0.0 * x, v * weight / weight.sum()
            else:
                f = 0.0 if kind == "pop" else phi
                part_known = (1 - f) * v * x / x.sum()
                part_imp = f * v * weight / weight.sum()
            res[f"sec_{k}"] = part_known + part_imp
            imputed += part_imp
        res["gmp"] = res.sum(axis=1)
        res["gmp_imputed"] = imputed
        res["region_code"] = r
        out.append(res.reset_index())
    d = pd.concat(out, ignore_index=True)
    d["year"] = year
    return d


METHODS = {"basic": basic, "sectoral": sectoral}


def compute(data: GMPInputs, method: str, years: list[int], params: dict | None = None) -> pd.DataFrame:
    """ВМП выбранным методом за годы years (одна строка на МО × год)."""
    f = METHODS[method]
    return pd.concat([f(data, y, params or {}) for y in years], ignore_index=True).assign(method=method)


def structure(d: pd.DataFrame) -> pd.DataFrame:
    """Отраслевая структура ВМП (метод 2): доля раздела k в ВМП МО (L1+L2 → L)."""
    s = d.copy()
    s["sec_L"] = s.pop("sec_L1") + s.pop("sec_L2")
    cols = [c for c in s.columns if c.startswith("sec_")]
    long = s.melt(id_vars=["territory_id", "year"], value_vars=cols, var_name="section", value_name="value")
    long["section"] = long["section"].str[4:]
    tot = s.set_index(["territory_id", "year"])["gmp"]
    long["share"] = long["value"] / long.set_index(["territory_id", "year"]).index.map(tot)
    return long[["territory_id", "year", "section", "value", "share"]]
