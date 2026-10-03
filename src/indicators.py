"""Расчёт показателей из реестра configs/indicators.yaml (этап 3).

Каждый показатель — функция с именем его кода: f(ctx) -> Series (индекс territory_id, year) или
DataFrame (отраслевые показатели: колонки <код>_<раздел>). Добавить показатель = запись в YAML +
функция здесь. Деление на ноль и отсутствующие знаменатели дают NaN (пропуск ≠ ноль).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.io import PROCESSED, load_yaml
from src.regional import regions_collection

OKVED2 = list("ABCDEFGHIJKLMNOPQRS")
SPEND_CATS = ["food", "marketplaces", "transport", "health", "catering", "other"]


def registry() -> list[dict]:
    """Записи реестра показателей."""
    return load_yaml("indicators.yaml")["indicators"]


def registry_params() -> dict:
    return load_yaml("indicators.yaml")["params"]


@dataclass
class Context:
    """Всё, что нужно функциям показателей. Таблицы индексированы (territory_id, year)."""

    wide: pd.DataFrame
    mo: pd.DataFrame
    gmp: pd.DataFrame
    structure: pd.DataFrame
    prices: pd.DataFrame
    params: dict = field(default_factory=dict)

    @classmethod
    def load(cls, params: dict | None = None) -> Context:
        p = {**registry_params(), **(params or {})}
        wide = pd.read_parquet(PROCESSED / "panel_wide.parquet").set_index(["territory_id", "year"]).sort_index()
        mo = pd.read_parquet(PROCESSED / "mo.parquet")
        gmp = pd.read_parquet(PROCESSED / "gmp.parquet")
        st = pd.read_parquet(PROCESSED / "gmp_structure.parquet")
        return cls(wide=wide, mo=mo, gmp=gmp, structure=st, prices=regional_prices(p["base_year"]), params=p)

    def col(self, name: str) -> pd.Series:
        return self.wide[name] if name in self.wide else pd.Series(np.nan, index=self.wide.index)

    def region(self) -> pd.Series:
        """region_code для каждой строки панели."""
        rc = self.mo.set_index("territory_id")["region_code"]
        return pd.Series(self.wide.index.get_level_values("territory_id").map(rc), index=self.wide.index)

    def by_region(self, s: pd.Series) -> pd.Series:
        """Региональный ряд (индекс region_code, year) → значения для строк панели."""
        idx = list(zip(self.region(), self.wide.index.get_level_values("year")))
        return pd.Series(s.reindex(idx).values, index=self.wide.index)


def regional_prices(base_year: int) -> pd.DataFrame:
    """ИПЦ-уровень цен субъекта (среднегодовой, = 1 в базовом году) и стоимость фиксированного набора.

    ИПЦ у Росстата — декабрь к декабрю; уровень на декабрь — цепное произведение, среднегодовой
    уровень приближается средним геометрическим уровней на декабрь прошлого и текущего года.
    """
    d = regions_collection(["Y477110111", "Y477110395"]).dropna(subset=["region_code"])
    cpi = d[d["indicator_code"].eq("Y477110111")].pivot_table(
        index="year", columns="region_code", values="indicator_value"
    )
    dec = (cpi / 100).cumprod()
    avg = np.sqrt(dec * dec.shift(1))
    avg = avg / avg.loc[base_year]
    basket = d[d["indicator_code"].eq("Y477110395") & d["indicator_unit"].eq("Рублей")].pivot_table(
        index="year", columns="region_code", values="indicator_value"
    )
    out = pd.DataFrame({"cpi_level": avg.stack(), "basket": basket.stack()})
    out.index = out.index.set_names(["year", "region_code"])
    return out.reset_index()


def _per_capita(ctx: Context, col: str, scale: float = 1000.0) -> pd.Series:
    """Ряд в тыс. руб. → руб. на жителя."""
    return ctx.col(col) * scale / ctx.col("pop")


# ============================================================================ экономика
def _gmp_wide(ctx: Context, col: str) -> pd.DataFrame:
    g = ctx.gmp.pivot_table(index=["territory_id", "year"], columns="method", values=col)
    return g.reindex(ctx.wide.index)


def _gmp_choose(ctx: Context, col: str) -> pd.Series:
    g = _gmp_wide(ctx, col)
    m = ctx.params.get("gmp_method", "default")
    if m == "default":
        return g.get("sectoral", np.nan).fillna(g.get("basic"))
    return g[m]


def gmp_method_used(ctx: Context) -> pd.Series:
    """Каким методом посчитан gmp_pc в этой строке (флаг для интерфейса)."""
    g = _gmp_wide(ctx, "gmp_pc")
    m = ctx.params.get("gmp_method", "default")
    if m != "default":
        return pd.Series(np.where(g[m].notna(), m, None), index=g.index)
    return pd.Series(
        np.where(g["sectoral"].notna(), "sectoral", np.where(g["basic"].notna(), "basic", None)), index=g.index
    )


def gmp_pc(ctx: Context) -> pd.Series:
    return _gmp_choose(ctx, "gmp_pc")


def gmp_pc_real(ctx: Context) -> pd.Series:
    return _gmp_choose(ctx, "gmp_pc_real")


def gmp_imputed_share(ctx: Context) -> pd.Series:
    return _gmp_choose(ctx, "gmp_imputed_share")


def gmp_structure(ctx: Context) -> pd.DataFrame:
    s = ctx.structure.pivot_table(index=["territory_id", "year"], columns="section", values="share")
    s.columns = [f"gmp_structure_{c}" for c in s.columns]
    return s.reindex(ctx.wide.index)


def wage(ctx: Context) -> pd.Series:
    return ctx.col("wage")


def wage_real(ctx: Context) -> pd.Series:
    b = ctx.prices.set_index(["region_code", "year"])["basket"]
    return ctx.col("wage") / ctx.by_region(b)


def payroll_pc(ctx: Context) -> pd.Series:
    return _per_capita(ctx, "payroll")


def shipped_pc(ctx: Context) -> pd.Series:
    return _per_capita(ctx, "shipped")


def invest_pc(ctx: Context) -> pd.Series:
    return _per_capita(ctx, "invest")


def invest_pc_nobudget(ctx: Context) -> pd.Series:
    return ctx.col("invest_pc_nobudget")


def invest_share(ctx: Context) -> pd.Series:
    gmp_abs = _gmp_choose(ctx, "gmp")
    return ctx.col("invest") / gmp_abs


def emp_share(ctx: Context) -> pd.DataFrame:
    return pd.DataFrame({f"emp_share_{k}": ctx.col(f"workers__{k}") / ctx.col("workers") for k in OKVED2})


def _lq(ctx: Context, base_ids: set | None, prefix: str) -> pd.DataFrame:
    sh = emp_share(ctx)
    valid = ctx.col("valid_in_year").fillna(False).astype(bool)
    tid = pd.Series(ctx.wide.index.get_level_values("territory_id"), index=ctx.wide.index)
    sample = valid & (tid.isin(base_ids) if base_ids is not None else True)
    out = {}
    for k in OKVED2:
        # доля отрасли в базе: Σ работников раздела / Σ всех работников базы за год
        num = ctx.col(f"workers__{k}").where(sample)
        yr = ctx.wide.index.get_level_values("year")
        base = num.groupby(yr).sum() / ctx.col("workers").where(sample).groupby(yr).sum()
        out[f"{prefix}_{k}"] = sh[f"emp_share_{k}"] / pd.Series(yr.map(base), index=ctx.wide.index)
    return pd.DataFrame(out)


def lq(ctx: Context) -> pd.DataFrame:
    base = ctx.params.get("lq_base", "ДФО")
    ids = set(ctx.mo.loc[ctx.mo["federal_district"].eq(base), "territory_id"]) if base != "Россия" else None
    return _lq(ctx, ids, "lq")


def lq_ru(ctx: Context) -> pd.DataFrame:
    return _lq(ctx, None, "lq_ru")


def hhi_emp(ctx: Context) -> pd.Series:
    sh = emp_share(ctx)
    return (sh**2).sum(axis=1, min_count=1)


def emp_share_unknown(ctx: Context) -> pd.Series:
    return (1 - emp_share(ctx).sum(axis=1, min_count=1)).clip(lower=0)


def budget_own_share(ctx: Context) -> pd.Series:
    return ctx.col("budget_own_share")


def budget_rev_pc(ctx: Context) -> pd.Series:
    return _per_capita(ctx, "budget_rev")


# ============================================================================ социальная сфера
def pop(ctx: Context) -> pd.Series:
    return ctx.col("pop")


def pop_growth(ctx: Context) -> pd.Series:
    p = ctx.col("pop")
    return p / p.groupby(level="territory_id").shift(1) - 1


def pop_growth_window(ctx: Context) -> pd.Series:
    p = ctx.col("pop").dropna()
    yr = p.index.get_level_values("year")
    g = pd.DataFrame({"p": p.values, "y": yr}, index=p.index.get_level_values("territory_id"))
    first = g.sort_values("y").groupby(level=0).first()
    last = g.sort_values("y").groupby(level=0).last()
    t = (last["y"] - first["y"]).replace(0, np.nan)
    rate = (last["p"] / first["p"]) ** (1 / t) - 1
    return pd.Series(ctx.wide.index.get_level_values("territory_id").map(rate), index=ctx.wide.index)


def density(ctx: Context) -> pd.Series:
    # площадь МО почти не меняется: пропуски заполняются ближайшим известным значением того же МО
    g = ctx.col("area_ha").groupby(level="territory_id")
    area = g.ffill().groupby(level="territory_id").bfill()
    return ctx.col("pop") / (area / 100)


def housing_pc(ctx: Context) -> pd.Series:
    return ctx.col("housing_pc_src")


def retail_pc(ctx: Context) -> pd.Series:
    return _per_capita(ctx, "retail")


def living_space_pc(ctx: Context) -> pd.Series:
    return ctx.col("living_space_pc")


def preschool_coverage(ctx: Context) -> pd.Series:
    return ctx.col("preschool_coverage")


def clinics_per_10k(ctx: Context) -> pd.Series:
    return ctx.col("clinics") / ctx.col("pop") * 1e4


def natural_growth(ctx: Context) -> pd.Series:
    return (ctx.col("births") - ctx.col("deaths")) / ctx.col("pop") * 1000


def old_age_share(ctx: Context) -> pd.Series:
    return ctx.col("pop_old") / ctx.col("pop")


# ============================================================================ потребление
def spend_share(ctx: Context) -> pd.DataFrame:
    return pd.DataFrame({f"spend_share_{c}": ctx.col(f"spend_share_{c}") for c in SPEND_CATS})


def spend_to_wage(ctx: Context) -> pd.Series:
    return ctx.col("spend_total") / ctx.col("wage")


def market_access(ctx: Context) -> pd.Series:
    return ctx.col("market_access")


# ============================================================================ сборка
def compute_all(ctx: Context) -> pd.DataFrame:
    """Все показатели реестра + версии «_cpi» для денежных. Возвращает wide (territory_id, year)."""
    cols = {}
    lvl = ctx.by_region(ctx.prices.set_index(["region_code", "year"])["cpi_level"])
    for ind in registry():
        f = globals().get(ind["code"])
        if f is None:
            raise NotImplementedError(f"Нет функции для показателя {ind['code']} в src/indicators.py")
        r = f(ctx)
        if isinstance(r, pd.Series):
            cols[ind["code"]] = r
            if ind.get("money"):
                cols[f"{ind['code']}_cpi"] = r / lvl
        else:
            cols.update({c: r[c] for c in r.columns})
    out = pd.DataFrame(cols, index=ctx.wide.index)
    out = out.replace([np.inf, -np.inf], np.nan)
    out["gmp_method_used"] = gmp_method_used(ctx)
    return out


def expand_registry() -> pd.DataFrame:
    """Реестр с раскрытыми отраслевыми и «_cpi» кодами: одна строка на колонку indicators_wide."""
    rows = []
    sectors = {"spend_share": SPEND_CATS, "gmp_structure": [*OKVED2, "T"]}
    for ind in registry():
        if ind.get("sectoral"):
            for k in sectors.get(ind["code"], OKVED2):
                rows.append({**ind, "code": f"{ind['code']}_{k}", "name": f"{ind['name']}: {k}", "parent": ind["code"]})
        else:
            rows.append({**ind, "parent": ind["code"]})
            if ind.get("money"):
                rows.append(
                    {
                        **ind,
                        "code": f"{ind['code']}_cpi",
                        "name": f"{ind['name']} (в ценах {registry_params()['base_year']} г., ИПЦ)",
                        "unit": f"{ind['unit']} {registry_params()['base_year']} г.",
                        "parent": ind["code"],
                        "money": False,
                    }
                )
    return pd.DataFrame(rows)
