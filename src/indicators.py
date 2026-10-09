"""Расчёт показателей из реестра configs/indicators.yaml.

Каждый показатель — функция с именем его кода: f(ctx) -> Series (индекс territory_id, year) или
DataFrame (отраслевые показатели: колонки <код>_<раздел>). Добавить показатель = запись в YAML +
функция здесь. Деление на ноль и отсутствующие знаменатели дают NaN (пропуск ≠ ноль).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.io import PROCESSED, load_yaml

OKVED2 = list("ABCDEFGHIJKLMNOPQRS")
SPEND_CATS = ["food", "marketplaces", "transport", "health", "catering", "other"]


def registry() -> list[dict]:
    """Записи реестра показателей."""
    return load_yaml("indicators.yaml")["indicators"]


def registry_params() -> dict:
    """Общие параметры реестра показателей (configs/indicators.yaml → params)."""
    return load_yaml("indicators.yaml")["params"]


@dataclass
class Context:
    """Всё, что нужно функциям показателей. Таблицы индексированы (territory_id, year)."""

    wide: pd.DataFrame
    mo: pd.DataFrame
    gmp: pd.DataFrame
    structure: pd.DataFrame
    params: dict = field(default_factory=dict)

    @classmethod
    def load(cls, params: dict | None = None) -> Context:
        """Читает панель, справочник МО и оценки ВМП из data/processed; params дополняют настройки реестра."""
        p = {**registry_params(), **(params or {})}
        wide = pd.read_parquet(PROCESSED / "panel_wide.parquet").set_index(["territory_id", "year"]).sort_index()
        mo = pd.read_parquet(PROCESSED / "mo.parquet")
        gmp = pd.read_parquet(PROCESSED / "gmp.parquet")
        st = pd.read_parquet(PROCESSED / "gmp_structure.parquet")
        return cls(wide=wide, mo=mo, gmp=gmp, structure=st, params=p)

    def col(self, name: str) -> pd.Series:
        """Столбец панели по имени; если его нет — ряд из NaN (показатель не публиковался)."""
        return self.wide[name] if name in self.wide else pd.Series(np.nan, index=self.wide.index)

    def region(self) -> pd.Series:
        """region_code для каждой строки панели."""
        rc = self.mo.set_index("territory_id")["region_code"]
        return pd.Series(self.wide.index.get_level_values("territory_id").map(rc), index=self.wide.index)

    def by_region(self, s: pd.Series) -> pd.Series:
        """Региональный ряд (индекс region_code, year) → значения для строк панели."""
        idx = list(zip(self.region(), self.wide.index.get_level_values("year")))
        return pd.Series(s.reindex(idx).values, index=self.wide.index)


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
    """ВМП на душу населения (оценка команды): ВМП / P (руб.). Источник — расчёт (src/gmp.py)."""
    return _gmp_choose(ctx, "gmp_pc")


def gmp_imputed_share(ctx: Context) -> pd.Series:
    """Доля ВМП, распределённая по правилу для скрытых данных: импутированная ВДС / ВМП (доля). Источник — расчёт."""
    return _gmp_choose(ctx, "gmp_imputed_share")


def gmp_structure(ctx: Context) -> pd.DataFrame:
    """Доля отрасли в ВМП: ВДС_k МО / ВМП (доля). Источник — расчёт, метод 2."""
    s = ctx.structure.pivot_table(index=["territory_id", "year"], columns="section", values="share")
    s.columns = [f"gmp_structure_{c}" for c in s.columns]
    return s.reindex(ctx.wide.index)


def wage(ctx: Context) -> pd.Series:
    """Среднемесячная зарплата: W (руб.). Источник — БДПМО 8423007/8123007."""
    return ctx.col("wage")


def payroll_pc(ctx: Context) -> pd.Series:
    """ФОТ на жителя: ФОТ / P (руб. в год). Источник — БДПМО 8423006/8123006."""
    return _per_capita(ctx, "payroll")


def shipped_pc(ctx: Context) -> pd.Series:
    """Отгрузка на жителя: отгрузка / P (руб. в год). Источник — БДПМО 8401011/8201001."""
    return _per_capita(ctx, "shipped")


def invest_pc(ctx: Context) -> pd.Series:
    """Инвестиции в основной капитал на жителя: инвестиции / P (руб. в год). Источник — БДПМО 8109001."""
    return _per_capita(ctx, "invest")


def invest_pc_nobudget(ctx: Context) -> pd.Series:
    """Инвестиции на жителя без бюджетных средств: ряд Росстата (руб. в год). Источник — БДПМО 8109003."""
    return ctx.col("invest_pc_nobudget")


def invest_share(ctx: Context) -> pd.Series:
    """Инвестиции к продукту: инвестиции / ВМП (доля). Источник — БДПМО 8109001; расчёт."""
    gmp_abs = _gmp_choose(ctx, "gmp")
    return ctx.col("invest") / gmp_abs


def emp_share(ctx: Context) -> pd.DataFrame:
    """Доля занятых в отрасли: L_k / L (доля). Источник — БДПМО 8423005."""
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
    """Коэффициент локализации (база — параметр lq_base): emp_share_k(МО) / emp_share_k(база) (раз). Источник — БДПМО 8423005."""
    base = ctx.params.get("lq_base", "ДФО")
    ids = set(ctx.mo.loc[ctx.mo["federal_district"].eq(base), "territory_id"]) if base != "Россия" else None
    return _lq(ctx, ids, "lq")


def lq_ru(ctx: Context) -> pd.DataFrame:
    """Коэффициент локализации (база — Россия): emp_share_k(МО) / emp_share_k(Россия) (раз). Источник — БДПМО 8423005."""
    return _lq(ctx, None, "lq_ru")


def hhi_emp(ctx: Context) -> pd.Series:
    """Индекс Херфиндаля структуры занятости: Σ_k emp_share_k² (индекс). Источник — БДПМО 8423005."""
    sh = emp_share(ctx)
    return (sh**2).sum(axis=1, min_count=1)


def emp_share_unknown(ctx: Context) -> pd.Series:
    """Доля работников вне опубликованных разделов: 1 − Σ_k emp_share_k (доля). Источник — БДПМО 8423005."""
    return (1 - emp_share(ctx).sum(axis=1, min_count=1)).clip(lower=0)


def manuf_shipped_pc(ctx: Context) -> pd.Series:
    """Отгрузка обрабатывающих производств на жителя: отгрузка раздела C / P (руб. в год). Источник — БДПМО 8401011 (раздел C)."""
    return _per_capita(ctx, "shipped__C")


def agri_output_pc(ctx: Context) -> pd.Series:
    """Продукция сельского хозяйства на жителя: продукция с/х (все категории хозяйств) / P (руб. в год). Источник — БДПМО 8007010."""
    return _per_capita(ctx, "agri_output")


def manuf_orgs_per_10k(ctx: Context) -> pd.Series:
    """Обрабатывающие организации на 10 тыс. жителей: организации раздела C, представившие отчёт / P × 10⁴ (на 10 тыс.). Источник — БДПМО 8942010 (раздел C)."""
    return ctx.col("n_orgs_rep__C") / ctx.col("pop") * 1e4


def employment_ratio(ctx: Context) -> pd.Series:
    """Работники крупных и средних организаций на жителя трудоспособного возраста: L / население трудоспособного возраста (доля). Источник — БДПМО 8423005, 8112014."""
    return ctx.col("workers") / ctx.col("pop_working")


def budget_own_share(ctx: Context) -> pd.Series:
    """Бюджетная самостоятельность: доля налоговых и неналоговых доходов (%). Источник — БДПМО 8313015/8013015."""
    return ctx.col("budget_own_share")


def budget_rev_pc(ctx: Context) -> pd.Series:
    """Доходы местного бюджета на жителя: доходы / P (руб. в год). Источник — БДПМО 8013001."""
    return _per_capita(ctx, "budget_rev")


# ============================================================================ социальная сфера
def pop(ctx: Context) -> pd.Series:
    """Население на 1 января: P (чел.). Источник — БДПМО 8112027 (+8112014, 8112013)."""
    return ctx.col("pop")


def pop_growth(ctx: Context) -> pd.Series:
    """Темп изменения населения за год: P_t / P_{t−1} − 1 (доля). Источник — БДПМО."""
    p = ctx.col("pop")
    return p / p.groupby(level="territory_id").shift(1) - 1


def pop_growth_window(ctx: Context) -> pd.Series:
    """Среднегодовой темп изменения населения за окно: (P_last / P_first)^(1/T) − 1 (доля в год). Источник — БДПМО."""
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
    """Плотность населения: P / площадь (км²) (чел./км²). Источник — БДПМО 8006001."""
    g = ctx.col("area_ha").groupby(level="territory_id")
    area = g.ffill().groupby(level="territory_id").bfill()
    return ctx.col("pop") / (area / 100)


def housing_pc(ctx: Context) -> pd.Series:
    """Ввод жилья на жителя: ряд Росстата 8215001 (м² на жителя). Источник — БДПМО 8215001."""
    return ctx.col("housing_pc_src")


def retail_pc(ctx: Context) -> pd.Series:
    """Оборот розничной торговли на жителя: оборот / P (руб. в год). Источник — БДПМО 8401003/8201003."""
    return _per_capita(ctx, "retail")


def living_space_pc(ctx: Context) -> pd.Series:
    """Жилая площадь на жителя: ряд Росстата 8211001 (м²). Источник — БДПМО 8211001."""
    return ctx.col("living_space_pc")


def preschool_coverage(ctx: Context) -> pd.Series:
    """Охват детей 1–6 лет дошкольным образованием: ряд Росстата 8014006 (%). Источник — БДПМО 8014006."""
    return ctx.col("preschool_coverage")


def clinics_per_10k(ctx: Context) -> pd.Series:
    """Лечебно-профилактические организации на 10 тыс. жителей: организации / P × 10⁴ (на 10 тыс.). Источник — БДПМО 8018000."""
    return ctx.col("clinics") / ctx.col("pop") * 1e4


def natural_growth(ctx: Context) -> pd.Series:
    """Естественный прирост на 1000 жителей: (родившиеся − умершие) / P × 1000 (‰). Источник — БДПМО 8112003, 8112001."""
    return (ctx.col("births") - ctx.col("deaths")) / ctx.col("pop") * 1000


def old_age_share(ctx: Context) -> pd.Series:
    """Доля населения старше трудоспособного возраста: P_old / P (доля). Источник — БДПМО 8112014."""
    return ctx.col("pop_old") / ctx.col("pop")


# ============================================================================ потребление
def spend_share(ctx: Context) -> pd.DataFrame:
    """Доля категории в безналичных тратах: траты категории / все траты (доля). Источник — СберИндекс."""
    return pd.DataFrame({f"spend_share_{c}": ctx.col(f"spend_share_{c}") for c in SPEND_CATS})


def spend_to_wage(ctx: Context) -> pd.Series:
    """Траты к зарплате: среднемесячные траты на жителя / W (доля). Источник — СберИндекс; БДПМО."""
    return ctx.col("spend_total") / ctx.col("wage")


def market_access(ctx: Context) -> pd.Series:
    """Индекс доступности рынков (2024): Σ_d N_d / τ_od, нормировано 0–1000 (индекс). Источник — СберИндекс."""
    return ctx.col("market_access")


# ============================================================================ сборка
def compute_all(ctx: Context) -> pd.DataFrame:
    """Все показатели реестра в текущих ценах. Возвращает wide (territory_id, year).

    Денежные показатели (monetary) пересчитываются в цены базового года при использовании —
    src/prices.py, параметры params.prices; отдельных «реальных» колонок здесь нет.
    """
    cols = {}
    for ind in registry():
        f = globals().get(ind["code"])
        if f is None:
            raise NotImplementedError(f"Нет функции для показателя {ind['code']} в src/indicators.py")
        r = f(ctx)
        if isinstance(r, pd.Series):
            cols[ind["code"]] = r
        else:
            cols.update({c: r[c] for c in r.columns})
    out = pd.DataFrame(cols, index=ctx.wide.index)
    out = out.replace([np.inf, -np.inf], np.nan)
    out["gmp_method_used"] = gmp_method_used(ctx)
    return out


def expand_registry() -> pd.DataFrame:
    """Реестр с раскрытыми отраслевыми кодами: одна строка на колонку indicators_wide."""
    rows = []
    sectors = {"spend_share": SPEND_CATS, "gmp_structure": [*OKVED2, "T"]}
    for ind in registry():
        if ind.get("sectoral"):
            for k in sectors.get(ind["code"], OKVED2):
                rows.append({**ind, "code": f"{ind['code']}_{k}", "name": f"{ind['name']}: {k}", "parent": ind["code"]})
        else:
            rows.append({**ind, "parent": ind["code"]})
    return pd.DataFrame(rows)
