"""Индексы цен и пересчёт денежных показателей в цены базового года (этап 3 доработки).

Определения:
- уровень цен L_t — среднегодовой уровень цен в году t (произвольная база);
- реальное значение в ценах базового года b: real_t = nominal_t × L_b / L_t;
- по России (deflator_scope = national, по умолчанию) уровни считаются точно: из помесячных индексов
  «к предыдущему месяцу» строится помесячный уровень, среднегодовой уровень — среднее 12 месяцев
  (отношение соседних среднегодовых уровней = индекс «в среднем за год к предыдущему году»);
  годовые индексы «к предыдущему году» (цены сельхозпроизводителей) сцепляются напрямую;
- по субъектам (deflator_scope = regional) Росстат публикует только «декабрь к декабрю»: уровень на
  декабрь — цепное произведение, среднегодовой уровень приближается средним геометрическим уровней
  на декабрь прошлого и текущего года (флаг dec_dec_approx);
- дефлятор ВРП субъекта: (ВРП_t / ВРП_{t−1}) / (индекс физического объёма ВРП_t / 100); по России — то же
  по сумме субъектов (флаг grp_sum_regions).

Если у показателя нет своего индекса за какие-то годы, ряд продлевается темпами ИПЦ (флаг
extended_cpi); если в режиме «по субъектам» нет регионального ряда — берётся общероссийский (флаг
national_fallback). Флаги — в таблице factors() и на странице «Данные и качество».

Межрегиональная поправка (spatial_price_adjustment, по умолчанию выключена): деление на относительную
стоимость фиксированного набора товаров и услуг (субъект / Россия, Росстат, на конец года). Применяется
только к показателям с дефлятором ИПЦ (потребительские суммы: зарплата, ФОТ, бюджет, розница).

Сырые файлы читаются только при сборке (scripts/build_indicators.py → data/processed/price_levels.parquet);
интерфейс и конвейер берут уровни из этого файла.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src.io import PROCESSED, RAW

PRICES_RAW = RAW / "rosstat" / "prices"
LEVELS_PARQUET = PROCESSED / "price_levels.parquet"
RUSSIA = 0  # region_code для общероссийских рядов
YEARS = range(2012, 2026)

DEFLATORS = {
    "cpi": "ИПЦ",
    "grp": "дефлятор ВРП",
    "invest": "индекс цен на продукцию инвестиционного назначения",
    "ppi_industry": "индекс цен производителей промышленных товаров",
    "ppi_manufacturing": "индекс цен производителей обрабатывающих производств",
    "ppi_agri": "индекс цен производителей сельхозпродукции",
}
FLAGS = {
    "monthly": "по России, точно (помесячный индекс)",
    "annual": "годовой индекс «к предыдущему году»",
    "grp_sum_regions": "дефлятор ВРП по сумме субъектов",
    "grp_region": "дефлятор ВРП субъекта",
    "dec_dec_approx": "по субъекту, приближённо из «декабрь к декабрю»",
    "extended_cpi": "своего индекса за год нет — продлено темпами ИПЦ",
    "national_fallback": "регионального ряда нет — общероссийский",
}
MONTHS = [
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
]

# региональные ряды коллекции tochno.st («Регионы России»), декабрь к декабрю
REGIONAL_DEC = {"cpi": "Y477110111", "ppi_industry": "Y477110136", "ppi_manufacturing": "Y477110139"}
AGRI = "Y477110142"  # в процентах к предыдущему году (и по России, и по субъектам)
BASKET = "Y477110395"  # стоимость фиксированного набора, % к среднероссийской


# ============================================================================ чтение сырых файлов
def _year(v) -> int | None:
    m = re.match(r"^\s*(\d{4})", str(v))
    return int(m.group(1)) if m and 1990 <= int(m.group(1)) <= 2035 else None


def parse_monthly(path, sheet: str) -> pd.DataFrame:
    """Лист Росстата «индексы к предыдущему месяцу»: строка годов + 12 строк месяцев → годы × месяцы (%)."""
    d = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = next(i for i in range(len(d)) if sum(_year(v) is not None for v in d.iloc[i, 1:]) >= 5)
    years = {j: _year(d.iat[hdr, j]) for j in range(1, d.shape[1]) if _year(d.iat[hdr, j])}
    first = next(i for i in range(hdr + 1, len(d)) if str(d.iat[i, 0]).strip().lower() == "январь")
    rows = range(first, first + 12)
    assert [str(d.iat[i, 0]).strip().lower() for i in rows] == MONTHS, f"{path}:{sheet} — неожиданный порядок строк"
    out = pd.DataFrame({y: pd.to_numeric(d.iloc[list(rows), j], errors="coerce").values for j, y in years.items()})
    out.index = range(1, 13)
    return out.T.sort_index()


def annual_avg_level(monthly: pd.DataFrame) -> pd.Series:
    """Помесячные индексы (годы × 12, %) → среднегодовой уровень цен (только полные годы подряд)."""
    m = monthly[monthly.notna().all(axis=1)]
    yrs = [y for y in m.index if y - 1 in m.index or y == m.index.min()]
    m = m.loc[yrs]
    lvl = (m.stack() / 100).cumprod()
    return lvl.groupby(level=0).mean()


def _clean_run(index_pct: pd.Series) -> pd.Series:
    """Годовые индексы (%): отбросить служебные коды Росстата (−777777 и т.п.) и неправдоподобные значения,
    оставить последний непрерывный по годам участок (цепочку через разрыв не перемножаем)."""
    s = pd.to_numeric(index_pct, errors="coerce").sort_index()
    s = s[(s > 30) & (s < 1000)]
    if s.empty:
        return s
    yrs = s.index.astype(int)
    brk = np.where(np.diff(yrs) != 1)[0]
    return s.iloc[brk[-1] + 1 :] if len(brk) else s


def chain_annual(index_pct: pd.Series) -> pd.Series:
    """Годовой индекс «к предыдущему году» (%) → уровень (первый год ряда — база 1 для предыдущего)."""
    s = _clean_run(index_pct) / 100
    lvl = s.cumprod()
    return pd.concat([pd.Series({s.index.min() - 1: 1.0}), lvl])


def dec_dec_avg_level(index_pct: pd.Series) -> pd.Series:
    """Индекс «декабрь к декабрю» (%) → среднегодовой уровень ≈ √(L_дек,t−1 · L_дек,t)."""
    s = _clean_run(index_pct) / 100
    dec = pd.concat([pd.Series({s.index.min() - 1: 1.0}), s.cumprod()])
    return np.sqrt(dec * dec.shift(1)).dropna()


def _grp_levels() -> pd.DataFrame:
    from src.regional import grp, grp_volume_index

    g = grp().merge(grp_volume_index(), on=["region_code", "year"]).sort_values(["region_code", "year"])
    g["grp_prev"] = g.groupby("region_code")["grp"].shift(1)
    g["grp_const"] = g["grp_prev"] * g["grp_vi"] / 100  # ВРП года t в ценах года t−1
    rows = []
    for r, x in g.groupby("region_code"):
        step = (x["grp"] / x["grp_const"]).set_axis(x["year"])
        rows.append(chain_annual(step * 100).rename_axis("year").reset_index(name="level").assign(region_code=r))
    reg = pd.concat(rows).assign(flag="grp_region")
    ok = g.dropna(subset=["grp", "grp_const"])
    nat = ok.groupby("year")["grp"].sum() / ok.groupby("year")["grp_const"].sum()
    nat = chain_annual(nat * 100).rename_axis("year").reset_index(name="level")
    return pd.concat([reg, nat.assign(region_code=RUSSIA, flag="grp_sum_regions")])


def build_levels() -> pd.DataFrame:
    """Все уровни цен из сырых файлов: region_code (0 — Россия), year, deflator, level, flag."""
    from src.regional import REGIONS_PARQUET, regions_collection

    out = []

    def add(s: pd.Series, deflator: str, region: int, flag: str) -> None:
        out.append(
            pd.DataFrame(
                {
                    "region_code": region,
                    "year": s.index.astype(int),
                    "deflator": deflator,
                    "level": s.values,
                    "flag": flag,
                }
            )
        )

    add(annual_avg_level(parse_monthly(PRICES_RAW / "ipc_mes.xlsx", "01")), "cpi", RUSSIA, "monthly")
    add(annual_avg_level(parse_monthly(PRICES_RAW / "Invest_ind_svodn.xlsx", "1")), "invest", RUSSIA, "monthly")
    ppi = PRICES_RAW / "Proizvoditeli_Ind_VED.xlsx"
    add(annual_avg_level(parse_monthly(ppi, "2.1")), "ppi_industry", RUSSIA, "monthly")
    add(annual_avg_level(parse_monthly(ppi, "2.3")), "ppi_manufacturing", RUSSIA, "monthly")

    rc = pd.read_parquet(REGIONS_PARQUET, columns=["indicator_code", "object_level", "year", "indicator_value"])
    agri_ru = rc[rc["indicator_code"].eq(AGRI) & rc["object_level"].eq("Страна")].set_index("year")["indicator_value"]
    add(chain_annual(agri_ru), "ppi_agri", RUSSIA, "annual")

    d = regions_collection([*REGIONAL_DEC.values(), AGRI]).dropna(subset=["region_code"])
    for defl, code in REGIONAL_DEC.items():
        for r, x in d[d["indicator_code"].eq(code)].groupby("region_code"):
            add(dec_dec_avg_level(x.set_index("year")["indicator_value"]), defl, int(r), "dec_dec_approx")
    for r, x in d[d["indicator_code"].eq(AGRI)].groupby("region_code"):
        add(chain_annual(x.set_index("year")["indicator_value"]), "ppi_agri", int(r), "annual")
    out.append(_grp_levels().assign(deflator="grp"))
    lv = pd.concat(out, ignore_index=True)
    lv["region_code"] = lv["region_code"].astype(int)
    lv["year"] = lv["year"].astype(int)
    return lv[lv["year"].isin(YEARS)].reset_index(drop=True)


def build_basket() -> pd.DataFrame:
    """Относительная стоимость фиксированного набора: region_code, year, basket_rel (Россия = 1)."""
    from src.regional import regions_collection

    d = regions_collection([BASKET]).dropna(subset=["region_code"])
    d = d[d["indicator_unit"].str.contains("среднероссийск", na=False)]
    return (
        pd.DataFrame(
            {
                "region_code": d["region_code"].astype(int),
                "year": d["year"].astype(int),
                "basket_rel": d["indicator_value"] / 100,
            }
        )
        .query("0 < basket_rel < 10")
        .reset_index(drop=True)
    )


def save() -> pd.DataFrame:
    """Сборка: data/processed/price_levels.parquet (уровни) и price_basket.parquet (набор)."""
    lv = build_levels()
    lv.to_parquet(LEVELS_PARQUET, index=False)
    build_basket().to_parquet(PROCESSED / "price_basket.parquet", index=False)
    return lv


# ============================================================================ коэффициенты пересчёта
@lru_cache(maxsize=1)
def levels() -> pd.DataFrame:
    return pd.read_parquet(LEVELS_PARQUET)


@lru_cache(maxsize=1)
def basket() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "price_basket.parquet")


def _complete(series: pd.Series, flag: pd.Series, ref: pd.Series, ref_flag: str) -> tuple[pd.Series, pd.Series]:
    """Продлить ряд уровней на все годы YEARS темпами ряда ref (ИПЦ или общероссийского)."""
    s = series.reindex(YEARS)
    f = flag.reindex(YEARS)
    ref = ref.reindex(YEARS)
    if s.notna().sum() == 0:
        return ref.copy(), pd.Series(ref_flag, index=YEARS)
    known = s.dropna().index
    for y in range(known.max() + 1, max(YEARS) + 1):
        s[y] = s[y - 1] * ref[y] / ref[y - 1]
        f[y] = ref_flag
    for y in range(known.min() - 1, min(YEARS) - 1, -1):
        s[y] = s[y + 1] * ref[y] / ref[y + 1]
        f[y] = ref_flag
    return s, f


@lru_cache(maxsize=8)
def _complete_levels(scope: str) -> pd.DataFrame:
    """Полные ряды уровней по всем годам: region_code, year, deflator, level, flag."""
    lv = levels()
    nat = lv[lv["region_code"].eq(RUSSIA)].set_index(["deflator", "year"])
    cpi_ru = nat.loc["cpi", "level"]
    rows = []
    nat_full = {}
    for defl in DEFLATORS:
        if defl in nat.index.get_level_values(0):
            s, f = _complete(nat.loc[defl, "level"], nat.loc[defl, "flag"], cpi_ru, "extended_cpi")
        else:
            s, f = cpi_ru.reindex(YEARS), pd.Series("extended_cpi", index=YEARS)
        nat_full[defl] = (s, f)
        rows.append(
            pd.DataFrame(
                {"region_code": RUSSIA, "year": list(YEARS), "deflator": defl, "level": s.values, "flag": f.values}
            )
        )
    if scope == "regional":
        reg = lv[lv["region_code"].ne(RUSSIA)].set_index(["deflator", "region_code", "year"]).sort_index()
        have = set(reg.index.droplevel(2))
        regions = sorted(pd.read_parquet(PROCESSED / "mo.parquet")["region_code"].dropna().astype(int).unique())
        for defl in DEFLATORS:
            ns, nf = nat_full[defl]
            for r in regions:
                if (defl, r) in have:
                    x = reg.loc[(defl, r)]
                    s, f = _complete(x["level"], x["flag"], ns, "national_fallback")
                else:
                    s, f = ns, pd.Series("national_fallback", index=YEARS)
                rows.append(
                    pd.DataFrame(
                        {"region_code": r, "year": list(YEARS), "deflator": defl, "level": s.values, "flag": f.values}
                    )
                )
    return pd.concat(rows, ignore_index=True)


def factors(base_year: int, scope: str = "national") -> pd.DataFrame:
    """Коэффициенты real = nominal × factor: region_code, year, deflator, factor = L_b / L_t, flag.

    scope = national: строки с region_code = 0 (один коэффициент на год для всех МО);
    scope = regional: по субъектам (уровни субъекта, базовый год — свой для каждого субъекта).
    """
    if scope not in ("national", "regional"):
        raise ValueError(f"deflator_scope: national | regional, получено {scope!r}")
    if base_year not in YEARS:
        raise ValueError(f"base_year вне {min(YEARS)}–{max(YEARS)}: {base_year}")
    c = _complete_levels(scope)
    if scope == "regional":
        c = c[c["region_code"].ne(RUSSIA)]
    else:
        c = c[c["region_code"].eq(RUSSIA)]
    base = c[c["year"].eq(base_year)].set_index(["region_code", "deflator"])["level"]
    c = c.copy()
    c["factor"] = c.set_index(["region_code", "deflator"]).index.map(base).values / c["level"].values
    return c[["region_code", "year", "deflator", "factor", "flag"]].reset_index(drop=True)


# ============================================================================ применение к панели
def price_params(params: dict | None = None) -> dict:
    """Параметры цен по умолчанию из configs/indicators.yaml (params.prices) поверх переданных."""
    from src.indicators import registry_params

    p = dict(registry_params().get("prices", {}))
    p.update(params or {})
    return p


def deflators_of(codes) -> dict[str, str]:
    """{код показателя: дефлятор} для денежных показателей реестра (отраслевые коды — по родителю)."""
    from src.indicators import registry

    reg = {i["code"]: i for i in registry()}
    out = {}
    for c in codes:
        ind = reg.get(c)
        if ind is None:
            parent = next((p for p in reg if reg[p].get("sectoral") and c.startswith(p + "_")), None)
            ind = reg.get(parent) if parent else None
        if ind and ind.get("monetary"):
            out[c] = ind["deflator"]
    return out


def to_real(
    df: pd.DataFrame,
    codes,
    base_year: int,
    scope: str = "national",
    spatial: bool = False,
    region_col: str = "region_code",
    year_col: str = "year",
) -> pd.DataFrame:
    """Копия df, где денежные колонки codes пересчитаны в цены base_year (остальные — без изменений).

    df должен содержать колонки region_code и year (или индексы с такими именами).
    """
    out = df.copy()
    flat = out.reset_index() if (year_col not in out or region_col not in out) else out
    yr = flat[year_col].astype(int).values
    need_reg = scope == "regional" or spatial
    reg = flat[region_col].astype("float").fillna(-1).astype(int).values if need_reg else None
    f = factors(base_year, scope)
    if scope == "national":
        f = f.drop(columns="region_code")
    defl = deflators_of([c for c in codes if c in out])
    for c, d in defl.items():
        fd = f[f["deflator"].eq(d)]
        if scope == "national":
            k = pd.Series(yr).map(fd.set_index("year")["factor"]).values
        else:
            k = pd.MultiIndex.from_arrays([reg, yr]).map(fd.set_index(["region_code", "year"])["factor"]).values
        k = np.asarray(k, dtype=float)
        if spatial and d == "cpi":
            b = basket().set_index(["region_code", "year"])["basket_rel"]
            by = yr if scope == "national" else np.full(len(yr), base_year)
            rel = np.asarray(pd.MultiIndex.from_arrays([reg, by]).map(b), dtype=float)
            k = k / np.where(np.isfinite(rel), rel, 1.0)
        out[c] = out[c].values * k
    return out


LOG_UNIT_YEAR = 2023  # опорный год единицы для логарифма денежных признаков (см. log_unit_scale)


def log_unit_scale(codes, base_year: int) -> dict[str, float]:
    """Делитель денежного признака перед ln(1 + x): 1 руб. в ценах LOG_UNIT_YEAR, выраженный в ценах base_year.

    ln(1 + x) не инвариантен к умножению x на константу (при нулевых значениях — инвестиции, с/х продукция),
    а смена базового года по России умножает все реальные значения на L_b'/L_b. Деление на эту единицу
    делает признаки сети (а значит, и кластеры) независимыми от базового года при deflator_scope = national;
    при base_year = LOG_UNIT_YEAR делитель равен 1.
    """
    f = factors(LOG_UNIT_YEAR, "national").set_index(["deflator", "year"])["factor"]
    # factor(LOG_UNIT_YEAR → base_year) = L_ref / L_b; 1 руб. ref-года в ценах b = L_b / L_ref
    return {c: 1.0 / float(f[(d, base_year)]) for c, d in deflators_of(codes).items()}


def unit_label(unit: str, monetary: bool, params: dict) -> str:
    """Подпись единицы: «руб., в ценах 2023 г.» для реальных денежных показателей."""
    if not monetary or params.get("values", "real") == "nominal":
        return unit + (", текущие цены" if monetary else "")
    s = f"{unit}, в ценах {params['base_year']} г."
    if params.get("deflator_scope") == "regional":
        s += " (дефляторы субъектов)"
    if params.get("spatial_price_adjustment"):
        s += ", с поправкой на межрегиональные цены"
    return s


def coverage_table(base_year: int, scope: str) -> pd.DataFrame:
    """Сводка флагов: deflator × flag → число лет-субъектов (для страницы «Данные и качество»)."""
    f = factors(base_year, scope)
    t = f.groupby(["deflator", "flag"]).size().rename("n").reset_index()
    t["дефлятор"] = t["deflator"].map(DEFLATORS)
    t["источник"] = t["flag"].map(FLAGS)
    return t
