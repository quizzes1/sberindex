"""Годовая панель territory_id × year (этап 2).

Шаги (см. reports/DATA.md):
1. каждый ряд из configs/panel.yaml читается из БДПМО (годовое итоговое значение, по отраслевым
   рядам — ещё и разделы ОКВЭД2 / ОКВЭД-2007) и привязывается к territory_id
   (src.dictionary.map_to_territory); непривязанные строки — в лог;
2. если в один territory_id × год попало несколько исходных МО (объединения до 2018 г.,
   старые и новые коды) — аддитивные ряды суммируются, удельные усредняются с весом
   (население или работники), строка получает флаг merged_sources;
3. пропуски населения заполняются запасными рядами (флаг filled_<код>);
4. для преемников объединений строятся значения прошлых лет из предшественников
   (флаг aggregated_predecessors, derived = True), при передаче части территории ряд
   предшественника с тем же ОКТМО переносится преемнику (linked_transfer);
5. траты СберИндекса агрегируются до года, добавляется индекс доступности рынков.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import bdmo
from src.dictionary import dict_for_year, load_dict, map_to_territory, region_to_fo
from src.io import RAW, load_yaml
from src.regional import okved2_letter

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

TERRITORIAL_HISTORY = r"Объединение|Выделение|территориальный состав"
OTHER = "other"


# ----------------------------------------------------------------------------- чтение рядов
def _section_suffix(label: str, classifier: str) -> str | None:
    letter = okved2_letter(label)
    if letter is None:
        return None
    return letter if classifier == "okved2" else f"o07_{letter}"


def read_series(spec: dict) -> pd.DataFrame:
    """Строки одного ряда на уровне исходных МО (oktmo8 × year × indicator).

    Колонки: oktmo8, oktmo_stable8, year, indicator, value, unit, source, oktmo_history, comment,
    priority (номер кода в списке codes; меньше — приоритетнее).
    """
    frames = []
    for prio, code in enumerate(spec["codes"]):
        if spec.get("sectoral"):
            raw = bdmo.annual(bdmo.read_top(code))
            clf = "okved2" if "okved2" in raw.columns else "okved"
            # «итог» по остальным разрезам (например, форма собственности okfs) — по правилам sources.yaml / select
            rules = {**(bdmo.indicator_meta(code).get("total") or {}), **(spec.get("select") or {})}
            for dim, vals in rules.items():
                if dim != clf and dim in raw.columns:
                    raw = raw[raw[dim].isin(vals if isinstance(vals, list) else [vals])]
            tot = raw[raw[clf].str.contains(bdmo.TOTAL_RE, regex=True, na=False)].assign(indicator=spec["id"])
            sec = raw.assign(_s=raw[clf].map(lambda x, c=clf: _section_suffix(x, c))).dropna(subset=["_s"])
            sec = sec.assign(indicator=spec["id"] + "__" + sec["_s"]).drop(columns="_s")
            d = pd.concat([tot, sec])
            d = d.assign(_na=d["value"].isna()).sort_values("_na", kind="stable")
            d = d.drop_duplicates(["oktmo8", "year", "indicator"]).drop(columns="_na")
        else:
            d, _ = bdmo.annual_total(code, spec.get("select"))
            d = d.assign(indicator=spec["id"])
        d = d.assign(source=str(code), priority=prio, unit=d["indicator_unit"])
        frames.append(
            d[
                [
                    "oktmo8",
                    "oktmo_stable8",
                    "year",
                    "indicator",
                    "value",
                    "unit",
                    "source",
                    "priority",
                    "oktmo_history",
                    "oktmo_year_from",
                    "comment",
                    "region_name",
                    "municipality",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def read_all(cfg: dict) -> pd.DataFrame:
    """Все ряды панели на уровне исходных МО."""
    out = [read_series(s) for s in cfg["series"]]
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- агрегация
def _weight_name(indicator: str, spec: dict) -> str:
    """Имя ряда-веса: для отраслевой зарплаты wage__C — workers__C."""
    w = spec.get("weight", "pop")
    if "__" in indicator and spec.get("sectoral"):
        return f"{w}__{indicator.split('__', 1)[1]}"
    return w


def collapse(raw: pd.DataFrame, cfg: dict, unmatched_log: list) -> pd.DataFrame:
    """Привязка к territory_id и свёртка нескольких исходных МО в одну территорию.

    Возвращает long-таблицу territory_id × year × indicator с колонками value, unit, source, flag.
    """
    specs = {s["id"]: s for s in cfg["series"]}

    def spec_of(ind: str) -> dict:
        return specs[ind.split("__", 1)[0]]

    keys = raw[["oktmo8", "oktmo_stable8", "year"]].drop_duplicates()
    m = map_to_territory(keys)
    raw = raw.merge(m[["oktmo8", "oktmo_stable8", "year", "territory_id", "match"]], how="left")

    bad = raw[raw["territory_id"].isna()]
    if len(bad):
        unmatched_log.append(
            bad.groupby(["source", "region_name", "municipality", "oktmo8", "oktmo_stable8", "match"], dropna=False)
            .agg(years=("year", lambda s: f"{s.min()}–{s.max()}"), rows=("value", "size"))
            .reset_index()
        )
    raw = raw.dropna(subset=["territory_id"])

    # 1) приоритет кодов: для territory × year × indicator берём самый приоритетный код с данными
    has = raw.dropna(subset=["value"])
    best = has.groupby(["territory_id", "year", "indicator"])["priority"].min().rename("best")
    raw = raw.merge(best, on=["territory_id", "year", "indicator"], how="left")
    raw = raw[raw["priority"].eq(raw["best"]) | raw["best"].isna()]
    raw = raw.dropna(subset=["value"])

    # 2) свёртка нескольких исходных МО
    raw["flag"] = ""
    raw.loc[raw["oktmo_history"].str.contains(TERRITORIAL_HISTORY, na=False, regex=True), "flag"] = (
        "territorial_history"
    )
    raw.loc[raw["comment"].eq("Аномальное значение показателя"), "flag"] += ";tochno_anomaly"
    n = raw.groupby(["territory_id", "year", "indicator"])["value"].transform("size")
    single = raw[n.eq(1)]
    multi = raw[n.gt(1)].copy()

    rows = []
    if len(multi):
        # веса на уровне исходных МО
        wtab = raw.set_index(["oktmo8", "year", "indicator"])["value"]
        wtab = wtab[~wtab.index.duplicated()]
        for (_tid, y, ind), g in multi.groupby(["territory_id", "year", "indicator"]):
            sp = spec_of(ind)
            agg = sp.get("agg", "sum")
            # одно и то же значение под старым и новым кодом (год смены типа МО) — это дубль
            g = g.drop_duplicates("value")
            territorial = g["oktmo_history"].str.contains(TERRITORIAL_HISTORY, na=False, regex=True).any()
            if len(g) == 1 or not territorial:
                # переименование без изменения территории: берём код, действовавший в этом году,
                # иначе самый новый; суммировать нельзя — это одна и та же территория
                pick = g[g["match"].eq("by_year")] if g["match"].eq("by_year").any() else g
                pick = pick.sort_values("oktmo_year_from", ascending=False).iloc[0]
                flags = set(filter(None, ";".join(g["flag"]).split(";"))) | {"duplicate_codes"}
                rows.append({**pick.to_dict(), "flag": ";".join(sorted(flags))})
                continue
            if agg == "sum":
                v = g["value"].sum()
            elif agg == "mean":
                wn = _weight_name(ind, sp)
                wv = np.array([wtab.get((o, y, wn), np.nan) for o in g["oktmo8"]], dtype=float)
                v = np.average(g["value"], weights=wv) if np.isfinite(wv).all() and wv.sum() > 0 else np.nan
            else:
                v = np.nan
            first = g.iloc[0]
            rows.append(
                {
                    **first.to_dict(),
                    "value": v,
                    "flag": ";".join(sorted(filter(None, set(";".join(g["flag"]).split(";")) | {"merged_sources"}))),
                }
            )
    merged = pd.DataFrame(rows) if rows else multi.iloc[0:0]
    out = pd.concat([single, merged], ignore_index=True)
    out["flag"] = out["flag"].str.strip(";")
    return out[["territory_id", "year", "indicator", "value", "unit", "source", "flag"]].dropna(subset=["value"])


POSITIVE = ("pop", "pop_avg", "workers", "payroll", "wage", "wage_lm", "area_ha")
# доли в процентах, которые по определению лежат в [0, 100]: доля собственных (налоговых и неналоговых) доходов
# в доходах бюджета. В БДПМО встречаются 3 272, 2 701, −13 % — ошибка ввода или сумма вместо доли.
PERCENT = ("budget_own_share",)


def reject_invalid(long: pd.DataFrame, rejected_log: list) -> pd.DataFrame:
    """Отбраковка заведомо ошибочных значений: ≤ 0 у рядов, которые обязаны быть положительными
    (население, работники, зарплата, площадь), и долей в процентах вне [0, 100]. Строки уходят в лог
    rejected (data/processed/rejected.csv) и становятся пропуском — значения не обрезаются и не заменяются."""
    bad = long["indicator"].isin(POSITIVE) & long["value"].le(0)
    if bad.any():
        rejected_log.append(long[bad].assign(reason="значение ≤ 0"))
    pct = long["indicator"].isin(PERCENT) & ~long["value"].between(0, 100)
    if pct.any():
        rejected_log.append(long[pct].assign(reason="доля вне 0–100 %"))
    return long[~(bad | pct)]


# ----------------------------------------------------------------------------- заполнение
def fill_population(long: pd.DataFrame, cfg: dict, unmatched_log: list, rejected_log: list) -> pd.DataFrame:
    """Пропуски pop: сначала 8112014 (половозрастной ряд, на 1 января), затем 8112013 (среднегодовая)."""
    spec = next(s for s in cfg["series"] if s["id"] == "pop")
    for code in spec.get("fill", []):
        alt = collapse(read_series({"id": "pop", "codes": [code], "agg": "sum"}), cfg, unmatched_log)
        have = long.loc[long["indicator"].eq("pop"), ["territory_id", "year"]]
        alt = alt.merge(have, how="left", indicator=True)
        alt = alt[alt["_merge"].eq("left_only")].drop(columns="_merge")
        alt = reject_invalid(alt, rejected_log)
        alt["flag"] = (alt["flag"] + f";filled_{code}").str.strip(";")
        long = pd.concat([long, alt], ignore_index=True)
    return long


# ----------------------------------------------------------------------------- границы
def boundary_events() -> pd.DataFrame:
    """События справочника: event, kind, year (с какого года действует преемник), preds, succs."""
    d = load_dict()
    to = d.dropna(subset=["change_id_to"]).groupby("change_id_to")["territory_id"].apply(sorted)
    fr = d.dropna(subset=["change_id_from"]).groupby("change_id_from")
    ev = pd.DataFrame({"preds": to, "succs": fr["territory_id"].apply(sorted), "year": fr["year_from"].min()})
    ev["kind"] = ev.index.str.split("_").str[0]
    return ev.sort_values("year")


def extend_successors(long: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Значения прошлых лет для преемников объединений (сумма/взвешенное среднее предшественников)
    и для преемников передач с тем же ОКТМО (перенос ряда). Такие строки: derived = True."""
    specs = {s["id"]: s for s in cfg["series"]}
    d = load_dict()
    okt = d.groupby("territory_id")["oktmo8"].apply(set)
    b = cfg.get("boundary", {})
    long = long.copy()
    long["derived"] = False
    for ev, e in boundary_events().iterrows():
        t0 = int(e["year"])
        if e["kind"] == "union" and b.get("aggregate_predecessors", True):
            succ = e["succs"][0]
            src = long[long["territory_id"].isin(e["preds"]) & long["year"].lt(t0)]
            if src.empty:
                continue
            piv = src.pivot_table(index=["year", "indicator"], columns="territory_id", values="value", aggfunc="first")
            piv = piv.reindex(columns=e["preds"])
            full = piv.dropna()  # только если есть у всех предшественников
            pop = long[long["territory_id"].isin(e["preds"])].pivot_table(
                index=["year", "indicator"], columns="territory_id", values="value", aggfunc="first"
            )
            rows = []
            for (y, ind), r in full.iterrows():
                sp = specs[ind.split("__", 1)[0]]
                agg = sp.get("agg", "sum")
                if agg == "sum":
                    v = r.sum()
                elif agg == "mean":
                    wn = _weight_name(ind, sp)
                    wv = pop.reindex(columns=e["preds"]).loc[(y, wn)] if (y, wn) in pop.index else None
                    if wv is None or wv.isna().any() or wv.sum() <= 0:
                        continue
                    v = float(np.average(r.values, weights=wv.values))
                else:
                    continue
                rows.append((succ, y, ind, v))
            add = pd.DataFrame(rows, columns=["territory_id", "year", "indicator", "value"])
        elif e["kind"] == "transfer" and b.get("link_transfer", True):
            add = []
            for s in e["succs"]:
                same = [p for p in e["preds"] if okt[p] & okt[s]]
                if len(same) == 1:
                    a = long[long["territory_id"].eq(same[0]) & long["year"].lt(t0)][["year", "indicator", "value"]]
                    add.append(a.assign(territory_id=s))
            add = pd.concat(add, ignore_index=True) if add else pd.DataFrame()
        else:
            continue
        if add.empty:
            continue
        units = long.drop_duplicates("indicator").set_index("indicator")[["unit", "source"]]
        add = add.join(units, on="indicator")
        tag = "aggregated_predecessors" if e["kind"] == "union" else "linked_transfer"
        add = add.assign(flag=f"{tag}:{ev}", derived=True)
        exists = long.set_index(["territory_id", "year", "indicator"]).index
        add = add[~add.set_index(["territory_id", "year", "indicator"]).index.isin(exists)]
        long = pd.concat([long, add], ignore_index=True)
    return long


# ----------------------------------------------------------------------------- проверки
def flag_sector_anomalies(long: pd.DataFrame, cfg: dict, tol: float = 0.05) -> pd.DataFrame:
    """Флаг anomaly_sector_sum: сумма разделов аддитивного ряда (работники, ФОТ, отгрузка) больше
    итога более чем на tol — ошибка источника; флаг ставится на все разделы этого МО-года-ряда."""
    long = long.copy()
    additive = [s["id"] for s in cfg["series"] if s.get("sectoral") and s.get("agg", "sum") == "sum"]
    is_sec = long["indicator"].str.contains("__", regex=False)
    sec = long[is_sec & long["indicator"].str.split("__").str[0].isin(additive)]
    base = sec["indicator"].str.split("__").str[0]
    clf = np.where(sec["indicator"].str.contains("__o07_"), "o07", "okved2")
    s = sec.assign(base=base, clf=clf).groupby(["territory_id", "year", "base", "clf"])["value"].sum()
    tot = long[~long["indicator"].str.contains("__")].set_index(["territory_id", "year", "indicator"])["value"]
    tot = tot[~tot.index.duplicated()]
    s = s.reset_index()
    s["tot"] = [tot.get((t, y, b), np.nan) for t, y, b in zip(s["territory_id"], s["year"], s["base"])]
    bad = s[s["value"] > (1 + tol) * s["tot"]]
    if bad.empty:
        return long
    key = set(zip(bad["territory_id"], bad["year"], bad["base"]))
    is_bad = [
        ("__" in ind) and ((t, y, ind.split("__")[0]) in key) and ind.split("__")[0] in additive
        for t, y, ind in zip(long["territory_id"], long["year"], long["indicator"])
    ]
    long.loc[is_bad, "flag"] = (long.loc[is_bad, "flag"] + ";anomaly_sector_sum").str.strip(";")
    return long


def flag_jumps(long: pd.DataFrame, indicators: tuple = ("pop",), threshold: float = 0.2) -> pd.DataFrame:
    """Флаг jump: значение отличается от обоих соседних лет более чем на threshold в одну сторону
    (одиночный выброс, например, население нового округа в старых границах). Значение не меняется."""
    long = long.copy()
    for ind in indicators:
        m = long["indicator"].eq(ind) & ~long["derived"]
        s = long[m].set_index(["territory_id", "year"])["value"].sort_index()
        prev = s.groupby(level=0).shift(1)
        nxt = s.groupby(level=0).shift(-1)
        r1, r2 = s / prev - 1, s / nxt - 1
        bad = ((r1 > threshold) & (r2 > threshold)) | ((r1 < -threshold) & (r2 < -threshold))
        idx = long[m].index[
            bad.reindex(list(zip(long.loc[m, "territory_id"], long.loc[m, "year"]))).fillna(False).values
        ]
        long.loc[idx, "flag"] = (long.loc[idx, "flag"] + ";jump").str.strip(";")
    return long


# ----------------------------------------------------------------------------- СберИндекс
def sber_annual(cfg: dict) -> pd.DataFrame:
    """Траты СберИндекса по годам: средние месячные траты на жителя и доли категорий.

    spend_total — среднее «Все категории» за месяцы года; spend_<кат> — то же по категории;
    spend_share_<кат> = Σмесяцев траты категории / Σмесяцев итог (по месяцам, где есть итог);
    spend_share_other = 1 − сумма пяти долей; spend_months — число месяцев с итогом.
    """
    k = pd.read_parquet(RAW / "sber/hackathonlicence/consumption.parquet")
    cats = cfg["sber_categories"]
    k["year"] = k["date"].str[:4].astype(int)
    piv = k.pivot_table(index=["territory_id", "year", "date"], columns="category", values="value", aggfunc="first")
    piv = piv.dropna(subset=["Все категории"])
    g = piv.groupby(level=["territory_id", "year"])
    out = {"spend_total": g["Все категории"].mean(), "spend_months": g["Все категории"].size().astype(float)}
    tot = g["Все категории"].sum()
    five = 0
    for ru, code in cats.items():
        out[f"spend_{code}"] = g[ru].mean()
        out[f"spend_share_{code}"] = g[ru].sum(min_count=1) / tot
        five = five + out[f"spend_share_{code}"]
    out[f"spend_share_{OTHER}"] = 1 - five
    out[f"spend_{OTHER}"] = out["spend_total"] * out[f"spend_share_{OTHER}"]
    w = pd.DataFrame(out).reset_index()
    long = w.melt(id_vars=["territory_id", "year"], var_name="indicator", value_name="value").dropna(subset=["value"])
    long["unit"] = np.where(
        long["indicator"].str.startswith("spend_share"),
        "доля",
        np.where(long["indicator"].eq("spend_months"), "месяцев", "руб. в месяц на жителя"),
    )
    long["source"] = "sber_consumption"
    months = w.set_index(["territory_id", "year"])["spend_months"]
    long["flag"] = np.where(long.set_index(["territory_id", "year"]).index.map(months) < 12, "lt12_months", "")
    ma = pd.read_parquet(RAW / "sber/hackathonlicence/market_access.parquet")
    ma = ma.assign(
        year=2024,
        indicator="market_access",
        value=ma["market_access"],
        unit="индекс 0–1000",
        source="sber_market_access",
        flag="",
    )
    return pd.concat([long, ma[long.columns]], ignore_index=True).assign(derived=False)


# ----------------------------------------------------------------------------- сборка
def valid_in_year(territory_id: pd.Series, year: pd.Series) -> pd.Series:
    """Действовала ли территория в этом году по справочнику (до 2018 г. — состав 2018 г.)."""
    sets = {y: set(dict_for_year(int(y))["territory_id"]) for y in year.dropna().unique()}
    return pd.Series([t in sets[y] for t, y in zip(territory_id, year)], index=territory_id.index)


def build(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Полная сборка: (panel_long, unmatched, rejected)."""
    cfg = cfg or load_yaml("panel.yaml")
    y0, y1 = cfg["years"]
    log: list = []
    rejected: list = []
    raw = read_all(cfg)
    raw = raw[raw["year"].between(y0, y1)]
    long = collapse(raw, cfg, log)
    long = reject_invalid(long, rejected)
    long = fill_population(long, cfg, log, rejected)
    long = long[long["year"].between(y0, y1)]
    long = extend_successors(long, cfg)
    long = flag_sector_anomalies(long, cfg)
    long = flag_jumps(long)
    long = pd.concat([long, sber_annual(cfg)], ignore_index=True)
    long["territory_id"] = long["territory_id"].astype("int64")
    long["year"] = long["year"].astype("int64")
    long["valid_in_year"] = valid_in_year(long["territory_id"], long["year"])
    long["flag"] = long["flag"].fillna("").astype(str)
    long = long.sort_values(["territory_id", "year", "indicator"]).reset_index(drop=True)
    unmatched = pd.concat(log, ignore_index=True) if log else pd.DataFrame()
    if len(unmatched):
        unmatched = unmatched[unmatched["years"].map(lambda s: int(str(s)[-4:]) >= y0)]
    rej = pd.concat(rejected, ignore_index=True) if rejected else pd.DataFrame()
    return long, unmatched, rej


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на territory_id × year; флаги — отдельными колонками."""
    w = long.pivot_table(index=["territory_id", "year"], columns="indicator", values="value", aggfunc="first")
    w.columns.name = None
    meta = long.groupby(["territory_id", "year"]).agg(
        valid_in_year=("valid_in_year", "max"), derived=("derived", "min")
    )
    pf = long[long["indicator"].eq("pop")].set_index(["territory_id", "year"])["flag"]
    meta["pop_filled"] = pf.str.contains("filled_").reindex(meta.index).fillna(False)
    meta["merged_sources"] = long.groupby(["territory_id", "year"])["flag"].apply(
        lambda s: s.str.contains("merged_sources").any()
    )
    meta["anomaly_sector_sum"] = long.groupby(["territory_id", "year"])["flag"].apply(
        lambda s: s.str.contains("anomaly_sector_sum").any()
    )
    return meta.join(w).reset_index()


def mo_table(long: pd.DataFrame) -> pd.DataFrame:
    """Справочник МО для панели (mo.parquet)."""
    d = load_dict().sort_values(["territory_id", "year_from"])
    last = d.groupby("territory_id").tail(1).set_index("territory_id")
    hist = d.groupby("territory_id").apply(
        lambda g: ";".join(
            f"{r.year_from}-{'' if r.year_to == 9999 else r.year_to - 1}:{r.oktmo8}" for r in g.itertuples()
        )
    )
    ev = boundary_events()
    bc = {}
    for name, e in ev.iterrows():
        for t in e["preds"]:
            bc.setdefault(t, []).append(f"{name}:предшественник")
        for t in e["succs"]:
            bc.setdefault(t, []).append(f"{name}:преемник")
    # МО, у которых набор tochno.st отмечает территориальные изменения или несколько исходных кодов
    pre = long[long["flag"].str.contains("merged_sources|territorial_history", regex=True)]
    for t in pre["territory_id"].unique():
        bc.setdefault(t, []).append("tochno_территориальные_изменения")
    mo = pd.DataFrame(
        {
            "territory_id": last.index,
            "name": last["municipal_district_name"].values,
            "name_short": last["municipal_district_name_short"].values,
            "type": last["municipal_district_type"].values,
            "status": last["municipal_district_status"].values,
            "oktmo": last["oktmo8"].values,
            "oktmo_history": hist.reindex(last.index).values,
            "region_code": last["region_code"].values,
            "region_name": last["region_name"].values,
            "federal_district": last["region_code"].map(region_to_fo()).values,
            "center": last["municipal_district_center"].values,
            "lat": last["municipal_district_center_lat"].values,
            "lon": last["municipal_district_center_lon"].values,
            "year_from": d.groupby("territory_id")["year_from"].min().reindex(last.index).values,
            "year_to": last["year_to"].values,
        }
    )
    mo["is_dfo"] = mo["federal_district"].eq("ДФО")
    mo["active"] = mo["year_to"].eq(9999)
    mo["boundary_change"] = mo["territory_id"].map(lambda t: t in bc)
    mo["boundary_events"] = mo["territory_id"].map(lambda t: ";".join(bc.get(t, [])))
    # флаги покрытия: число лет с данными по ключевым рядам
    real = long[~long["derived"]]
    for ind in ("pop", "workers", "wage", "shipped", "invest", "budget_own_share", "spend_total"):
        n = real[real["indicator"].eq(ind)].groupby("territory_id")["year"].nunique()
        mo[f"years_{ind}"] = mo["territory_id"].map(n).fillna(0).astype(int)
    return mo
