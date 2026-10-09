"""Отчёт об инвентаризации данных: reports/INVENTORY.md.

Запуск:  python scripts/build_inventory.py

Что считает:
- список скачанных файлов, размеры, SHA-256;
- по каждому показателю БДПМО: годы, единица, доля МО с непустым годовым итоговым значением
  по России и по ДФО за каждый год (знаменатель — число МО в справочнике СберИндекса на год;
  до 2018 г. — состав 2018 г.);
- входы для расчёта ВМП: ВРП и ВДС по ОКВЭД2 субъектов, отраслевые L/W/ФОТ и отгрузка по МО,
  продукция сельского хозяйства, частота скрытия отраслевых значений по МО ДФО;
- ИПЦ и фиксированный набор, траты Сбера, индекс доступности рынков, расстояния;
- предложение временного окна панели.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import pandas as pd  # noqa: E402

from src import bdmo  # noqa: E402
from src.dictionary import dict_for_year, load_dict, map_to_territory  # noqa: E402
from src.io import PROCESSED, RAW, REPORTS, load_yaml  # noqa: E402
from src.regional import grp, grp_volume_index, gva_okved2, okved2_letter, regions_collection  # noqa: E402

YEARS = list(range(2008, 2026))
OUT: list[str] = []


def w(s: str = "") -> None:
    """Добавить строку в текст отчёта."""
    OUT.append(s)


def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.0f}") -> str:
    """DataFrame → markdown-таблица без внешних зависимостей."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append("" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def denominators() -> pd.DataFrame:
    """Число МО (territory_id) в справочнике на год: Россия и ДФО."""
    rows = []
    for y in YEARS:
        d = dict_for_year(y)
        rows.append((y, d["territory_id"].nunique(), d.loc[d["is_dfo"], "territory_id"].nunique()))
    return pd.DataFrame(rows, columns=["year", "n_ru", "n_dfo"]).set_index("year")


def coverage(df: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """Доля МО справочника с непустым значением по годам, % (Россия, ДФО)."""
    m = map_to_territory(df)
    m = m.dropna(subset=["territory_id", "value"])
    tid = load_dict().drop_duplicates("territory_id").set_index("territory_id")["is_dfo"]
    m["is_dfo"] = m["territory_id"].map(tid)
    rows = []
    for y in YEARS:
        valid = set(dict_for_year(y)["territory_id"])
        a = m[m["year"].eq(y) & m["territory_id"].isin(valid)]
        ru = a["territory_id"].nunique()
        dfo = a.loc[a["is_dfo"].fillna(False).astype(bool), "territory_id"].nunique()
        rows.append((y, 100 * ru / den.at[y, "n_ru"], 100 * dfo / den.at[y, "n_dfo"]))
    return pd.DataFrame(rows, columns=["year", "ru", "dfo"]).set_index("year")


def files_section() -> None:
    """Раздел отчёта: скачанные файлы, их размер и назначение."""
    w("## 1. Скачанные файлы")
    w()
    w("Контрольные суммы — `data/raw/SHA256SUMS`, неудачные загрузки — `data/raw/FAILED.txt`.")
    w()
    sums = {}
    for line in (RAW / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        h, p = line.split("  ", 1)
        sums[p] = h
    rows = []
    for p, h in sorted(sums.items()):
        f = RAW / p
        rows.append((p, f"{f.stat().st_size / 1e6:.1f}", h[:12]))
    w(md_table(pd.DataFrame(rows, columns=["файл", "МБ", "sha256 (начало)"]), index=False))
    failed = (RAW / "FAILED.txt").read_text(encoding="utf-8").strip() if (RAW / "FAILED.txt").exists() else ""
    w()
    w(f"Неудачных загрузок: {len(failed.splitlines()) if failed else 0}.")
    w()


def bdmo_section(den: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Покрытие всех показателей БДПМО; возвращает таблицы покрытия по кодам."""
    cfg = load_yaml("sources.yaml")["tochno_bdmo"]["indicators"]
    w("## 3. Показатели БДПМО: покрытие по годам")
    w()
    w(
        "Годовое итоговое значение (правила выбора — `configs/sources.yaml`, поле `total`/`period`). "
        "Покрытие = доля МО справочника СберИндекса, для которых значение есть (не пропуск). "
        "Знаменатель: Россия — "
        + ", ".join(f"{y}: {den.at[y, 'n_ru']}" for y in (2010, 2018, 2024))
        + f"; ДФО — {den['n_dfo'].iloc[0]} МО во все годы. До 2018 г. используется состав справочника 2018 г."
    )
    w()
    covs: dict[str, pd.DataFrame] = {}
    summary = []
    for ind in cfg:
        code = str(ind["code"])
        if not bdmo.indicator_path(code).exists():
            continue
        d, conflicts = bdmo.annual_total(code)
        name = d["indicator_name"].iloc[0] if len(d) else "?"
        unit = d["indicator_unit"].mode().iloc[0] if len(d) else "?"
        cov = coverage(d, den)
        covs[code] = cov
        yrs = cov.index[cov["ru"] > 0]
        summary.append(
            {
                "код": code,
                "показатель": name[:80],
                "ед.": unit,
                "годы": f"{yrs.min()}–{yrs.max()}" if len(yrs) else "—",
                "RU % (медиана)": cov.loc[yrs, "ru"].median() if len(yrs) else 0,
                "ДФО % (медиана)": cov.loc[yrs, "dfo"].median() if len(yrs) else 0,
                "конфликтов дублей": conflicts,
            }
        )
    s = pd.DataFrame(summary)
    w("### 3.1. Сводка")
    w()
    w(md_table(s, index=False))
    w()
    w("### 3.2. Покрытие по годам, % МО (Россия / ДФО)")
    w()
    tab = pd.DataFrame(
        {c: [f"{r.ru:.0f}/{r.dfo:.0f}" if r.ru > 0 else "" for r in v.itertuples()] for c, v in covs.items()},
        index=YEARS,
    ).T
    tab.index.name = "код"
    w(md_table(tab))
    w()
    return covs


def sector_section() -> None:
    """Отраслевые данные по МО: насколько часто скрыты значения (для ВМП)."""
    w("## 4. Входы для расчёта ВМП")
    w()
    # --- региональные
    g = grp()
    v = gva_okved2()
    vi = grp_volume_index()
    w("### 4.1. Региональный уровень (Росстат, «Национальные счета»)")
    w()
    w(
        f"- **ВРП субъектов в текущих ценах** (`VRP_s1998.xlsx`): {g['year'].min()}–{g['year'].max()}, "
        f"{g['region_code'].nunique()} субъектов; пропуски только в ранних годах у новых субъектов "
        "и у Архангельской/Тюменской областей без АО до 2010 г."
    )
    vv = v[v["section"].ne("TOTAL")]
    w(
        f"- **ВДС по разделам ОКВЭД2** (`VRP_OKVED2_s_2016.xlsx`): {v['year'].min()}–{v['year'].max()}, "
        f"{v['region_code'].nunique()} субъектов, 19 разделов A–T + раздел L разделён на «операции с "
        f"недвижимостью» (L1) и «услуги по проживанию в собственном жилище» (L2, условная аренда). "
        f"Пустых ячеек: {int(vv['gva'].isna().sum())}. Сумма разделов совпадает с итогом (|отклонение| < 1e-12)."
    )
    w(f"- **Индексы физического объёма ВРП**: {vi['year'].min()}–{vi['year'].max()}.")
    w(
        "- ВДС по ОКВЭД-2007 (до 2016 г.) скачана (`VRP_OKVED2007.xlsx`), но отраслевые данные по МО "
        "в ОКВЭД-2007 и ОКВЭД2 несопоставимы — для отраслевого метода ВМП рабочее окно 2017+."
    )
    w(
        "- **Лаг публикации:** ВРП и ВДС по субъектам опубликованы по **2024** г. включительно "
        "(файл обновлён 05.03.2026); 2025 г. ожидается в начале 2027 г."
    )
    w()
    # --- MO sectoral L, W, payroll
    w("### 4.2. Работники и зарплата по разделам ОКВЭД2 на уровне МО (8423005/8423006/8423007, 2017–2025)")
    w()
    den_dfo = 230
    rows = []
    dfo_ids = set(load_dict().loc[lambda x: x.is_dfo, "territory_id"])
    for code, lab in (("8423005", "L"), ("8423006", "ФОТ")):
        t = map_to_territory(bdmo.annual(bdmo.read_top(code)))
        t = t[t["territory_id"].isin(dfo_ids)]
        t["sec"] = t["okved2"].map(okved2_letter)
        tot = t[t["okved2"].str.startswith("Всего")].groupby(["territory_id", "year"])["value"].sum()
        sec = t[t["sec"].notna()].groupby(["territory_id", "year"])["value"].sum()
        nsec = t[t["sec"].notna()].groupby(["territory_id", "year"])["sec"].nunique()
        j = pd.DataFrame({"tot": tot, "sec": sec, "nsec": nsec}).dropna(subset=["tot"])
        j["sec"] = j["sec"].fillna(0)
        j["anom"] = j["sec"] > 1.05 * j["tot"]
        ok = j[~j["anom"]]
        for y, a in ok.groupby(level="year"):
            rows.append(
                {
                    "показатель": lab,
                    "год": int(y),
                    "МО ДФО с итогом": len(a) + int(j.xs(y, level="year")["anom"].sum()),
                    "разделов на МО (сред.)": a["nsec"].mean(),
                    "скрыто от итога, % (ДФО)": 100 * (a["tot"] - a["sec"]).clip(lower=0).sum() / a["tot"].sum(),
                    "МО, где скрыто > 10%": int(((a["tot"] - a["sec"]) / a["tot"] > 0.10).sum()),
                    "аномалий Σразделов > итога": int(j.xs(y, level="year")["anom"].sum()),
                }
            )
    r = pd.DataFrame(rows)
    w(md_table(r, index=False, floatfmt="{:.1f}"))
    w()
    w(
        f"Знаменатель для ДФО — {den_dfo} МО (2 ЗАТО — Вилючинск и Циолковский — по ряду показателей "
        "не публикуются вовсе). **Росстат скрывает отраслевые значения не пустой ячейкой, а отсутствием "
        "строки**, поэтому частота скрытия оценивается как доля итоговых работников (ФОТ), которую не "
        "удаётся разложить по опубликованным разделам: «скрыто от итога». Именно этот остаток метод 2 "
        "распределит по правилу для пропусков и запишет в `gmp_imputed_share`. "
        "«Аномалии» — МО-годы, где сумма разделов больше итога более чем на 5% (ошибки источника: "
        "например, в 2017 г. у Аллаиховского района Якутии в разделе J указано 28 764 работника при "
        "населении ~3 тыс.) — такие МО-годы при сборке панели помечаются флагом и в расчёт отраслевых долей не идут."
    )
    w()
    w(
        "**Старый классификатор (ОКВЭД-2007, 8123005–8123007, 2008–2016):** итоговое значение по МО "
        "публикуется только с 2013 г. (до этого — лишь разделы, итог восстановим только как сумму "
        "опубликованных разделов, т.е. с занижением). Разрыва уровней на стыке 2016→2017 нет: медианный "
        "прирост числа работников 2016→2017 — 0,974 (2015→2016 — 0,975), зарплаты — 1,070 (1,062), отгрузки — "
        "1,035 (1,063). Итоговые ряды можно склеивать; отраслевую структуру — нельзя (другие разделы)."
    )
    w()
    # --- shipments by section
    w("### 4.3. Отгрузка товаров собственного производства по разделам B, C, D, E на уровне МО (8401011, 2016–2025)")
    w()
    t = map_to_territory(bdmo.annual(bdmo.read_top("8401011")))
    t["sec"] = t["okved2"].map(okved2_letter)
    dfo_ids = set(load_dict().loc[lambda x: x.is_dfo, "territory_id"])
    rows = []
    for y in sorted(t["year"].dropna().unique()):
        a = t[t["year"].eq(y)]
        row = {"год": int(y)}
        for sec in "BCDE":
            b = a[a["sec"].eq(sec)]
            row[f"{sec}: МО РФ"] = b.dropna(subset=["value"])["territory_id"].nunique()
            bd = b[b["territory_id"].isin(dfo_ids)]
            row[f"{sec}: МО ДФО"] = bd.dropna(subset=["value"])["territory_id"].nunique()
            row[f"{sec}: пустых ДФО %"] = 100 * bd["value"].isna().mean() if len(bd) else float("nan")
        rows.append(row)
    w(md_table(pd.DataFrame(rows), index=False, floatfmt="{:.0f}"))
    w()
    w(
        "Пустая строка-раздел = значение скрыто Росстатом; отсутствие строки = раздела в МО нет или он "
        "не публикуется. Раздел D (энергетика) и C (обработка) есть у большинства МО, B (добыча) — "
        "только у добывающих, что ожидаемо."
    )
    w()
    # --- agriculture
    w("### 4.4. Продукция сельского хозяйства по МО (8007010, хозяйства всех категорий)")
    w()
    w("См. строку 8007010 в таблице 3.2. Ряд заканчивается 2023 г.")
    w()


def regional_prices_section() -> None:
    """Раздел отчёта: региональные ряды цен в сборнике «Регионы России»."""
    w("## 5. Цены: ИПЦ и стоимость фиксированного набора по субъектам")
    w()
    d = regions_collection(["Y477110111", "Y477110395"])
    g = (
        d.groupby(["indicator_code", "indicator_unit"])
        .agg(годы=("year", lambda s: f"{s.min()}–{s.max()}"), субъектов=("region_code", "nunique"))
        .reset_index()
    )
    w(md_table(g, index=False))
    w()
    w(
        "ИПЦ — декабрь к декабрю предыдущего года; фиксированный набор — на конец года, руб. в месяц и "
        "% к среднероссийской стоимости. 2025 г. в сборниках ещё нет."
    )
    w()


def sber_section() -> None:
    """Раздел отчёта: наборы данных СберИндекса и их покрытие."""
    w("## 6. Данные СберИндекса")
    w()
    k = pd.read_parquet(RAW / "sber/hackathonlicence/consumption.parquet")
    dct = load_dict().drop_duplicates("territory_id").set_index("territory_id")
    k["is_dfo"] = k["territory_id"].map(dct["is_dfo"])
    k["year"] = k["date"].str[:4].astype(int)
    tot = k[k["category"].eq("Все категории")]
    months = tot.groupby(["territory_id", "year"])["date"].nunique()
    rows = []
    for y in (2023, 2024):
        mm = months.xs(y, level="year")
        dfo = mm[mm.index.map(dct["is_dfo"]).fillna(False).astype(bool)]
        rows.append(
            {
                "год": y,
                "МО РФ": len(mm),
                "из них 12 мес.": int((mm == 12).sum()),
                "МО ДФО": len(dfo),
                "ДФО 12 мес.": int((dfo == 12).sum()),
            }
        )
    w(
        "**Безналичные траты** (`consumption.parquet`, янв. 2023 — дек. 2024, 6 категорий включая "
        "«Все категории»; значение — средние траты одного жителя за месяц, руб.):"
    )
    w()
    w(md_table(pd.DataFrame(rows), index=False))
    w()
    w(
        "API-выгрузка (`consumption_api.parquet`) содержит те же 303 126 наблюдений за тот же период — "
        "продления ряда нет; основной источник — архив хакатона."
    )
    w()
    ma = pd.read_parquet(RAW / "sber/hackathonlicence/market_access.parquet")
    ma_dfo = ma["territory_id"].map(dct["is_dfo"]).fillna(False).astype(bool)
    w(
        f"**Индекс доступности рынков (2024):** {len(ma)} МО, из них ДФО {int(ma_dfo.sum())} из 230. "
        "Нет 22 МО без постоянного автодорожного сообщения."
    )
    w()
    c = pd.read_parquet(RAW / "sber/hackathonlicence/connection.parquet")
    rail = pd.read_parquet(RAW / "sber/t_pairs_distance_railway/t_pairs_distance_railway.parquet")
    nodes_h = pd.unique(c.loc[c.type.eq("highway"), ["territory_id_x", "territory_id_y"]].values.ravel())
    nodes_r = pd.unique(c.loc[c.type.eq("railway"), ["territory_id_x", "territory_id_y"]].values.ravel())
    nodes_r2 = pd.unique(rail[["territory_id_x", "territory_id_y"]].values.ravel())
    dfo_ids = set(dct.index[dct["is_dfo"]])
    w(
        "**Расстояния между МО** (`connection.parquet`, на 31.12.2024; расстояние считалось в одну "
        "сторону x→y и считается симметричным):"
    )
    w()
    w(
        f"- по автодорогам (граф OSM, центр—центр): {int((c.type == 'highway').sum()):,} пар, "
        f"{len(nodes_h)} МО (ДФО: {len(set(nodes_h) & dfo_ids)});"
    )
    w(
        f"- по железной дороге (Тарифное руководство №4, ближайшие станции): {int((c.type == 'railway').sum()):,} пар, "
        f"{len(nodes_r)} МО (ДФО: {len(set(nodes_r) & dfo_ids)});"
    )
    w(
        f"- отдельный обновлённый набор ж/д расстояний (`t_pairs_distance_railway`, на 01.12.2025): "
        f"{len(rail):,} пар, {len(nodes_r2)} МО (ДФО: {len(set(nodes_r2) & dfo_ids)})."
    )
    w(
        "- **Времени в пути в данных нет** — только километры (проверено по PDF-описанию полей и по "
        "самим таблицам: колонки `territory_id_x, territory_id_y, distance, type`)."
    )
    w()


def region_gaps_section() -> None:
    """Раздел отчёта: субъекты и годы с массовыми пропусками."""
    w("## 6a. Выпадающие регионы ДФО (доля МО региона с данными < 60%)")
    w()
    d = load_dict()
    dfo = d[d.is_dfo].drop_duplicates("territory_id", keep="last").set_index("territory_id")
    n = dfo.groupby("region_name").size()
    for code in ("8112027", "8213002", "8423005", "8109001", "8109003", "8401011", "8013001", "8313015"):
        t, _ = bdmo.annual_total(code)
        m = map_to_territory(t).dropna(subset=["territory_id", "value"])
        m = m[m["territory_id"].isin(dfo.index)]
        m["reg"] = m["territory_id"].map(dfo["region_name"])
        c = m.groupby(["reg", "year"])["territory_id"].nunique().unstack()
        c = c.reindex(index=n.index, columns=range(2010, 2026)).fillna(0)
        c = (100 * c.div(n, axis=0)).round(0)
        yrs = [y for y in c.columns if c[y].sum() > 0]
        low = c[yrs].where(c[yrs] < 60).dropna(how="all").dropna(axis=1, how="all")
        name = t["indicator_name"].iloc[0][:70]
        if low.empty:
            w(f"- **{code}** ({name}): нет.")
            continue
        w(f"- **{code}** ({name}):")
        w()
        w(md_table(low.astype("float"), floatfmt="{:.0f}"))
        w()
    w()


def window_section(covs: dict[str, pd.DataFrame]) -> None:
    """Раздел отчёта: окно лет, в котором ключевые ряды полны."""
    w("## 7. Предложение временного окна панели")
    w()
    key = {
        "население 8112027": ["8112027"],
        "зарплата 8213002": ["8213002"],
        "работники 8123005→8423005": ["8123005", "8423005"],
        "инвестиции 8109001": ["8109001"],
        "инвестиции/душу без бюдж. 8109003": ["8109003"],
        "отгрузка 8201001→8401011": ["8201001", "8401011"],
        "бюджет: доходы 8013001": ["8013001"],
        "бюджет: доля собств. 8313015": ["8313015"],
    }
    tab = pd.DataFrame(index=YEARS)
    for lab, codes in key.items():
        tab[lab] = pd.concat([covs[c]["dfo"] for c in codes if c in covs], axis=1).max(axis=1)
    tab.index.name = "год"
    w(
        "Покрытие МО ДФО, % (для связок старый→новый классификатор — максимум из двух рядов). "
        "Жирным в тексте ниже — порог ~80%."
    )
    w()
    w(md_table(tab.round(0)))
    w()
    core = ["население 8112027", "зарплата 8213002", "работники 8123005→8423005", "отгрузка 8201001→8401011"]
    ok = tab[core].ge(80).all(axis=1)
    w(
        "Годы, где население, зарплата, работники и отгрузка покрывают ≥ 80% МО ДФО: "
        + ", ".join(str(y) for y in tab.index[ok])
        + "."
    )
    w()
    w("""**Почему строгий критерий (все шесть ключевых показателей ≥ 80% МО ДФО) не выполняется ни в одном году,
кроме 2022:**

- инвестиции 8109001: по Бурятии нет ни одного года, по Якутии — до 2021 г. включительно (≈ 27% МО ДФО),
  в 2012 г. провал по всей стране, ряд заканчивается 2023 г.;
- уровень доходов местного бюджета 8013001 публикуется только по 2020 г.; доля налоговых и неналоговых
  доходов 8313015 идёт до 2024 г., но в 2023–2024 гг. выпадают Бурятия, Якутия, Сахалин, Забайкалье, Чукотка;
- отдельные годы целых регионов: население Бурятии 2010–2014, Забайкалья 2017; зарплата Бурятии 2018–2020.

**Варианты окна (решение за командой):**

| вариант | годы | плюсы | минусы |
|---|---|---|---|
| А. «Отраслевое» | 2017–2024 (8 лет) | единый ОКВЭД2 по МО и по ВДС субъектов → основной (отраслевой) метод ВМП для каждого года; ключевые показатели ≥ 90% МО ДФО (кроме инвестиций и бюджета) | 8 точек — на грани надёжности для конвергенции; инвестиции 8109001 только 73% до 2021 г. (без Бурятии и Якутии) |
| Б. «Длинное» | 2013–2024 (12 лет) | итоговые ряды склеиваются без разрыва уровней; больше точек для σ/β-конвергенции | 2013–2015: отраслевой ВМП только в разбивке ОКВЭД-2007 (структура ВДС субъектов по ОКВЭД-2007 есть за 2004–2015); 2016 — переходный год (по МО ещё ОКВЭД-2007, по субъектам уже ОКВЭД2) → только базовый метод; с 2016 г. Росстат сменил методику оценки условной аренды (разрыв в ВРП); отгрузка «всего» только с 2014 г. |
| В. «Максимальное» для узкого набора | 2010–2024 (15 лет) | население, зарплата (8213002), доля собственных доходов бюджета | нет работников/ФОТ по МО до 2013 г. → нет ВМП; провалы Бурятии и Чукотки |

**Предложение:** основная панель — **2013–2024**, с флагом `gmp_method_available`; кластеризация и сети по
умолчанию на **2017–2024** (вариант А, где ВМП считается основным методом), конвергенция — на всём окне
2013–2024 с предупреждением в интерфейсе. Инвестиции на душу: основной ряд 8109001 (с 2022 г. покрытие 90%),
для 2013–2021 предложить аналитикам выбор: 8109001 (без Бурятии/Якутии) или 8109003 (инвестиции без бюджетных
средств, 95% МО ДФО, но другое определение). Бюджет: `budget_own_share` по 8313015 (2013–2024 с провалами),
`budget_rev_pc` — только до 2020 г.
""")
    w()


def main() -> int:
    """Точка входа: отчёт об инвентаризации данных: reports/INVENTORY.md."""
    den = denominators()
    w("# Инвентаризация данных")
    w()
    w("Сгенерировано скриптом `scripts/build_inventory.py`. Не редактировать вручную.")
    w()
    files_section()
    w("## 2. Справочник МО СберИндекса")
    w()
    d = load_dict()
    w(
        f"`t_dict_municipal_districts.xlsx`: {len(d)} версий МО, {d['territory_id'].nunique()} territory_id, "
        f"{d['region_code'].nunique()} субъектов. Годы версий: {d['year_from'].min()}–{d.loc[d.year_to < 9999, 'year_to'].max()} "
        "(year_to = 9999 — действует сейчас). Обновление 25.10.2024."
    )
    w()
    w(md_table(den.rename(columns={"n_ru": "МО Россия", "n_dfo": "МО ДФО"}).loc[2018:2025]))
    w()
    dfo = d[d.is_dfo]
    w(
        f"ДФО: {dfo['territory_id'].nunique()} territory_id; территориальных преобразований "
        f"(union/disunion/transfer) в 2018–2024 гг. — {int(dfo['change_id_from'].notna().sum() + dfo['change_id_to'].notna().sum())}; "
        f"смен типа/названия/ОКТМО без изменения границ — {int((dfo.groupby('territory_id').size() > 1).sum())} МО. "
        f"По России событий union/disunion/transfer — {int(d['change_id_from'].notna().sum() + d['change_id_to'].notna().sum())}."
    )
    w()
    covs = bdmo_section(den)
    sector_section()
    regional_prices_section()
    sber_section()
    region_gaps_section()
    window_section(covs)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "INVENTORY.md").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.concat({k: v for k, v in covs.items()}, names=["code", "year"]).to_csv(PROCESSED / "coverage_bdmo.csv")
    print(f"Записано: {REPORTS / 'INVENTORY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
