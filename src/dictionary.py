"""Справочник МО СберИндекса: версии муниципалитетов, привязка ОКТМО → territory_id."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.io import RAW, load_yaml

DICT_DIR = RAW / "sber" / "t_dict_municipal"
DICT_XLSX = DICT_DIR / "t_dict_municipal_districts.xlsx"
DICT_GPKG = DICT_DIR / "t_dict_municipal_districts_poly.gpkg"

# year_to = 9999 означает «действует до сих пор»; справочник покрывает 01.01.2018–01.01.2024
DICT_FIRST_YEAR = 2018


def norm_oktmo(s: pd.Series) -> pd.Series:
    """Приводит ОКТМО к 8 цифрам: «79-701-000-000» → «79701000», «7548000» → «07548000»."""
    x = s.astype("string").str.replace(r"\D", "", regex=True)
    x = x.where(x.str.len() != 11, x.str[:8])
    return x.str.zfill(8)


def region_to_fo() -> dict[int, str]:
    """Код субъекта → федеральный округ."""
    fo = load_yaml("regions.yaml")["federal_districts"]
    return {code: name for name, codes in fo.items() for code in codes}


@lru_cache(maxsize=1)
def load_dict() -> pd.DataFrame:
    """Все версии МО из справочника СберИндекса с нормализованным ОКТМО и ФО."""
    d = pd.read_excel(DICT_XLSX, dtype={"oktmo": str})
    d["oktmo8"] = norm_oktmo(d["oktmo"])
    fo = region_to_fo()
    d["federal_district"] = d["region_code"].map(fo)
    d["is_dfo"] = d["federal_district"].eq("ДФО")
    return d


def dict_for_year(year: int) -> pd.DataFrame:
    """Версии МО, действовавшие в году year: year_from <= Y < year_to.

    Для лет до 2018 справочник не определён — берётся состав 2018 года
    (длинные ряды строятся через oktmo_stable из набора «Если быть точным»).
    """
    d = load_dict()
    y = max(year, DICT_FIRST_YEAR)
    return d[(d.year_from <= y) & (d.year_to > y)]


def map_to_territory(df: pd.DataFrame, oktmo: str = "oktmo8", stable: str = "oktmo_stable8") -> pd.DataFrame:
    """Привязывает строки (ОКТМО × год) к territory_id. Возвращает df + territory_id, match.

    Порядок:
    1. by_year   — ОКТМО есть в версии справочника, действовавшей в этом году
                   (для лет до 2018 — в версии 2018 года);
    2. by_oktmo  — ОКТМО встречается в справочнике ровно у одного territory_id (любой год);
    3. by_stable — то же для последнего действующего ОКТМО (oktmo_stable из набора tochno.st);
    4. by_stable_peer — через другие строки df с тем же oktmo_stable (новые коды после 2024 г.).
    Остальное: match = «нет в справочнике» или «неоднозначно» (territory_id пустой).
    """
    d = load_dict()
    out = df.copy()
    out["territory_id"] = pd.NA
    out["match"] = pd.NA

    years = out["year"].dropna().unique()
    for y in years:
        v = dict_for_year(int(y))[["oktmo8", "territory_id"]].drop_duplicates("oktmo8")
        m = out["year"].eq(y) & out["territory_id"].isna()
        tid = out.loc[m, oktmo].map(v.set_index("oktmo8")["territory_id"])
        out.loc[m, "territory_id"] = tid
        out.loc[m & tid.notna().reindex(out.index, fill_value=False), "match"] = "by_year"

    uniq = d.groupby("oktmo8")["territory_id"].nunique()
    one = d[d["oktmo8"].isin(uniq[uniq == 1].index)].drop_duplicates("oktmo8").set_index("oktmo8")["territory_id"]
    amb = set(uniq[uniq > 1].index)
    for col, tag in ((oktmo, "by_oktmo"), (stable, "by_stable")):
        if col not in out:
            continue
        m = out["territory_id"].isna()
        tid = out.loc[m, col].map(one)
        out.loc[tid.dropna().index, "territory_id"] = tid.dropna()
        out.loc[tid.dropna().index, "match"] = tag

    # 4. by_stable_peer — код новее справочника (например, район стал округом в конце 2024 г.):
    #    берём territory_id других строк с тем же oktmo_stable, уже привязанных шагами 1–3,
    #    если он единственный.
    if stable in out:
        peers = out.dropna(subset=["territory_id"]).groupby(stable)["territory_id"].agg(["nunique", "first"])
        peers = peers[peers["nunique"] == 1]["first"]
        m = out["territory_id"].isna()
        tid = out.loc[m, stable].map(peers).dropna()
        out.loc[tid.index, "territory_id"] = tid
        out.loc[tid.index, "match"] = "by_stable_peer"

    m = out["territory_id"].isna()
    ambiguous = out[oktmo].isin(amb) | (out[stable].isin(amb) if stable in out else False)
    out.loc[m & ambiguous, "match"] = "неоднозначно"
    out.loc[m & ~ambiguous, "match"] = "нет в справочнике"
    out["territory_id"] = out["territory_id"].astype("Int64")
    return out


def territories() -> pd.DataFrame:
    """Одна строка на territory_id: последняя версия (название, тип, ОКТМО, регион, ФО)."""
    d = load_dict().sort_values(["territory_id", "year_from"])
    last = d.groupby("territory_id").tail(1).copy()
    first = d.groupby("territory_id")[["year_from"]].min().rename(columns={"year_from": "first_year"})
    return last.merge(first, on="territory_id")
