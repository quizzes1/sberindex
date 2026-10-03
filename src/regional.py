"""Региональные данные Росстата: ВРП, ВДС по разделам ОКВЭД2, индексы физического объёма,
ИПЦ и стоимость фиксированного набора (сборники Росстата в обработке «Если быть точным»).

Все функции возвращают «длинные» таблицы с колонкой region_code (код субъекта как в
справочнике СберИндекса). Архангельская и Тюменская области берутся БЕЗ автономных
округов — так же, как МО в справочнике (НАО, ХМАО, ЯНАО — отдельные субъекты).
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from src.dictionary import load_dict
from src.io import RAW

ROSSTAT = RAW / "rosstat"
REGIONS_PARQUET = RAW / "tochno" / "regions" / "data_regions_collection_102_v20260313.parquet"

# Разделы ОКВЭД2 (латиница). L1 — операции с недвижимостью, L2 — услуги по проживанию
# в собственном жилище (условно исчисленная аренда).
OKVED2_SECTIONS = list("ABCDEFGHIJKLMNOPQRST")
_CYR2LAT = str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX")


def okved2_letter(label: str) -> str | None:
    """«Раздел Н Транспортировка...» → 'H' (кириллические буквы-двойники → латиница)."""
    m = re.search(r"Раздел\s+([A-Za-zА-Яа-я])\b", str(label))
    return m.group(1).upper().translate(_CYR2LAT) if m else None


def _key(name: str) -> str:
    s = str(name).lower().replace("ё", "е")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\b(республика|область|край|автономная|автономный|округ|г\.|город|федерального|значения)\b", " ", s)
    s = re.sub(r"[^а-я]+", "", s)
    return s


@lru_cache(maxsize=4096)
def region_code_from_name(name: str) -> int | None:
    """Название субъекта из таблицы Росстата → код субъекта; None для ФО, РФ и «с АО»."""
    n = re.sub(r"\s+", " ", str(name)).strip()
    low = n.lower()
    if not n or n == "nan" or "федеральный округ" in low or "российская федерация" in low:
        return None
    # сначала «область без АО» (в названии есть имена АО), потом сами АО
    if low.startswith("архангельская"):
        return 29 if "без" in low else None
    if low.startswith("тюменская"):
        return 72 if "без" in low else None
    if "ямало" in low:
        return 89
    if "ханты" in low:
        return 86
    if "ненецкий" in low:
        return 83
    if "кемеровская" in low:
        return 42
    if "москва" in low and "обл" not in low:
        return 77
    if "петербург" in low:
        return 78
    if "севастополь" in low:
        return 92
    keys = {_key(v): k for k, v in load_dict().groupby("region_code")["region_name"].first().items()}
    k = _key(n)
    if k in keys:
        return keys[k]
    for kk, code in keys.items():  # «Кабардино-Балкарская», «Северная Осетия-Алания»
        if k and (kk.startswith(k) or k.startswith(kk)):
            return code
    return None


def _region_rows(df: pd.DataFrame, col: int = 0) -> pd.Series:
    return df.iloc[:, col].map(region_code_from_name)


def grp() -> pd.DataFrame:
    """ВРП субъектов в текущих ценах, млн руб.: region_code, year, grp (1998–последний год)."""
    x = pd.ExcelFile(ROSSTAT / "VRP_s1998.xlsx")
    out = []
    for sheet in ("1", "2"):
        d = x.parse(sheet, header=None)
        years = [int(str(v)[:4]) if str(v)[:4].isdigit() else None for v in d.iloc[2]]
        codes = _region_rows(d)
        for i in d.index[codes.notna()]:
            for j, y in enumerate(years):
                if y:
                    out.append((int(codes[i]), y, pd.to_numeric(d.iat[i, j], errors="coerce")))
    r = pd.DataFrame(out, columns=["region_code", "year", "grp"])
    return r.drop_duplicates(["region_code", "year"], keep="last")


def gva_okved2() -> pd.DataFrame:
    """ВДС субъектов по разделам ОКВЭД2 в текущих ценах, тыс. руб.

    Колонки: region_code, year, section (A…T, L1, L2, TOTAL), gva.
    L1 — операции с недвижимостью без условной аренды, L2 — условно исчисленная аренда.
    """
    x = pd.ExcelFile(ROSSTAT / "VRP_OKVED2_s_2016.xlsx")
    out = []
    for sheet in x.sheet_names:
        m = re.fullmatch(r"2\.\s*(\d{4})", sheet)
        if not m:
            continue
        year = int(m.group(1))
        d = x.parse(sheet, header=None)
        # шапка: строка 3 — «Раздел X», строка 5 — подразделы L
        cols = {1: "TOTAL"}
        for j in range(2, d.shape[1]):
            lab = str(d.iat[3, j])
            sub = str(d.iat[5, j]).lower()
            if lab.startswith("Раздел"):
                cols[j] = okved2_letter(lab)
            elif "собственном жилище" in sub:
                cols[j] = "L2"
            elif "операции с недвижимым" in sub:
                cols[j] = "L1"
        codes = _region_rows(d)
        for i in d.index[codes.notna()]:
            for j, sec in cols.items():
                out.append((int(codes[i]), year, sec, pd.to_numeric(d.iat[i, j], errors="coerce")))
    r = pd.DataFrame(out, columns=["region_code", "year", "section", "gva"])
    return r.drop_duplicates(["region_code", "year", "section"], keep="last")


def grp_volume_index() -> pd.DataFrame:
    """Индекс физического объёма ВРП, % к предыдущему году: region_code, year, grp_vi."""
    x = pd.ExcelFile(ROSSTAT / "VRP_s1998.xlsx")
    out = []
    for sheet in ("5", "6"):
        d = x.parse(sheet, header=None)
        hdr = next(i for i in range(10) if sum(str(v)[:4].isdigit() for v in d.iloc[i]) > 3)
        years = [int(str(v)[:4]) if str(v)[:4].isdigit() else None for v in d.iloc[hdr]]
        codes = _region_rows(d)
        for i in d.index[codes.notna()]:
            for j, y in enumerate(years):
                if y:
                    out.append((int(codes[i]), y, pd.to_numeric(d.iat[i, j], errors="coerce")))
    r = pd.DataFrame(out, columns=["region_code", "year", "grp_vi"])
    return r.drop_duplicates(["region_code", "year"], keep="last")


def regions_collection(codes: list[str] | None = None) -> pd.DataFrame:
    """Сборники Росстата по регионам (tochno.st) с колонкой region_code."""
    d = pd.read_parquet(REGIONS_PARQUET)
    if codes:
        d = d[d["indicator_code"].isin(codes)]
    d = d[d["object_level"].eq("Регион")].copy()
    d["region_code"] = d["object_name"].map(region_code_from_name)
    return d
