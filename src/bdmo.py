"""Чтение показателей БДПМО Росстата в обработке «Если быть точным» (tochno.st/datasets/bdmo).

Файл показателя — zip с CSV (UTF-8, «;»). Общие атрибуты описаны в
description_bdpmo_v20250918.pdf; специальные атрибуты (okved2, istinv, mest, ...)
задают разрезы показателя.
"""

from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.dictionary import norm_oktmo
from src.io import RAW, load_yaml

BDMO_DIR = RAW / "tochno" / "bdmo"
TOP_LEVEL = "Муниципальное образование верхнего уровня"

# Общие (не разрезные) атрибуты набора
COMMON = [
    "indicator_section_code",
    "indicator_section",
    "indicator_code",
    "indicator_name",
    "region_id",
    "region_name",
    "mun_level",
    "mun_district",
    "municipality",
    "oktmo",
    "mun_type",
    "mun_type_oktmo",
    "oktmo_stable",
    "oktmo_history",
    "oktmo_year_from",
    "oktmo_year_to",
    "year",
    "indicator_value",
    "indicator_unit",
    "indicator_period",
    "comment",
]
MISSING_CODES = {"ND", "UD", "CD"}
LOWER_TYPES = {"Сельское поселение", "Городское поселение", "Внутригородской район", "Межселенная территория"}


def indicator_path(code: int | str) -> Path:
    """Путь к zip-файлу показателя в data/raw/."""
    b = load_yaml("sources.yaml")["tochno_bdmo"]
    for ind in b["indicators"]:
        if str(ind["code"]) == str(code):
            url = b["indicator_url"].format(section=ind["section"], code=ind["code"])
            return RAW / b["dest_dir"] / url.rsplit("/", 1)[-1]
    raise KeyError(f"Показатель {code} не описан в configs/sources.yaml")


def is_summary_oktmo(oktmo8: pd.Series) -> pd.Series:
    """Сводные коды («все муниципальные районы региона» и т.п.): цифры 4–5 ОКТМО = «00».

    Проверено на справочнике СберИндекса: у всех настоящих МО верхнего уровня эти цифры
    ненулевые, а последние три цифры — «000».
    """
    return oktmo8.str[3:5].eq("00") | oktmo8.str[5:].ne("000")


def read_raw(code: int | str, top_only: bool = True) -> pd.DataFrame:
    """Строки показателя как есть (все колонки — строки).

    top_only=True оставляет только mun_level = «МО верхнего уровня» уже при чтении parquet
    (у показателей населения 90% строк — поселения, так экономится память).
    """
    path = indicator_path(code)
    # В архиве большого показателя лежат полный CSV, parquet и его части (_parts/ по годам
    # и регионам) — читаем только корневой parquet, иначе корневой CSV, чтобы не задвоить.
    with zipfile.ZipFile(path) as z:
        root = [n for n in z.namelist() if "/" not in n]
        pq = [n for n in root if n.endswith(".parquet")]
        if pq:
            filters = [("mun_level", "==", TOP_LEVEL)] if top_only else None
            d = pd.read_parquet(z.open(pq[0]), filters=filters)
            d = d.astype("string").fillna("")
            return pd.DataFrame({c: d[c].astype(str) for c in d.columns})
        csv = [n for n in root if n.endswith(".csv")]
        frames = [pd.read_csv(z.open(n), sep=";", dtype=str, keep_default_na=False) for n in csv]
    d = pd.concat(frames, ignore_index=True)
    return d[d["mun_level"].eq(TOP_LEVEL)] if top_only else d


def dims(df: pd.DataFrame) -> list[str]:
    """Специальные атрибуты (разрезы) показателя."""
    return [c for c in df.columns if c not in COMMON]


def read_top(code: int | str) -> pd.DataFrame:
    """Строки МО верхнего уровня без сводных кодов; числовое значение и нормализованные ОКТМО.

    Пропуск значения остаётся NaN (не ноль). Результат кэшируется, поэтому
    вызывающий код не должен менять таблицу на месте.
    """
    return _read_top(str(code))


@lru_cache(maxsize=4)
def _read_top(code: str) -> pd.DataFrame:
    d = read_raw(code)
    # Строки «верхнего уровня» с типом поселения — это суммы поселений района с неполным
    # ОКТМО (см. раздел 5 описания набора, пример 8112035), а не сам район.
    d = d[d["mun_level"].eq(TOP_LEVEL) & ~d["mun_type"].isin(LOWER_TYPES)].copy()
    d["oktmo8"] = norm_oktmo(d["oktmo"])
    d = d[~is_summary_oktmo(d["oktmo8"])]
    st = d["oktmo_stable"].where(~d["oktmo_stable"].isin(MISSING_CODES))
    d["oktmo_stable8"] = norm_oktmo(st.fillna(d["oktmo"]))
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    d = d[d["year"].ne(9999)]
    v = d["indicator_value"].str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    d["value"] = pd.to_numeric(v, errors="coerce")
    return d


ANNUAL_PERIODS = ["Январь-декабрь", "Значение показателя за год", "На 1 января", "На конец года"]
TOTAL_RE = r"^(?:Всего|Все население|Хозяйства всех категорий|Итого)"


def indicator_meta(code: int | str) -> dict:
    """Запись о показателе из configs/sources.yaml."""
    for ind in load_yaml("sources.yaml")["tochno_bdmo"]["indicators"]:
        if str(ind["code"]) == str(code):
            return ind
    raise KeyError(code)


def select_total(d: pd.DataFrame, code: int | str, rules: dict | None = None) -> pd.DataFrame:
    """Оставляет строки «итого» по всем разрезам показателя.

    Правило rules (переопределение) и правило из sources.yaml (поле total: {разрез: [значения]})
    имеют приоритет;
    для остальных разрезов берётся значение, начинающееся с «Всего»/«Итого»/…,
    если оно есть. Разрезы без итогового значения не фильтруются.
    """
    rules = {**(indicator_meta(code).get("total", {}) or {}), **(rules or {})}
    for c in dims(d):
        if c in ("oktmo8", "oktmo_stable8", "value", "_pref"):
            continue
        if c in rules:
            vals = rules[c] if isinstance(rules[c], list) else [rules[c]]
            # порядок в списке = предпочтение при дублях (см. annual_total)
            d = d[d[c].isin(vals)].assign(_pref=lambda x, c=c, v=vals: x[c].map({k: i for i, k in enumerate(v)}))
        else:
            tot = d[c].str.contains(TOTAL_RE, regex=True, na=False)
            if tot.any():
                d = d[tot]
    return d


def annual(d: pd.DataFrame, period: str | None = None) -> pd.DataFrame:
    """Годовые значения: для каждой пары (ОКТМО, год, разрезы) — первый период из ANNUAL_PERIODS.

    Квартальные нарастающие итоги (январь-март и т.п.) отбрасываются.
    """
    order = [period] if period else ANNUAL_PERIODS
    d = d[d["indicator_period"].isin(order)].copy()
    d["_p"] = d["indicator_period"].map({p: i for i, p in enumerate(order)})
    keys = ["oktmo8", "year"] + [c for c in dims(d) if c not in ("oktmo8", "oktmo_stable8", "value", "_p", "_pref")]
    best = d.groupby(keys, dropna=False)["_p"].transform("min")
    return d[d["_p"].eq(best)].drop(columns="_p")


def annual_total(code: int | str, rules: dict | None = None) -> tuple[pd.DataFrame, int]:
    """Годовое итоговое значение показателя по МО верхнего уровня.

    Возвращает (таблица oktmo8 × year с колонкой value, число ключей с конфликтом значений).
    При дублях берётся первое непустое значение; конфликт (разные непустые значения) считается.
    """
    meta = indicator_meta(code)
    d = annual(select_total(read_top(code), code, rules), meta.get("period"))
    pref = set((meta.get("total") or {}) | (rules or {}))
    for c in dims(d):
        if c in ("oktmo8", "oktmo_stable8", "value", "_pref") or c in pref:
            continue
        if (d.groupby(["oktmo8", "year"])[c].nunique() > 1).any():
            raise ValueError(f"{code}: в разрезе {c} нет однозначного итога — задайте total в sources.yaml")
    g = d.groupby(["oktmo8", "year"])
    conflicts = int((g["value"].nunique() > 1).sum())
    pref = d["_pref"] if "_pref" in d else 0
    d = d.assign(_na=d["value"].isna(), _pref=pref).sort_values(["_na", "_pref"], kind="stable")
    d = d.drop_duplicates(["oktmo8", "year"], keep="first").drop(columns=["_na", "_pref"])
    return d, conflicts
