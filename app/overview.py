"""Обзор: выборка, цены, сводные таблицы с текущими настройками (бывшая главная страница)."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import (  # noqa: E402
    METHOD_NAMES,
    cluster_color,
    cluster_partition,
    cluster_table1,
    cluster_table2,
    current_partition_params,
    downloads,
    mo,
    plural,
    sample_ids,
    sidebar,
    table2_periods,
)

from src import network, summary

st.set_page_config(page_title="Обзор", page_icon="🗺️", layout="wide")
side = sidebar()

st.title("Обзор")
st.markdown(
    "Мы делим муниципалитеты России на типы по экономическим показателям и смотрим, как они переходят из типа "
    "в тип с 2014 по 2024 год. Каждый год строится сеть: муниципалитеты связаны, если похожи по экономике. "
    "Выборку, годы и цены можно поменять в боковой панели — настройки действуют на всех страницах."
)
m = mo()
ids = sample_ids(side)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Муниципалитетов", f"{len(ids & set(m.loc[m['active'], 'territory_id']))}")
c2.metric("Субъектов", f"{m[m['territory_id'].isin(ids)]['region_code'].nunique()}")
c3.metric("Годы", f"{side['years'][0]}–{side['years'][1]}")
c4.metric("Меняли границы", f"{int(m[m['territory_id'].isin(ids)]['boundary_change'].sum())}")

st.subheader("Что где смотреть")
st.markdown(
    """
- **Данные и качество** — насколько полны данные по годам и регионам, где пропуски, откуда данные.
- **Показатели** — описания показателей, их распределения, корреляции и карта.
- **Сеть** — из каких показателей складывается сходство муниципалитетов и как устроена сеть.
- **Кластеры** — типы муниципалитетов на карте и в сети, их профили, выбор метода и числа типов.
- **Динамика** — кто и когда переходил из типа в тип.
- **Сводные таблицы** — описания типов и переходы по периодам, с выгрузкой в Excel.

Под таблицами на страницах есть кнопки выгрузки: сама таблица (CSV) и параметры, с которыми она получена (YAML).
"""
)
st.info(
    "ВМП (валовой муниципальный продукт) — наша расчётная оценка, а не данные Росстата. Как она считается — "
    "в `reports/METHODS.md`, проверки — в `reports/GMP_CHECKS.md`."
)

# ---------------------------------------------------------------- цены
st.subheader("Цены")
pr = side["prices"]
st.markdown(
    f"""
За 2014–2024 годы цены выросли примерно вдвое, поэтому рубли разных лет сравнивать напрямую нельзя. Все денежные
показатели переведены в цены {pr["base_year"]} года: значение делится на рост цен с базового года. Для каждого
показателя берётся свой индекс цен.

| показатель | индекс цен |
|---|---|
| зарплата, ФОТ, доходы бюджета, розничная торговля | потребительские цены |
| ВМП на душу | дефлятор ВРП |
| инвестиции | цены на продукцию инвестиционного назначения |
| отгрузка | цены производителей (промышленность, обработка) |
| продукция сельского хозяйства | цены сельхозпроизводителей |

Базовый год на типы муниципалитетов не влияет: при общероссийских индексах все значения года умножаются на одно и
то же число, и после нормировки ничего не меняется. Меняются только суммы в таблицах. Индексы по субъектам и
поправка на разницу цен между регионами включаются в боковой панели — они уже меняют результат.
"""
)

# ---------------------------------------------------------------- сводные таблицы — с текущими настройками
# Тот же общий кэш, что у страниц «Кластеры», «Динамика», «Сводные таблицы»: состав и номера кластеров совпадают.
params, pj, mode, method, k = current_partition_params(side)
h = network.config_hash(params)
lab = cluster_partition(pj, method, k, mode)
y0, y1 = side["years"]
year = int(st.session_state.get("_cl_year", y1))
year = year if y0 <= year <= y1 and year in set(lab["year"]) else int(lab["year"].max())
in_window = [int(v) for v in sorted(lab["year"].unique()) if y0 <= v <= y1]  # окно — только показ, модель — вся панель
periods = table2_periods(in_window) if len(in_window) >= 2 else []
mode_txt = "одна модель на все годы" if mode == "pooled" else "каждый год отдельно"
st.subheader("Типы муниципалитетов")
st.caption(
    f"{year} год, {METHOD_NAMES.get(method, method)}, {plural(k, 'тип', 'типа', 'типов')}, {mode_txt}. K1 — тип с самым высоким ВМП на душу. "
    "Описания можно поправить на странице «Сводные таблицы»."
)
res = cluster_table1(pj, method, k, mode, year)
t1 = summary.apply_descriptions(res["table"], f"{h}|{method}|k{k}|{mode}|{year}")
st.dataframe(
    t1[["Кластер", "Число МО", "Характерные признаки", "Примеры МО"]].style.apply(
        lambda col: [f"background-color: {cluster_color(int(v[1:]) - 1)}; color: white; font-weight: 600" for v in col],
        subset=["Кластер"],
    ),
    width="stretch",
    hide_index=True,
    column_config={"Характерные признаки": st.column_config.TextColumn(width="large")},
)

if len(periods) >= 2:
    st.subheader("Типы по субъектам и периодам")
    mt, sc, sp, sm = cluster_table2(pj, method, k, mode, tuple(periods))
    years = [str(y) for y in periods]
    view = sc[["№", "Субъект", "ФО", "МО"]].copy()
    for y in periods:
        view[str(y)] = ["—" if pd.isna(v) else f"K{int(v) + 1} ({sh:.0%})" for v, sh in zip(sc[y], sc[f"{y} доля"])]
    view["траектория"] = sc["траектория"]

    def row_style(r):
        bg = summary.row_css(r["траектория"])
        return [
            f"background-color: {cluster_color(int(r[c].split()[0][1:]) - 1)}; color: white"
            if c in years and r[c] != "—"
            else bg
            for c in r.index
        ]

    st.caption(
        "В ячейке — самый частый тип среди муниципалитетов субъекта и их доля. Рост — переход в тип с более высоким ВМП."
    )
    st.dataframe(view.style.apply(row_style, axis=1), width="stretch", hide_index=True, height=420)

    st.markdown("**Найти муниципалитет**")
    q = st.text_input("Название муниципалитета или субъекта", "", placeholder="например, Хабаровск")
    if q:
        hit = mt[mt["МО"].str.contains(q, case=False, na=False) | mt["Субъект"].str.contains(q, case=False, na=False)]
        hit = hit[["№", "Субъект", "МО", "тип МО", *periods, "траектория"]].head(100).copy()
        for y in periods:
            hit[y] = hit[y].map(summary.label_text)
        hit.columns = [str(c) for c in hit.columns]
        st.caption(f"Найдено: {len(hit)}" + (" (показаны первые 100)" if len(hit) >= 100 else ""))
        st.dataframe(
            hit.style.apply(
                lambda r: [
                    f"background-color: {cluster_color(int(r[c][1:]) - 1)}; color: white"
                    if c in years and r[c] != "—"
                    else summary.row_css(r["траектория"])
                    for c in r.index
                ],
                axis=1,
            ),
            width="stretch",
            hide_index=True,
        )
    title2 = (
        f"Результаты кластеризации по периодам {', '.join(map(str, periods))} ({METHOD_NAMES.get(method, method)}, "
        f"k = {k}, {mode_txt}; K1…K{k} — по убыванию ВМП на душу в ценах {side['prices']['base_year']} г.)"
    )
    c1, c2 = st.columns(2)
    c1.download_button(
        "Скачать в Excel",
        summary.table2_excel(mt, sc, sp, sm, periods, title2),
        "table2.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    c2.download_button(
        "Версия для печати",
        summary.table2_html(sc, sm, periods, title2, "Доля — по числу МО субъекта.").encode("utf-8"),
        "table2_print.html",
        "text/html",
    )


downloads(None, {"боковая_панель": side}, "home")
