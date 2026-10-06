"""Главная страница интерфейса аналитиков. Запуск: streamlit run app/Home.py"""

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
    sample_ids,
    sidebar,
    table2_periods,
)

from src import network, summary

st.set_page_config(page_title="Кластеризация МО", page_icon="🗺️", layout="wide")
side = sidebar()

st.title("Экономическая структура муниципалитетов")
st.markdown(
    "Рабочий интерфейс команды (конкурс СберИндекса, трек «Кластеризация»). Выборка — **все муниципальные "
    "образования России** верхнего уровня в неизменных границах (федеральные округа, в том числе ДФО, — пресеты в "
    "боковой панели). Узел сети — МО; атрибуты — экономические и социальные показатели; рёбра — экономическая "
    "близость МО; сеть строится на каждый год."
)
m = mo()
ids = sample_ids(side)
c1, c2, c3, c4 = st.columns(4)
c1.metric("МО в выборке", f"{len(ids & set(m.loc[m['active'], 'territory_id']))}")
c2.metric("Субъектов", f"{m[m['territory_id'].isin(ids)]['region_code'].nunique()}")
c3.metric("Годы", f"{side['years'][0]}–{side['years'][1]}")
c4.metric("МО со сменой границ", f"{int(m[m['territory_id'].isin(ids)]['boundary_change'].sum())}")

st.subheader("Страницы")
st.markdown(
    """
1. **Данные и качество** — покрытие показателей по годам и субъектам, МО с пропусками, непривязанные строки, источники.
2. **Показатели** — реестр, распределения до и после нормировки, корреляции, картограмма, таблица.
3. **Сеть** — признаки и веса расстояния, география, вес ребра, прореживание; граф на карте и силовая раскладка, статистика.
4. **Кластеры** — метод и число кластеров; карта, граф, паспорта типов, индексы SW / CH / S_Dbw / AVI / AVU / MQ, сравнение методов, названия типов.
5. **Динамика** — переходы МО между типами по годам (диаграмма Санки), матрица переходов, мигранты, ARI, карта «кто куда перешёл».
6. **Конвергенция** — σ по годам, «начальный уровень — рост» с β-регрессией, панельная β, клубы, то же по типам.
7. **Сводные таблицы** — характерные признаки кластеров (автоматическое описание, которое можно исправить вручную); результаты кластеризации по периодам — по МО и по субъектам, траектории, сводка переходов; выгрузка в Excel с цветами и страница для печати.

Настройки выборки и нормировки — в боковой панели, они общие для всех страниц. На каждой странице есть
выгрузка текущей таблицы (CSV) и текущих параметров (YAML): по YAML любой результат воспроизводится.
"""
)
st.info(
    "**ВМП** (валовой муниципальный продукт) — расчётная оценка команды, а не официальная статистика. "
    "Методика — `reports/METHODS.md`; проверки — `reports/GMP_CHECKS.md`."
)

# ---------------------------------------------------------------- цены
st.subheader("Цены: всё в рублях одного года")
pr = side["prices"]
st.markdown(
    f"""
Рубль 2013 г. и рубль 2024 г. — разные единицы: за это время потребительские цены выросли примерно вдвое. Поэтому
денежные показатели пересчитываются **в цены {pr["base_year"]} г.** (базовый год — в боковой панели):
реальное значение = номинальное × уровень цен базового года / уровень цен года наблюдения.

| показатель | индекс цен |
|---|---|
| зарплата, ФОТ, доходы бюджета, розничная торговля | индекс потребительских цен |
| ВМП на душу | дефлятор ВРП |
| инвестиции | индекс цен на продукцию инвестиционного назначения |
| отгрузка; отгрузка обрабатывающих производств | индексы цен производителей (промышленность; обработка) |
| продукция сельского хозяйства | индекс цен производителей сельхозпродукции |

**Почему кластеры не зависят от базового года.** По умолчанию индексы — общероссийские: в каждом году все МО
умножаются на одно и то же число. Смена базового года умножает все реальные значения на общую константу, а
нормировка признаков от этого не меняется — сеть и кластеры остаются теми же (это проверяет тест). Меняются только
подписи и суммы в таблицах. Индексы по субъектам (приближённые) и поправку на межрегиональные различия цен можно
включить в боковой панели — тогда результат меняется. Подробно — `reports/METHODS.md`, раздел 3.1.
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
periods = [y for y in table2_periods(sorted(int(v) for v in lab["year"].unique())) if y in set(lab["year"])]
mode_txt = "одна модель на все годы" if mode == "pooled" else "каждый год отдельно"
st.subheader("Типы муниципалитетов (таблица 1)")
st.caption(
    f"{year} г.; {METHOD_NAMES.get(method, method)}, k = {k}, {mode_txt} — текущие настройки (боковая панель и страницы "
    "«Сеть», «Кластеры»). K1 — самый высокий ВМП на душу в ценах базового года. Правка описаний и выгрузки — страница "
    "«Сводные таблицы»."
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
    st.subheader("Результаты по периодам: субъекты (таблица 2)")
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
        "В ячейке — доминирующий кластер МО субъекта и доля МО субъекта в нём. Траектория: стабильный / рост (переход к "
        "кластеру с более высоким ВМП) / снижение / колебание. Периоды — как на странице «Сводные таблицы»."
    )
    st.dataframe(view.style.apply(row_style, axis=1), width="stretch", hide_index=True, height=420)

    st.markdown("**Поиск МО**")
    q = st.text_input("Название МО или субъекта", "", placeholder="например, Хабаровск")
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
        "⬇ Таблица 2 (Excel, с цветами)",
        summary.table2_excel(mt, sc, sp, sm, periods, title2),
        "table2.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    c2.download_button(
        "⬇ Страница для печати (HTML)",
        summary.table2_html(sc, sm, periods, title2, "Доля — по числу МО субъекта.").encode("utf-8"),
        "table2_print.html",
        "text/html",
    )


downloads(None, {"боковая_панель": side}, "home")
