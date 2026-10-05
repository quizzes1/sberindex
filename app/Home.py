"""Главная страница интерфейса аналитиков. Запуск: streamlit run app/Home.py"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import cluster_color, downloads, mo, sample_ids, sidebar  # noqa: E402

from src import summary
from src.io import PROCESSED, load_yaml

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

# ---------------------------------------------------------------- сводные таблицы (разбиение по умолчанию)
part = load_yaml("dynamics.yaml")["partition"]
t1p, t2p = PROCESSED / "summary_table1.csv", PROCESSED / "summary_table2_subjects_count.csv"
if t1p.exists() and t2p.exists():
    st.subheader("Типы муниципалитетов (таблица 1)")
    st.caption(
        f"Разбиение по умолчанию: вся Россия, «Уорд + k-means», k = {part['k']}, одна модель на все годы; K1 — самый "
        "высокий ВМП на душу в ценах базового года. Свои настройки, правка описаний и выгрузки — страница «Сводные таблицы»."
    )
    t1 = pd.read_csv(t1p, encoding="utf-8-sig")
    st.dataframe(
        t1[["Кластер", "Число МО", "Характерные признаки", "Примеры МО"]].style.apply(
            lambda col: [
                f"background-color: {cluster_color(int(v[1:]) - 1)}; color: white; font-weight: 600" for v in col
            ],
            subset=["Кластер"],
        ),
        width="stretch",
        hide_index=True,
        column_config={"Характерные признаки": st.column_config.TextColumn(width="large")},
    )

    st.subheader("Результаты по периодам: субъекты (таблица 2)")
    t2 = pd.read_csv(t2p)
    years = [c for c in t2.columns if c.isdigit()]
    tcol = summary.cfg()["periods"]["trajectory_colors"]
    view = t2[["№", "Субъект", "ФО", "МО"]].copy()
    for y in years:
        view[y] = ["—" if pd.isna(v) else f"K{int(v) + 1} ({sh:.0%})" for v, sh in zip(t2[y], t2[f"{y} доля"])]
    view["траектория"] = t2["траектория"]

    def row_style(r):
        bg = f"background-color: #{summary.tint(tcol.get(r['траектория'], '#ffffff'))}"
        return [
            f"background-color: {cluster_color(int(r[c].split()[0][1:]) - 1)}; color: white"
            if c in years and r[c] != "—"
            else bg
            for c in r.index
        ]

    st.caption(
        "В ячейке — доминирующий кластер МО субъекта и доля МО субъекта в нём. Траектория: стабильный / рост (переход к "
        "кластеру с более высоким ВМП) / снижение / колебание."
    )
    st.dataframe(view.style.apply(row_style, axis=1), width="stretch", hide_index=True, height=420)

    st.markdown("**Поиск МО**")
    q = st.text_input("Название МО или субъекта", "", placeholder="например, Хабаровск")
    if q:
        mt = pd.read_csv(PROCESSED / "summary_table2_mo.csv", encoding="utf-8-sig")
        hit = mt[mt["МО"].str.contains(q, case=False, na=False) | mt["Субъект"].str.contains(q, case=False, na=False)]
        ycols = [c for c in mt.columns if c.isdigit()]
        st.caption(f"Найдено: {len(hit)}" + (" (показаны первые 100)" if len(hit) > 100 else ""))
        st.dataframe(
            hit.head(100).style.apply(
                lambda r: [
                    f"background-color: {cluster_color(int(r[c][1:]) - 1)}; color: white"
                    if c in ycols and isinstance(r[c], str) and r[c].startswith("K")
                    else f"background-color: #{summary.tint(tcol.get(r['траектория'], '#ffffff'))}"
                    for c in r.index
                ],
                axis=1,
            ),
            width="stretch",
            hide_index=True,
        )
    c1, c2 = st.columns(2)
    if (PROCESSED / "summary_table2.xlsx").exists():
        c1.download_button(
            "⬇ Таблица 2 (Excel, с цветами)",
            (PROCESSED / "summary_table2.xlsx").read_bytes(),
            "table2.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if (PROCESSED / "summary_table2_print.html").exists():
        c2.download_button(
            "⬇ Страница для печати (HTML)",
            (PROCESSED / "summary_table2_print.html").read_bytes(),
            "table2_print.html",
            "text/html",
        )

downloads(None, {"боковая_панель": side}, "home")
