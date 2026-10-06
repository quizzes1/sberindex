"""Сводные таблицы: 1 — характерные признаки кластеров (автотекст с ручной правкой), 2 — кластеры по периодам."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    cluster_color,
    cluster_controls,
    cluster_partition,
    cluster_table1,
    cluster_table2,
    cluster_year,
    clustering_cfg,
    downloads,
    layout,
    network_params,
    sidebar,
    table2_periods,
)

from src import clustering, network, summary
from src.io import load_yaml

st.set_page_config(page_title="Сводные таблицы", layout="wide")
side = sidebar()
st.title("Сводные таблицы")

cc = clustering_cfg()
part = load_yaml("dynamics.yaml")["partition"]
override = st.session_state.get("net_override", {})
params = network_params(side, override)
if "features" in override:
    params["features"] = override["features"]
pj = json.dumps(params, sort_keys=True, ensure_ascii=False)
h = network.config_hash(params)
METHOD_LABELS = {"ward_kmeans": "Уорд + k-means", "kmeans": "k-средних", "ward": "Уорд", "gmm": "гауссовы смеси"}

c0, c1, c2, c3 = st.columns([1.2, 1.2, 1, 1])
mode, method, k = cluster_controls(c0, c1, c2, exclude=("canus",))
year = cluster_year(c3, side, "Год таблицы 1")
pr = side["prices"]
st.caption(
    f"Сеть `{h}`; кластеры пронумерованы K1…Kn по убыванию медианы ВМП на душу в ценах {pr['base_year']} г. "
    "(K1 — самый высокий). Режим «одна модель на все годы» — по умолчанию: номер кластера одинаков во всех годах."
)


labels_of = cluster_partition  # общий кэш: то же разбиение, что на страницах «Кластеры» и «Динамика»
table1 = cluster_table1


lab = labels_of(pj, method, k, mode)
if year not in set(lab["year"]):
    st.warning(f"В {year} г. нет МО с полным набором признаков сети.")
    st.stop()
res = table1(pj, method, k, mode, year)
key = f"{h}|{method}|k{k}|{mode}|{year}"
t = summary.apply_descriptions(res["table"], key)

# ---------------------------------------------------------------- таблица 1
st.header("Таблица 1. Характерные признаки кластеров")
th = summary.cfg()["characteristic"]["thresholds"]
st.caption(
    f"{year} г. Средний z-score признака по МО кластера (z — по всем МО выборки за год, денежные — в ценах "
    f"{pr['base_year']} г., логарифм — по реестру). «Максимальные» — наибольшее среднее среди кластеров и z ≥ "
    f"{th['extreme']}; «минимальные» — наименьшее и z ≤ −{th['extreme']}; «высокие» — z ≥ {th['high']}; «низкие» — "
    f"z ≤ −{th['high']}; «близкие к среднему» — |z| < {th['high']}. Пороги — configs/summary.yaml."
)
changed = t[t["правка"].eq("состав изменился")]
if len(changed):
    st.warning(
        "Состав кластеров "
        + ", ".join(changed["Кластер"])
        + " изменился после ручной правки текста (пересчёт, другая сеть или данные): проверьте описание — "
        "сохранённый текст показан, но писался для другого состава."
    )
show = t[["Кластер", "Число МО", "Характерные признаки", "Примеры МО", "правка"]]
st.dataframe(
    show.style.apply(
        lambda col: [f"background-color: {cluster_color(int(v[1:]) - 1)}; color: white; font-weight: 600" for v in col],
        subset=["Кластер"],
    ),
    width="stretch",
    hide_index=True,
    column_config={
        "Характерные признаки": st.column_config.TextColumn(width="large"),
        "правка": st.column_config.TextColumn("правка вручную", width="small"),
    },
)

with st.expander("Исправить тексты (сохраняются в data/cluster_descriptions.yaml и не затираются при пересчёте)"):
    with st.form("desc"):
        new = {}
        for _, r in t.iterrows():
            new[r["Кластер"]] = st.text_area(
                f"{r['Кластер']} ({r['Число МО']} МО) · авто: {r['авто']}",
                value=r["Характерные признаки"],
                key=f"d_{key}_{r['Кластер']}",
                height=80,
            )
        if st.form_submit_button("Сохранить"):
            n = 0
            for _, r in t.iterrows():
                txt = new[r["Кластер"]].strip()
                if txt != r["Характерные признаки"] or (r["правка"] == "состав изменился"):
                    # пустой текст или текст, равный автоматическому, — правка снимается
                    summary.save_description(
                        key, r["Кластер"], "" if txt in ("", r["авто"]) else txt, r["состав"], r["авто"]
                    )
                    n += 1
            st.success(f"Сохранено: {n}.")
            st.cache_data.clear()
            st.rerun()

st.subheader("Развёрнутый вариант: средние значения признаков по кластерам")
raw, zz = res["raw"].copy(), res["z"].copy()
raw.index = zz.index = [clustering.code(i) for i in raw.index]
raw.columns = zz.columns = [res["names"][f] for f in res["features"]]
st.dataframe(
    raw.style.apply(lambda col: [f"background-color: #{summary.z_fill(zz.at[i, col.name])}" for i in col.index]).format(
        "{:,.3f}"
    ),
    width="stretch",
)
st.caption(
    "Средние в исходных единицах (денежные — в ценах базового года); цвет — средний z-score: синий — ниже среднего по "
    "МО выборки, красный — выше. Пропуски не заполняются: среднее — по МО, где показатель есть."
)

title = (
    f"Характерные признаки кластеров, {year} г. ({METHOD_LABELS.get(method, method)}, k = {k}, "
    f"{'pooled' if mode == 'pooled' else 'по годам'}; денежные — в ценах {pr['base_year']} г.)"
)
st.download_button(
    "⬇ Таблица 1 (Excel, с цветами)",
    summary.table1_excel(res, t, title),
    f"table1_{year}.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
downloads(
    t[["Кластер", "Число МО", "Характерные признаки", "Примеры МО"]],
    {"боковая_панель": side, "режим": mode, "метод": method, "k": k, "год": year, "hash_сети": h},
    "table1",
)

# ---------------------------------------------------------------- таблица 2
st.divider()
st.header("Таблица 2. Результаты кластеризации по периодам")
pc = summary.cfg()["periods"]
tcol = pc["trajectory_colors"]
years_all = sorted(int(y) for y in lab["year"].unique())
periods = table2_periods(years_all, st)
if len(periods) < 2:
    st.info("Выберите хотя бы два года.")
    st.stop()
st.caption(
    "В ячейке — кластер МО в этом году (K1 — самый высокий ВМП на душу в ценах базового года; «—» — МО нет в сети года: "
    "не хватает признаков). Траектория: **стабильный** — один кластер во всех периодах; **рост** — номер только "
    "уменьшается (переход к кластеру с более высоким ВМП); **снижение** — только растёт; **колебание** — и то и другое."
    + (
        ""
        if mode == "pooled"
        else " ⚠️ Режим «каждый год отдельно»: номера K упорядочены по ВМП внутри каждого года, но состав кластеров "
        "разных лет определён разными моделями — переходы отчасти отражают перекластеризацию."
    )
)


table2 = cluster_table2


mt, sc, sp, sm = table2(pj, method, k, mode, tuple(periods))


def row_style_str(r, lc):
    """Ячейки меток — цвет кластера; остальные ячейки — фон по траектории (summary.row_css: читается в любой теме)."""
    bg = summary.row_css(r["траектория"])
    return [
        f"background-color: {cluster_color(int(r[c][1:]) - 1)}; color: white; font-weight: 600"
        if c in lc and r[c] != "—"
        else bg
        for c in r.index
    ]


def style_labels(df: pd.DataFrame, label_cols: list):
    view = df.copy()
    for y in label_cols:
        view[y] = view[y].map(summary.label_text)
    view.columns = [str(c) for c in view.columns]
    lc = [str(c) for c in label_cols]
    return view.style.apply(lambda r: row_style_str(r, lc), axis=1)


tab_mo, tab_subj, tab_sum = st.tabs(["Уровень МО", "Уровень субъектов (для отчёта)", "Сводка переходов"])

with tab_mo:
    f1, f2, f3 = st.columns([2, 1, 1])
    q = f1.text_input("Поиск по названию МО или субъекта", "")
    f_fd = f2.multiselect("Федеральный округ", sorted(mt["ФО"].dropna().unique()))
    f_type = f3.multiselect("Тип МО", sorted(mt["тип МО"].dropna().unique()))
    f4, f5, f6 = st.columns([2, 1, 1])
    f_reg = f4.multiselect("Субъект", sorted(mt["Субъект"].dropna().unique()))
    f_cl = f5.multiselect("Кластер (хотя бы в одном периоде)", [clustering.code(i) for i in range(k)])
    f_tr = f6.multiselect("Траектория", summary.TRAJECTORIES)
    v = mt
    if q:
        v = v[v["МО"].str.contains(q, case=False, na=False) | v["Субъект"].str.contains(q, case=False, na=False)]
    if f_fd:
        v = v[v["ФО"].isin(f_fd)]
    if f_type:
        v = v[v["тип МО"].isin(f_type)]
    if f_reg:
        v = v[v["Субъект"].isin(f_reg)]
    if f_cl:
        want = {int(c[1:]) - 1 for c in f_cl}
        v = v[v[periods].isin(want).any(axis=1)]
    if f_tr:
        v = v[v["траектория"].isin(f_tr)]
    s1, s2, s3 = st.columns([2, 1, 1])
    sort_by = s1.selectbox(
        "Сортировка",
        ["№", "Субъект", "МО", "тип МО", "траектория", *periods],
        format_func=lambda c: f"кластер {c} г." if isinstance(c, int) else c,
    )
    desc = s2.checkbox("по убыванию", value=False)
    v = v.sort_values(sort_by, ascending=not desc, kind="stable", na_position="last")
    size = int(pc["page_size"])
    pages = max(1, -(-len(v) // size))
    page = s3.number_input(f"Страница (из {pages})", 1, pages, 1)
    st.caption(
        f"Найдено МО: {len(v)} из {len(mt)}. Цвет строки — траектория: рост — зелёный, снижение — красный, "
        "колебание — жёлтый; стабильные и без данных — без цвета."
    )
    cols = ["№", "Субъект", "МО", "тип МО", *periods, "траектория"]
    part_v = v[cols].iloc[(page - 1) * size : page * size]
    if len(part_v):
        st.dataframe(
            style_labels(part_v, periods), width="stretch", hide_index=True, height=min(38 * (len(part_v) + 1), 1900)
        )
    csv_mo = v[cols].copy()
    for y in periods:
        csv_mo[y] = csv_mo[y].map(summary.label_text)
    st.download_button(
        "⬇ Таблица МО с учётом фильтров (CSV)",
        csv_mo.to_csv(index=False).encode("utf-8-sig"),
        "table2_mo.csv",
        "text/csv",
    )

with tab_subj:
    by = st.radio(
        "Доля в доминирующем кластере",
        ["count", "pop"],
        horizontal=True,
        format_func={"count": "по числу МО", "pop": "по населению"}.get,
    )
    sv = sc if by == "count" else sp
    st.caption(
        "Строка — субъект; в ячейке — доминирующий кластер МО субъекта в этом году и доля "
        + ("МО субъекта в нём." if by == "count" else "населения субъекта (по МО в сети) в МО этого кластера.")
        + " Траектория — по доминирующему кластеру. «МО» — число МО субъекта в сети последнего периода."
    )
    show_s = sv.copy()
    lc = [str(y) for y in periods]
    for y in periods:
        show_s[f"{y} доля"] = show_s[f"{y} доля"].map(lambda x: "" if pd.isna(x) else f"{x:.0%}")
        show_s[y] = show_s[y].map(summary.label_text)
    show_s.columns = [str(c) for c in show_s.columns]
    st.dataframe(
        show_s.style.apply(lambda r: row_style_str(r, lc), axis=1), width="stretch", hide_index=True, height=600
    )

with tab_sum:
    st.caption("Число и доля МО по траекториям — по всей выборке и по федеральным округам.")
    sm_show = sm.copy()
    st.dataframe(
        sm_show.style.format({c: "{:.1f}" for c in sm_show.columns if str(c).endswith("%")}),
        width="stretch",
        hide_index=True,
    )
    fig = go.Figure()
    for tr in summary.TRAJECTORIES[:-1]:
        fig.add_trace(
            go.Bar(
                y=sm["территория"],
                x=sm[f"{tr}, %"],
                name=tr,
                orientation="h",
                marker_color=tcol[tr] if tr != "стабильный" else "#9a9994",
                marker_line_width=0,
                hovertemplate="%{y}: %{x:.1f}%<extra>" + tr + "</extra>",
            )
        )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(layout(fig, 360, barmode="stack", title="Траектории МО, % (без «нет данных»)"), width="stretch")

title2 = (
    f"Результаты кластеризации по периодам {', '.join(map(str, periods))} ({METHOD_LABELS.get(method, method)}, k = {k}, "
    f"{'pooled' if mode == 'pooled' else 'по годам'}; кластеры K1…K{k} — по убыванию ВМП на душу в ценах {pr['base_year']} г.)"
)
d1, d2 = st.columns(2)
d1.download_button(
    "⬇ Таблица 2 (Excel, с цветами: МО, субъекты, переходы)",
    summary.table2_excel(mt, sc, sp, sm, periods, title2),
    "table2.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
d2.download_button(
    "⬇ Страница для печати (HTML: субъекты и переходы)",
    summary.table2_html(
        sv,
        sm,
        periods,
        title2,
        ("Доля — по числу МО субъекта." if by == "count" else "Доля — по населению субъекта.")
        + " Откройте в браузере и напечатайте или сохраните в PDF.",
    ).encode("utf-8"),
    "table2_print.html",
    "text/html",
)
