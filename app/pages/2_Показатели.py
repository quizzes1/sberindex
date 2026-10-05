"""Показатели: реестр, распределения до и после нормировки, корреляции, картограмма, таблица."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import (
    CATEGORICAL,
    DIVERGING,
    choropleth,
    downloads,
    layout,
    mo,
    registry,
    sample_rows,
    sidebar,
    to_view,
    unit_label,
)

from src.normalize import prepare

st.set_page_config(page_title="Показатели", layout="wide")
side = sidebar()
st.title("Показатели")

reg = registry().set_index("code")
rows = sample_rows(side)
rows = to_view(rows, list(rows.columns), side)  # денежные — в ценах из боковой панели
blocks = {
    "economy": "экономика",
    "social": "социальная сфера",
    "consumption": "потребление",
    "quality": "качество данных",
}
c1, c2 = st.columns([1, 3])
blk = c1.multiselect("Блоки", list(blocks), default=["economy", "social"], format_func=blocks.get)
hide = c1.checkbox("Скрыть отраслевые", value=True)
avail = [c for c in reg.index if c in rows and reg.at[c, "block"] in blk]
if hide:
    avail = [c for c in avail if reg.at[c, "parent"] == c]
default = [
    c
    for c in [
        "gmp_pc",
        "wage",
        "invest_pc_nobudget",
        "hhi_emp",
        "density",
        "old_age_share",
        "budget_own_share",
        "retail_pc",
    ]
    if c in avail
]
sel = c2.multiselect("Показатели", avail, default=default, format_func=lambda c: f"{reg.at[c, 'name']} ({c})")
if not sel:
    st.stop()

with st.expander("Описание выбранных показателей", expanded=False):
    t = reg.loc[sel, ["name", "formula", "unit", "source", "block", "log", "direction", "note"]].reset_index()
    t["block"] = t["block"].map(blocks)
    t["unit"] = [unit_label(c, side) for c in sel]
    st.dataframe(t, width="stretch")
if "gmp_pc" in sel:
    st.caption(
        "ВМП — расчётная оценка команды (не официальная статистика). Для 2013–2016 гг. — базовый метод "
        "(колонка gmp_method_used), с 2017 г. — отраслевой."
    )

st.subheader("Распределения: исходные значения и после нормировки")
logc = [c for c in sel if bool(reg.at[c, "log"])]
X = rows.set_index(["territory_id", "year"])[sel]
Z = prepare(X, sel, log_columns=logc, method=side["method"], scope=side["scope"])
ind = st.selectbox("Показатель", sel, format_func=lambda c: reg.at[c, "name"])
a, b = st.columns(2)
fa = go.Figure(
    go.Histogram(
        x=X[ind].dropna(),
        nbinsx=40,
        marker_color=CATEGORICAL[0],
        marker_line_width=0,
        hovertemplate="%{x}: %{y} МО-лет<extra></extra>",
    )
)
a.plotly_chart(layout(fa, 300, title=f"Исходные, {unit_label(ind, side)}", bargap=0.06), width="stretch")
fb = go.Figure(
    go.Histogram(
        x=Z[ind].dropna(),
        nbinsx=40,
        marker_color=CATEGORICAL[0],
        marker_line_width=0,
        hovertemplate="%{x:.2f}: %{y} МО-лет<extra></extra>",
    )
)
b.plotly_chart(
    layout(
        fb, 300, title=f"После {'логарифма и ' if ind in logc else ''}{side['method']} ({side['scope']})", bargap=0.06
    ),
    width="stretch",
)

st.subheader("Корреляции (ρ Спирмена) и разброс")
corr = X.corr(method="spearman")
lab = [reg.at[c, "name"][:40] for c in sel]
fc = go.Figure(
    go.Heatmap(
        z=corr.values,
        x=lab,
        y=lab,
        zmin=-1,
        zmax=1,
        colorscale=DIVERGING,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}<br>ρ = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=12),
    )
)
st.plotly_chart(layout(fc, 60 + 40 * len(sel)), width="stretch")
strong = [
    (sel[i], sel[j], corr.iat[i, j])
    for i in range(len(sel))
    for j in range(i + 1, len(sel))
    if abs(corr.iat[i, j]) >= 0.8
]
if strong:
    st.warning(
        "Сильно связанные пары (|ρ| ≥ 0,8) — кандидаты на исключение дублей: "
        + "; ".join(f"{reg.at[x, 'name']} — {reg.at[y, 'name']} ({r:.2f})" for x, y, r in strong)
    )
var = (
    Z.var()
    .rename("дисперсия после нормировки")
    .to_frame()
    .join((X.std() / X.mean().abs()).rename("коэф. вариации"))
    .join(X.notna().mean().mul(100).rename("покрытие, %"))
)
var.index = [reg.at[c, "name"] for c in var.index]
st.dataframe(var.style.format("{:.3f}"), width="stretch")

st.subheader("Картограмма")
y0, y1 = side["years"]
yr = st.select_slider("Год", list(range(y0, y1 + 1)), value=y1)
v = rows[rows["year"].eq(yr)].set_index("territory_id")[ind]
st.plotly_chart(
    choropleth(v, f"{reg.at[ind, 'name']}, {yr} ({unit_label(ind, side)})", log=bool(reg.at[ind, "log"])),
    width="stretch",
)

st.subheader("Таблица")
m = mo().set_index("territory_id")
tab = rows[rows["year"].eq(yr)][["territory_id", "year", *sel]].assign(
    МО=lambda t: t["territory_id"].map(m["name"]), регион=lambda t: t["territory_id"].map(m["region_name"])
)
tab = tab[["territory_id", "МО", "регион", "year", *sel]]
st.dataframe(tab, width="stretch", height=360)
downloads(tab, {"боковая_панель": side, "показатели": sel, "год": yr}, "indicators")
