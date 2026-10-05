"""Динамика: переходы МО между типами по годам."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    CATEGORICAL,
    COARSE_FROM,
    OTHER_GRAY,
    cluster_color,
    clustering_cfg,
    downloads,
    features_subset,
    geo_layout,
    layout,
    mo,
    network_params,
    sidebar,
)

from src import clustering, dynamics, network
from src.io import load_yaml

st.set_page_config(page_title="Динамика", layout="wide")
side = sidebar()
st.title("Динамика типов")

cc = clustering_cfg()
dc = load_yaml("dynamics.yaml")
METHOD_LABELS = {"ward": "Уорд", "ward_kmeans": "Уорд + k-means", "kmeans": "k-средних", "gmm": "гауссовы смеси"}
override = st.session_state.get("net_override", {})
params = network_params(side, override)
if "features" in override:
    params["features"] = override["features"]
pj = json.dumps(params, sort_keys=True, ensure_ascii=False)

c1, c2, c3, c4 = st.columns(4)
mode = c1.selectbox(
    "Режим",
    ["pooled", "per_year"],
    format_func={"pooled": "одна модель на всю панель (по умолчанию)", "per_year": "каждый год заново"}.get,
)
pool_methods = [m for m in cc["methods"] if clustering.FAMILIES[m] == "attributes"]
all_methods = [m for m in cc["methods"] if clustering.available(m)[0] and m != "canus"]
methods = pool_methods if mode == "pooled" else all_methods
default_m = dc["partition"]["method"] if mode == "pooled" else ("kefrin" if "kefrin" in methods else methods[0])
method = c2.selectbox(
    "Метод",
    methods,
    index=methods.index(default_m) if default_m in methods else 0,
    format_func=lambda m: METHOD_LABELS.get(m, m),
)
k = c3.slider("k", 2, 10, dc["partition"]["k"])
thr = c4.slider("Порог Жаккара (режим «каждый год заново»)", 0.0, 0.9, float(dc["matching"]["min_jaccard"]), 0.05)
st.caption(
    "Режим «одна модель на всю панель» — решение команды: типы общие для всех лет, и смена типа означает "
    "изменение показателей МО, а не перекластеризацию года (при «каждый год заново» заметная часть переходов — шум)."
)


@st.cache_data(show_spinner="Кластеризация по годам…", max_entries=32)
def partition(pj: str, mode: str, method: str, k: int) -> pd.DataFrame:
    p = json.loads(pj)
    mp = {**cc["methods"][method], "method": method, "k": k, "seed": cc["seed"]}
    if mode == "pooled":
        X = network.features(p).dropna()
        return clustering.fit_pooled(X, mp).reset_index()
    out = []
    for y, net in network.build(p).items():
        lab = clustering.fit(net.X.to_numpy(), net.W * net.A, mp)
        out.append(pd.DataFrame({"territory_id": net.ids, "year": y, "label": lab}))
    return pd.concat(out, ignore_index=True)


lab = partition(pj, mode, method, k)
# сквозные типы K1…Kn: 0 → K1 — самый высокий ВМП на душу в ценах базового года
th = dynamics.through_labels(lab, mode, thr, params["prices"])
st.session_state["types_last"] = th[th["year"].eq(th["year"].max())].set_index("territory_id")["through"].to_dict()
summ = dynamics.year_summary(th)
tr = dynamics.transitions(th)
mig = dynamics.migrants(th)
years = sorted(th["year"].unique())

st.subheader("Переходы между типами (диаграмма Санки)")
nodes = [(y, t) for y in years for t in sorted(th.loc[th["year"].eq(y), "through"].unique())]
idx = {n: i for i, n in enumerate(nodes)}
cnt = th.groupby(["year", "through"]).size()
link_color = [cluster_color(int(r["from"])).replace("#", "") for _, r in tr.iterrows()]
fig = go.Figure(
    go.Sankey(
        arrangement="snap",
        node=dict(
            label=[clustering.code(t) for _, t in nodes],
            color=[cluster_color(int(t)) for _, t in nodes],
            pad=8,
            thickness=14,
            customdata=[f"{y}: {clustering.code(t)}, {cnt[(y, t)]} МО" for y, t in nodes],
            hovertemplate="%{customdata}<extra></extra>",
            x=[(years.index(y)) / max(len(years) - 1, 1) * 0.98 + 0.01 for y, _ in nodes],
        ),
        link=dict(
            source=[idx[(r.year_from, r["from"])] for _, r in tr.iterrows()],
            target=[idx[(r.year_to, r["to"])] for _, r in tr.iterrows()],
            value=tr["n"].tolist(),
            color=[f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},0.35)" for c in link_color],
            hovertemplate="%{source.label} → %{target.label}: %{value} МО<extra></extra>",
        ),
    )
)
fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=30, b=30),
    annotations=[
        dict(x=i / max(len(years) - 1, 1), y=-0.06, text=str(y), showarrow=False, xref="paper", yref="paper")
        for i, y in enumerate(years)
    ],
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Колонки — годы, узлы — сквозные типы K1…Kn (K1 — самый высокий ВМП на душу в ценах базового года), ленты — МО, "
    "перешедшие из типа в тип. В режиме pooled тип одинаково определён во всех годах; в режиме «каждый год заново» "
    "кластеры соседних лет сопоставляются венгерским алгоритмом по мере Жаккара."
)

st.subheader("Устойчивость по годам")
c1, c2 = st.columns([2, 3])
c1.dataframe(summ.style.format({"ARI": "{:.2f}", "доля сменивших тип": "{:.0%}"}), width="stretch")
f = go.Figure(
    go.Scatter(
        x=[f"{a}→{b}" for a, b in zip(summ["year_from"], summ["year_to"])],
        y=summ["доля сменивших тип"],
        mode="lines+markers",
        line=dict(width=2, color=CATEGORICAL[0]),
        marker=dict(size=8),
        hovertemplate="%{x}: %{y:.0%}<extra></extra>",
    )
)
c2.plotly_chart(layout(f, 260, title="Доля МО, сменивших тип", yaxis_tickformat=".0%"), width="stretch")

st.subheader("Матрица переходов и «мигранты»")
a, b = st.columns(2)
y_from = a.selectbox("Из года", years[:-1], index=len(years) - 2)
y_to = b.selectbox("В год", [y for y in years if y > y_from], index=0)
s = th.set_index(["year", "territory_id"])["through"]
common = s.loc[y_from].index.intersection(s.loc[y_to].index)
mat = pd.crosstab(
    s.loc[y_from].loc[common].map(clustering.code).rename(f"кластер в {y_from}"),
    s.loc[y_to].loc[common].map(clustering.code).rename(f"кластер в {y_to}"),
)
st.dataframe(mat, width="stretch")
m = mo().set_index("territory_id")
moved = pd.DataFrame({"было": s.loc[y_from].loc[common], "стало": s.loc[y_to].loc[common]})
moved = moved[moved["было"].ne(moved["стало"])]
moved = moved.assign(МО=moved.index.map(m["name"]), регион=moved.index.map(m["region_name"]))
st.markdown(f"Сменили тип с {y_from} по {y_to}: **{len(moved)}** МО из {len(common)}.")

st.subheader(f"Карта «кто куда перешёл», {y_from} → {y_to}")
fig = go.Figure()
stay = [int(i) for i in common if i not in moved.index]
if stay:
    fig.add_trace(
        go.Choropleth(
            geojson=features_subset(set(stay), coarse=len(common) > COARSE_FROM),
            featureidkey="properties.territory_id",
            locations=stay,
            z=[1] * len(stay),
            colorscale=[[0, OTHER_GRAY], [1, OTHER_GRAY]],
            showscale=False,
            marker_line_width=0.3,
            marker_line_color="white",
            name=f"не сменили тип ({len(stay)})",
            showlegend=True,
            text=[m.at[i, "name"] for i in stay],
            hovertemplate="%{text}<br>тип не изменился<extra></extra>",
            marker_opacity=0.35,
        )
    )
for t in sorted(moved["стало"].unique()):
    sel = moved[moved["стало"].eq(t)]
    fig.add_trace(
        go.Choropleth(
            geojson=features_subset(set(int(i) for i in sel.index), coarse=len(common) > COARSE_FROM),
            featureidkey="properties.territory_id",
            locations=list(sel.index),
            z=[1] * len(sel),
            showscale=False,
            colorscale=[[0, cluster_color(int(t))], [1, cluster_color(int(t))]],
            marker_line_width=0.3,
            marker_line_color="white",
            name=f"перешли в {clustering.code(t)} ({len(sel)})",
            showlegend=True,
            text=[
                f"{m.at[i, 'name']}<br>{clustering.code(sel.at[i, 'было'])} → {clustering.code(t)}" for i in sel.index
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )
fig.update_layout(legend=dict(orientation="h", y=-0.02))
st.plotly_chart(geo_layout(fig, set(int(i) for i in common)), width="stretch")
st.dataframe(moved[["МО", "регион", "было", "стало"]], width="stretch", height=300)

st.caption(
    f"Всего переходов между соседними годами: {len(mig)}; МО, хотя бы раз сменивших тип: "
    f"{mig['territory_id'].nunique()} из {th['territory_id'].nunique()}."
)
downloads(
    th.assign(МО=th["territory_id"].map(m["name"])),
    {"боковая_панель": side, "режим": mode, "метод": method, "k": k, "порог_Жаккара": thr, "сеть": params},
    "dynamics",
)
