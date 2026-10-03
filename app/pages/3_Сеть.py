"""Сеть: расстояния между МО и граф выбранного года."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    CATEGORICAL,
    SEQUENTIAL,
    downloads,
    graph_force,
    graph_on_map,
    layout,
    mo,
    network_params,
    registry,
    sidebar,
)

from src import network

st.set_page_config(page_title="Сеть", layout="wide")
side = sidebar()
st.title("Сеть")


@st.cache_data(show_spinner="Строю сеть…", max_entries=16)
def build(params_json: str, year: int):
    p = json.loads(params_json)
    net = network.build(p, years=[year])[year]
    return net.ids, net.D, net.W, net.A, net.edges(), net.stats(), net.X


reg = registry().set_index("code")
defaults = network.default_params()
feat_pool = [
    c for c in reg.index if reg.at[c, "block"] in ("economy", "social", "consumption") and not c.endswith("_cpi")
]

with st.form("net"):
    st.markdown("**Экономическое расстояние** — признаки и веса a_k")
    feats = st.multiselect(
        "Признаки", feat_pool, default=list(defaults["features"]), format_func=lambda c: f"{reg.at[c, 'name']} ({c})"
    )
    cols = st.columns(max(len(feats), 1))
    weights = {
        f: cols[i].number_input(f, 0.0, 10.0, float(defaults["features"].get(f, 1.0)), 0.5, key=f"w_{f}")
        for i, f in enumerate(feats)
    }
    c1, c2 = st.columns(2)
    metric = c1.selectbox(
        "Метрика",
        ["weighted_l1", "euclidean"],
        format_func={"weighted_l1": "взвешенная сумма |x_i − x_j|", "euclidean": "евклидова"}.get,
    )
    missing = c2.selectbox(
        "МО с пропуском признака",
        ["drop", "pairwise"],
        format_func={"drop": "исключить из сети года", "pairwise": "по общим признакам пары"}.get,
    )
    st.markdown("**География** — D = α·D_econ + (1 − α)·D_geo (времени в пути в данных нет — только км)")
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    alpha = g1.slider("α (доля экономики)", 0.0, 1.0, 1.0, 0.05)
    wr = g2.number_input("дороги, км", 0.0, 5.0, 1.0, 0.5)
    wrl = g3.number_input("ж/д, км", 0.0, 5.0, 0.0, 0.5)
    wl = g4.number_input("по прямой", 0.0, 5.0, 0.0, 0.5)
    wa = g5.number_input("смежность", 0.0, 5.0, 0.0, 0.5)
    nr = g6.selectbox(
        "МО без дорог",
        ["line", "adjacency", "exclude"],
        format_func={"line": "прямая × извилистость", "adjacency": "только смежность", "exclude": "без географии"}.get,
    )
    st.markdown("**Ребро и прореживание**")
    e1, e2, e3, e4, e5 = st.columns(5)
    ew = e1.selectbox(
        "Расстояние → вес",
        ["gaussian", "inverse"],
        format_func={"gaussian": "exp(−D²/2σ²)", "inverse": "1 / (1 + D)"}.get,
    )
    sm = e2.selectbox(
        "Прореживание",
        ["knn_mst", "knn", "threshold"],
        format_func={"knn_mst": "kNN + остовное дерево", "knn": "kNN", "threshold": "порог веса"}.get,
    )
    k = e3.number_input("k соседей", 1, 30, 7)
    km = e4.selectbox(
        "kNN", ["symmetric", "mutual"], format_func={"symmetric": "симметризованный", "mutual": "взаимный"}.get
    )
    thr = e5.number_input("Порог веса", 0.0, 1.0, 0.5, 0.05)
    y0, y1 = side["years"]
    year = st.select_slider("Год", list(range(y0, y1 + 1)), value=y1)
    st.form_submit_button("Построить")

if not feats:
    st.stop()
override = {
    "features": weights,
    "metric": metric,
    "missing": missing,
    "geo": {"alpha": alpha, "weights": {"road_km": wr, "rail_km": wrl, "line_km": wl, "adjacency": wa}, "no_road": nr},
    "edge_weight": ew,
    "sparsify": {"method": sm, "k": int(k), "knn_mode": km, "threshold": thr},
}
params = network_params(side, override)
params["features"] = weights
st.session_state["net_override"] = override  # страница «Кластеры» строит кластеры на этой же сети
h = network.config_hash(params)
ids, D, W, A, edges, stats, X = build(json.dumps(params, sort_keys=True, ensure_ascii=False), year)
if len(feats) == 1:
    st.warning(
        "Ребро по одному показателю: кластеризация такой сети сводится к нарезке МО на интервалы значений "
        "этого показателя. Для содержательной сети берите несколько показателей."
    )

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Узлов", stats["nodes"])
s2.metric("Рёбер", stats["edges"])
s3.metric("Компонент связности", stats["components"])
s4.metric("Изолированных узлов", stats["isolated"])
s5.metric("Степень: мин / сред / макс", f"{stats['degree_min']} / {stats['degree_mean']:.1f} / {stats['degree_max']}")
if stats["dropped_missing"]:
    names = mo().set_index("territory_id")["name"]
    st.caption(
        f"Исключено из сети {year} г. из-за пропуска признака: {len(stats['dropped_missing'])} МО — "
        + ", ".join(names.get(t, str(t)) for t in stats["dropped_missing"][:15])
        + ("…" if len(stats["dropped_missing"]) > 15 else "")
    )
st.caption(
    f"Хэш конфигурации: `{h}` — по нему сеть воспроизводится (`data/networks/{h}/params.json` после сохранения)."
)

t1, t2, t3 = st.tabs(["Граф на карте", "Силовая раскладка", "Матрица расстояний"])
with t1:
    st.plotly_chart(graph_on_map(edges, title=f"Сеть {year} г."), width="stretch")
with t2:
    st.plotly_chart(graph_force(edges, title=f"Сеть {year} г. (силовая раскладка, seed 42)"), width="stretch")
with t3:
    m = mo().set_index("territory_id")
    order = np.lexsort((ids, m.loc[ids, "region_name"].values))
    names = [m.at[i, "name_short"] for i in ids[order]]
    Dm = np.where(np.isfinite(D), D, np.nan)[np.ix_(order, order)]
    fig = go.Figure(
        go.Heatmap(
            z=Dm,
            x=names,
            y=names,
            colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
            hovertemplate="%{y} — %{x}<br>D = %{z:.3f}<extra></extra>",
            colorbar=dict(thickness=12),
        )
    )
    fig.update_xaxes(showticklabels=len(ids) <= 80)
    fig.update_yaxes(showticklabels=len(ids) <= 80)
    st.plotly_chart(layout(fig, 640, title="Итоговое расстояние D (МО упорядочены по субъектам)"), width="stretch")

deg = pd.Series(A.sum(axis=1), index=ids)
fd = go.Figure(
    go.Histogram(
        x=deg.values,
        marker_color=CATEGORICAL[0],
        marker_line_width=0,
        hovertemplate="степень %{x}: %{y} МО<extra></extra>",
    )
)
st.plotly_chart(layout(fd, 260, title="Распределение степеней узлов", bargap=0.06), width="stretch")

if st.button("Сохранить сеть за все годы выборки в data/networks/"):
    nets = network.build(params)
    hh = network.save(nets, params)
    st.success(f"Сохранено: data/networks/{hh}/ (edges_{{год}}.parquet, params.json, stats.json)")
m = mo().set_index("territory_id")
edges_out = edges.assign(source_name=edges["source"].map(m["name"]), target_name=edges["target"].map(m["name"]))
downloads(edges_out, {"боковая_панель": side, "год": year, "hash": h, "сеть": params}, "network")
