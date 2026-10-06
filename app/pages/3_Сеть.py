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
    force_tab,
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


@st.cache_resource(show_spinner="Строю сеть…", max_entries=2)  # сеть всей России — ~150 МБ; общая, без копий
def build(params_json: str, year: int):
    p = json.loads(params_json)
    net = network.build(p, years=[year])[year]
    return net.ids, net.D, net.W, net.A, net.edges(), net.stats(), net.X


reg = registry().set_index("code")
defaults = network.default_params()
feat_pool = [c for c in reg.index if reg.at[c, "block"] in ("economy", "social", "consumption")]
# параметры, построенные на этой странице раньше, сохраняются между страницами и визитами (st.session_state):
# форма открывается с ними, а не с configs/network.yaml
cur = st.session_state.get("net_override") or st.session_state.get("_net_override") or {}
cur_feats = {f: a for f, a in (cur.get("features") or defaults["features"]).items() if f in feat_pool}
cur_geo = cur.get("geo") or defaults["geo"]
cur_sp = cur.get("sparsify") or defaults["sparsify"]
if cur and st.button("Вернуть параметры сети по умолчанию (configs/network.yaml)"):
    st.session_state.pop("net_override", None)
    st.session_state.pop("_net_override", None)
    st.rerun()


def _ix(options: list, value, default: int = 0) -> int:
    return options.index(value) if value in options else default


with st.form("net"):
    st.markdown("**Экономическое расстояние** — признаки и веса a_k")
    feats = st.multiselect(
        "Признаки", feat_pool, default=list(cur_feats), format_func=lambda c: f"{reg.at[c, 'name']} ({c})"
    )
    cols = st.columns(max(len(feats), 1))
    weights = {
        f: cols[i].number_input(f, 0.0, 10.0, float(cur_feats.get(f, 1.0)), 0.5, key=f"w_{f}")
        for i, f in enumerate(feats)
    }
    c1, c2 = st.columns(2)
    metric = c1.selectbox(
        "Метрика",
        ["weighted_l1", "euclidean"],
        index=_ix(["weighted_l1", "euclidean"], cur.get("metric", defaults.get("metric"))),
        format_func={"weighted_l1": "взвешенная сумма |x_i − x_j|", "euclidean": "евклидова"}.get,
    )
    missing = c2.selectbox(
        "МО с пропуском признака",
        ["drop", "pairwise"],
        index=_ix(["drop", "pairwise"], cur.get("missing", defaults.get("missing"))),
        format_func={"drop": "исключить из сети года", "pairwise": "по общим признакам пары"}.get,
    )
    st.markdown("**География** — D = α·D_econ + (1 − α)·D_geo (времени в пути в данных нет — только км)")
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    gw = cur_geo.get("weights", {})
    alpha = g1.slider("α (доля экономики)", 0.0, 1.0, float(cur_geo.get("alpha", 1.0)), 0.05)
    wr = g2.number_input("дороги, км", 0.0, 5.0, float(gw.get("road_km", 1.0)), 0.5)
    wrl = g3.number_input("ж/д, км", 0.0, 5.0, float(gw.get("rail_km", 0.0)), 0.5)
    wl = g4.number_input("по прямой", 0.0, 5.0, float(gw.get("line_km", 0.0)), 0.5)
    wa = g5.number_input("смежность", 0.0, 5.0, float(gw.get("adjacency", 0.0)), 0.5)
    nr = g6.selectbox(
        "МО без дорог",
        ["line", "adjacency", "exclude"],
        index=_ix(["line", "adjacency", "exclude"], cur_geo.get("no_road", "line")),
        format_func={"line": "прямая × извилистость", "adjacency": "только смежность", "exclude": "без географии"}.get,
    )
    st.markdown("**Ребро и прореживание**")
    e1, e2, e3, e4, e5 = st.columns(5)
    ew = e1.selectbox(
        "Расстояние → вес",
        ["gaussian", "inverse"],
        index=_ix(["gaussian", "inverse"], cur.get("edge_weight", defaults.get("edge_weight"))),
        format_func={"gaussian": "exp(−D²/2σ²)", "inverse": "1 / (1 + D)"}.get,
    )
    sm = e2.selectbox(
        "Прореживание",
        ["knn_mst", "knn", "threshold"],
        index=_ix(["knn_mst", "knn", "threshold"], cur_sp.get("method")),
        format_func={"knn_mst": "kNN + остовное дерево", "knn": "kNN", "threshold": "порог веса"}.get,
    )
    k = e3.number_input("k соседей", 1, 30, int(cur_sp.get("k", 7)))
    km = e4.selectbox(
        "kNN",
        ["symmetric", "mutual"],
        index=_ix(["symmetric", "mutual"], cur_sp.get("knn_mode")),
        format_func={"symmetric": "симметризованный", "mutual": "взаимный"}.get,
    )
    thr = e5.number_input("Порог веса", 0.0, 1.0, float(cur_sp.get("threshold", 0.5)), 0.05)
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
st.session_state["_net_override"] = override  # постоянная копия: не стирается при переходе между страницами
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
    force_tab(edges, title=f"Сеть {year} г. (силовая раскладка, seed 42)", key="force_net")
with t3:
    m = mo().set_index("territory_id")
    Dfin = np.where(np.isfinite(D), D, np.nan)
    if len(ids) > 400:
        # большая выборка: в браузер — не n×n (вся Россия — 6 млн ячеек, десятки МБ), а «субъект × субъект»:
        # среднее расстояние между МО двух субъектов
        reg = m.loc[ids, "region_name"].to_numpy()
        names = sorted(set(reg), key=lambda r: (m.loc[m["region_name"].eq(r), "federal_district"].iloc[0], r))
        code = pd.Series(range(len(names)), index=names)[reg].to_numpy()
        M = np.zeros((len(ids), len(names)))
        M[np.arange(len(ids)), code] = 1.0
        ok = np.isfinite(Dfin)
        S = M.T @ np.nan_to_num(Dfin) @ M
        C = M.T @ ok.astype(float) @ M
        Dm = np.where(C > 0, S / np.maximum(C, 1), np.nan)
        title = "Итоговое расстояние D: среднее между МО двух субъектов (субъекты — по федеральным округам)"
        show_labels = True
    else:
        order = np.lexsort((ids, m.loc[ids, "region_name"].values))
        names = [m.at[i, "name_short"] for i in ids[order]]
        Dm = Dfin[np.ix_(order, order)]
        title = "Итоговое расстояние D (МО упорядочены по субъектам)"
        show_labels = len(ids) <= 80
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
    fig.update_xaxes(showticklabels=show_labels, tickfont=dict(size=8))
    fig.update_yaxes(showticklabels=show_labels, tickfont=dict(size=8))
    st.plotly_chart(layout(fig, 760 if len(ids) > 400 else 640, title=title), width="stretch")

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
