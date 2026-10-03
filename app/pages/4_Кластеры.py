"""Кластеры: метод, k, год; карта, граф, паспорта типов, индексы качества, сравнение методов, названия."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    CATEGORICAL,
    cluster_color,
    cluster_map,
    clustering_cfg,
    downloads,
    graph_force,
    graph_on_map,
    indicators_wide,
    layout,
    load_names,
    mo,
    network_params,
    registry,
    save_names,
    sidebar,
)

from src import clustering, icvi, network

st.set_page_config(page_title="Кластеры", layout="wide")
side = sidebar()
st.title("Кластеры")

cc = clustering_cfg()
METHOD_NAMES = {
    "kmeans": "k-средних",
    "ward": "иерархическая Уорда",
    "gmm": "гауссовы смеси",
    "leiden": "Leiden (граф)",
    "spectral": "спектральная (граф)",
    "kefrin": "KEFRiN (атрибутированная сеть)",
    "canus": "CANUS (атрибутированная сеть, медленный)",
}
avail = {m: clustering.available(m) for m in cc["methods"]}


@st.cache_data(show_spinner="Строю сеть…", max_entries=16)
def net_year(params_json: str, year: int):
    net = network.build(json.loads(params_json), years=[year])[year]
    return net.ids, net.X, net.W * net.A, net.edges()


@st.cache_data(show_spinner="Кластеризация…", max_entries=256)
def run(params_json: str, year: int, method: str, k: int):
    ids, X, Wsp, _ = net_year(params_json, year)
    mp = {**cc["methods"][method], "method": method, "k": k, "seed": cc["seed"]}
    lab = clustering.fit(X.to_numpy(), Wsp, mp)
    return pd.Series(lab, index=ids), icvi.compute_all(X.to_numpy(), Wsp, lab)


override = st.session_state.get("net_override", {})
params = network_params(side, override)
if "features" in override:
    params["features"] = override["features"]
pj = json.dumps(params, sort_keys=True, ensure_ascii=False)
h = network.config_hash(params)
st.caption(
    f"Сеть: признаки {list(params['features'])}, α = {params['geo']['alpha']}, хэш `{h}` "
    f"({'настроена на странице «Сеть»' if override else 'параметры по умолчанию — configs/network.yaml'})."
)

c1, c2, c3 = st.columns(3)
ok_methods = [m for m in cc["methods"] if avail[m][0]]
method = c1.selectbox(
    "Метод", ok_methods, index=ok_methods.index("kefrin") if "kefrin" in ok_methods else 0, format_func=METHOD_NAMES.get
)
k = c2.slider("Число кластеров k", 2, 10, 6)
y0, y1 = side["years"]
year = c3.select_slider("Год", list(range(y0, y1 + 1)), value=y1)
missing = [f"{m}: {w}" for m, (ok, w) in avail.items() if not ok]
if missing:
    st.caption("Недоступны: " + "; ".join(missing))
if k > len(CATEGORICAL):
    st.caption(
        f"Кластеры с 9-го окрашены нейтральным серым: различимых цветов не больше {len(CATEGORICAL)} — "
        "ориентируйтесь на подсказки и таблицу."
    )

labels, scores = run(pj, year, method, k)
ids, X, Wsp, edges = net_year(pj, year)
nkey = f"{h}|{method}|k{k}|{year}"
names = load_names().get(nkey, {})
disp = {int(c): names.get(int(c)) or f"Тип {c}" for c in sorted(labels.unique())}

t1, t2, t3 = st.tabs(["Карта", "Граф на карте", "Силовая раскладка"])
with t1:
    st.plotly_chart(cluster_map(labels, disp, f"{METHOD_NAMES[method]}, k = {k}, {year} г."), width="stretch")
with t2:
    st.plotly_chart(graph_on_map(edges, labels, f"Сеть {year} г., цвет — кластер"), width="stretch")
with t3:
    st.plotly_chart(graph_force(edges, labels, title=f"Сеть {year} г., цвет — кластер"), width="stretch")

st.subheader("Индексы качества")
order = ["K", "SW", "CH", "DBI", "S_Dbw", "AVI", "AVU", "ANUI", "MQ"]
arrows = {k_: ("↑" if v > 0 else "↓") for k_, v in icvi.BETTER.items()}
st.dataframe(
    pd.DataFrame([{f"{c} {arrows.get(c, '')}".strip(): scores[c] for c in order}]).style.format("{:.3f}"),
    width="stretch",
)
st.caption(
    "↑ — больше лучше, ↓ — меньше лучше. SW, CH, DBI, S_Dbw — по нормированным признакам; AVI, AVU, ANUI, MQ — "
    "по весам рёбер сети. Формулы — reports/METHODS.md, раздел 7."
)

st.subheader("Паспорта типов")
reg = registry().set_index("code")
feats = list(params["features"])
w = indicators_wide()
raw = w[w["year"].eq(year)].set_index("territory_id").reindex(labels.index)
extra = [c for c in ["pop", "wage", "payroll_pc", "budget_own_share", "gmp_imputed_share"] if c not in feats]
prof_n = X.assign(тип=labels.values).groupby("тип")[feats].mean()
prof_r = raw[feats + extra].assign(тип=labels.values).groupby("тип").median()
size = labels.value_counts().sort_index()
fig = go.Figure()
for c in prof_n.index:
    fig.add_trace(
        go.Bar(
            x=[reg.at[f, "name"][:30] for f in feats],
            y=prof_n.loc[c],
            name=f"{disp[int(c)]} ({size[c]})",
            marker_color=cluster_color(int(c)),
            marker_line_width=0,
            hovertemplate="%{x}<br>%{y:.2f}<extra>" + disp[int(c)] + "</extra>",
        )
    )
st.plotly_chart(
    layout(fig, 380, title="Средние нормированные значения признаков по типам", barmode="group", bargap=0.2),
    width="stretch",
)
mnames = mo().set_index("territory_id")
top = raw.assign(тип=labels.values, name=raw.index.map(mnames["name_short"])).sort_values("pop", ascending=False)
prof_r.insert(0, "МО", size)
prof_r["крупнейшие МО"] = top.groupby("тип")["name"].apply(lambda s: ", ".join(s.head(4)))
prof_r.index = [disp[int(c)] for c in prof_r.index]
prof_r.columns = [reg.at[c, "name"] if c in reg.index else c for c in prof_r.columns]
st.dataframe(prof_r, width="stretch")
st.caption(
    "Медианы исходных значений (ВМП — расчётная оценка команды; gmp_imputed_share — доля ВМП, распределённая по "
    "правилу для скрытых данных)."
)

with st.expander("Названия типов (сохраняются в data/cluster_names.yaml)"):
    with st.form("names"):
        new = {
            int(c): st.text_input(f"Тип {c} ({size[c]} МО)", value=names.get(int(c), ""), key=f"n_{nkey}_{c}")
            for c in sorted(labels.unique())
        }
        if st.form_submit_button("Сохранить названия"):
            save_names(nkey, new)
            st.success("Сохранено.")

st.subheader("Сравнение методов")
cmp_methods = st.multiselect(
    "Методы", ok_methods, default=[m for m in ok_methods if m != "canus"], format_func=METHOD_NAMES.get
)
rows = [{"метод": METHOD_NAMES[m], **run(pj, year, m, k)[1]} for m in cmp_methods]
comp = pd.DataFrame(rows)[["метод", *order]] if rows else pd.DataFrame()
if len(comp):
    st.dataframe(comp.style.format({c: "{:.3f}" for c in order}), width="stretch")

st.subheader(f"Выбор числа кластеров: {METHOD_NAMES[method]}, {year} г.")
if method == "canus":
    st.caption("CANUS медленный (30–120 с на разбиение): перебор k может занять несколько минут.")
kt = pd.DataFrame([{"k": kk, **run(pj, year, method, kk)[1]} for kk in range(2, 11)])
cols = st.columns(4)
for i, ind in enumerate(["SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "DBI"]):
    f = go.Figure(
        go.Scatter(
            x=kt["k"],
            y=kt[ind],
            mode="lines+markers",
            line=dict(width=2, color=CATEGORICAL[0]),
            marker=dict(size=8),
            hovertemplate="k = %{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    cols[i % 4].plotly_chart(layout(f, 220, title=f"{ind} {arrows[ind]}"), width="stretch")
st.caption(
    "S_Dbw на этих данных обычно монотонно убывает с ростом k и выбирает почти максимальное k (9–10) — сам по себе для выбора k "
    "он не годится. Сводная таблица по всем годам и методам — reports/CLUSTERS.md."
)

table = pd.DataFrame(
    {
        "territory_id": labels.index,
        "МО": labels.index.map(mnames["name"]),
        "регион": labels.index.map(mnames["region_name"]),
        "тип": labels.values,
        "название": [disp[int(c)] for c in labels.values],
    }
)
st.dataframe(table, width="stretch", height=300)
downloads(
    table,
    {"боковая_панель": side, "метод": method, "k": k, "год": year, "hash_сети": h, "индексы": scores, "сеть": params},
    "clusters",
)
