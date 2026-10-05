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
    to_view,
    unit_label,
)

from src import clustering, icvi, network

st.set_page_config(page_title="Кластеры", layout="wide")
side = sidebar()
st.title("Кластеры")

cc = clustering_cfg()
METHOD_NAMES = {
    "kmeans": "k-средних",
    "ward": "иерархическая Уорда",
    "ward_kmeans": "Уорд + k-means",
    "gmm": "гауссовы смеси",
    "leiden": "Leiden (граф)",
    "spectral": "спектральная (граф)",
    "kefrin": "KEFRiN (атрибутированная сеть)",
    "canus": "CANUS (атрибутированная сеть, медленный)",
}
avail = {m: clustering.available(m) for m in cc["methods"]}


@st.cache_data(show_spinner="Строю сеть…", max_entries=4)  # сеть всей России — ~120 МБ
def net_year(params_json: str, year: int):
    net = network.build(json.loads(params_json), years=[year])[year]
    return net.ids, net.X, net.W * net.A, net.edges()


@st.cache_data(show_spinner="Одна модель на всю панель…", max_entries=16)
def pooled(params_json: str, method: str, k: int) -> pd.Series:
    """Режим pooled: метки (territory_id, year) одной модели на все МО-годы, K1 — самый высокий ВМП."""
    p = json.loads(params_json)
    X = network.features(p).dropna()
    mp = {**cc["methods"][method], "method": method, "k": k, "seed": cc["seed"]}
    return clustering.order_labels(clustering.fit_pooled(X, mp), p["prices"])


@st.cache_data(show_spinner="Кластеризация…", max_entries=256)
def run(params_json: str, year: int, method: str, k: int, mode: str = "per_year"):
    """Метки года (0 → K1 — самый высокий ВМП на душу в ценах базового года) и индексы качества."""
    ids, X, Wsp, _ = net_year(params_json, year)
    if mode == "pooled" and clustering.FAMILIES.get(method) == "attributes":
        lab = pooled(params_json, method, k).xs(year, level="year").reindex(ids)
        lab = lab.fillna(-1).astype(int).to_numpy()  # узлы сети = МО с полным набором признаков, как и в pooled
    else:
        mp = {**cc["methods"][method], "method": method, "k": k, "seed": cc["seed"]}
        lab = clustering.fit(X.to_numpy(), Wsp, mp)
        lab = clustering.order_labels(pd.Series(lab, index=ids), json.loads(params_json)["prices"], year).to_numpy()
    return pd.Series(lab, index=ids), icvi.compute_all(X.to_numpy(), Wsp, lab)


@st.cache_data(show_spinner="Строю дерево Уорда…", max_entries=8)
def ward_tree(params_json: str, year: int, mode: str):
    p = json.loads(params_json)
    if mode == "pooled":
        X = network.features(p).dropna().to_numpy()
    else:
        X = net_year(params_json, year)[1].to_numpy()
    Z, idx = clustering.ward_linkage(X, cc["methods"]["ward_kmeans"].get("ward_max_n"), cc["seed"])
    return Z, len(X), len(idx)


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

c0, c1, c2, c3 = st.columns([1.2, 1.2, 1, 1])
mode = c0.selectbox(
    "Режим",
    ["per_year", "pooled"],
    format_func={"per_year": "каждый год отдельно", "pooled": "одна модель на все годы (pooled)"}.get,
    help="pooled — одна модель на все МО-годы (только методы по атрибутам): тип одинаково определён во всех годах.",
)
ok_methods = [m for m in cc["methods"] if avail[m][0]]
if mode == "pooled":
    ok_methods = [m for m in ok_methods if clustering.FAMILIES.get(m) == "attributes"]
default_m = "ward_kmeans" if mode == "pooled" else ("kefrin" if "kefrin" in ok_methods else ok_methods[0])
method = c1.selectbox("Метод", ok_methods, index=ok_methods.index(default_m), format_func=METHOD_NAMES.get)
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

labels, scores = run(pj, year, method, k, mode)
ids, X, Wsp, edges = net_year(pj, year)
nkey = f"{h}|{method}|k{k}|pooled" if mode == "pooled" else f"{h}|{method}|k{k}|{year}"
names = load_names().get(nkey, {})
# подпись: K1…Kn (по убыванию медианы ВМП на душу в ценах базового года), название аналитиков — рядом
disp = {
    int(c): clustering.code(c) + (f" · {names[int(c)]}" if names.get(int(c)) else "") for c in sorted(labels.unique())
}
st.caption(
    "Кластеры пронумерованы K1…Kn по медиане ВМП на душу в ценах базового года, по убыванию: K1 — самый высокий."
    + (" Режим pooled: номера одинаковы во всех годах." if mode == "pooled" else "")
)

if method in ("ward", "ward_kmeans"):
    with st.expander("Дендрограмма Уорда и выбор k по скачку расстояния слияния", expanded=method == "ward_kmeans"):
        from scipy.cluster.hierarchy import dendrogram

        Z, n_all, n_tree = ward_tree(pj, year, mode)
        P = 40  # усечённое дерево: верхние 40 ветвей
        d = dendrogram(Z, truncate_mode="lastp", p=P, no_plot=True)
        cut = (Z[-(k - 1), 2] + Z[-k, 2]) / 2  # между слияниями k → k−1 и k+1 → k
        fd = go.Figure()
        for xs, ys in zip(d["icoord"], d["dcoord"]):
            fd.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=CATEGORICAL[0], width=1.5), hoverinfo="skip")
            )
        fd.add_hline(y=cut, line=dict(color="#e34948", dash="dash"))
        fd.update_layout(showlegend=False)
        fd.update_xaxes(
            tickvals=[5 + 10 * i for i in range(len(d["ivl"]))], ticktext=d["ivl"], tickangle=-90, tickfont=dict(size=9)
        )
        fd.update_yaxes(title="расстояние слияния")
        st.plotly_chart(
            layout(fd, 360, title=f"Верхние {P} ветвей дерева; пунктир — разрез на k = {k}"), width="stretch"
        )
        jt = clustering.ward_jumps(Z)
        best = int(jt[jt["k"].ge(3)].sort_values("скачок, раз", ascending=False)["k"].iloc[0])
        st.dataframe(
            jt.style.format({"высота слияния k→k−1": "{:.3f}", "скачок": "{:.3f}", "скачок, раз": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Подписи внизу — число {'МО-лет' if mode == 'pooled' else 'МО'} в ветви (в скобках). "
            f"Наибольший относительный скачок высоты слияния — естественная граница: по нему k = {best} "
            "(k = 2 не рассматривается — последнее слияние всегда самое высокое). Сравните с индексами качества ниже."
            + (
                f" Дерево построено по случайной подвыборке {n_tree} из {n_all} строк (память растёт как n²)."
                if n_tree < n_all
                else ""
            )
        )

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
raw = to_view(w[w["year"].eq(year)], list(w.columns), side).set_index("territory_id").reindex(labels.index)
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
    f"Медианы исходных значений; ВМП на душу — {unit_label('gmp_pc', side)}, остальные денежные — так же "
    "(ВМП — расчётная оценка команды; gmp_imputed_share — доля ВМП, распределённая по "
    "правилу для скрытых данных)."
)

with st.expander("Названия типов (сохраняются в data/cluster_names.yaml)"):
    with st.form("names"):
        new = {
            int(c): st.text_input(
                f"{clustering.code(c)} ({size[c]} МО)", value=names.get(int(c), ""), key=f"n_{nkey}_{c}"
            )
            for c in sorted(labels.unique())
        }
        if st.form_submit_button("Сохранить названия"):
            save_names(nkey, new)
            st.success("Сохранено.")

st.subheader("Сравнение методов")
cmp_methods = st.multiselect(
    "Методы", ok_methods, default=[m for m in ok_methods if m != "canus"], format_func=METHOD_NAMES.get
)
rows = [{"метод": METHOD_NAMES[m], **run(pj, year, m, k, mode)[1]} for m in cmp_methods]
comp = pd.DataFrame(rows)[["метод", *order]] if rows else pd.DataFrame()
if len(comp):
    st.dataframe(comp.style.format({c: "{:.3f}" for c in order}), width="stretch")

st.subheader(f"Выбор числа кластеров: {METHOD_NAMES[method]}, {year} г.")
slow = method in ("kefrin", "canus") and len(ids) > 1000
if method == "canus":
    st.caption("CANUS медленный (30–120 с на разбиение): перебор k может занять несколько минут.")
if slow and not st.session_state.get(f"ktable_{method}_{year}_{len(ids)}"):
    st.caption(f"{METHOD_NAMES[method]} на {len(ids)} МО считается долго (минуты на перебор k = 2…10).")
    if st.button("Посчитать перебор k"):
        st.session_state[f"ktable_{method}_{year}_{len(ids)}"] = True
        st.rerun()
else:
    kt = pd.DataFrame([{"k": kk, **run(pj, year, method, kk, mode)[1]} for kk in range(2, 11)])
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
        "кластер": [clustering.code(c) for c in labels.values],
        "название": [names.get(int(c), "") for c in labels.values],
    }
)
st.dataframe(table, width="stretch", height=300)
downloads(
    table,
    {
        "боковая_панель": side,
        "режим": mode,
        "метод": method,
        "k": k,
        "год": year,
        "hash_сети": h,
        "индексы": scores,
        "сеть": params,
    },
    "clusters",
)
