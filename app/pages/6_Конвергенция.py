"""Конвергенция: σ по годам, «начальный уровень — рост» с β-регрессией, коэффициенты, по типам."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    CATEGORICAL,
    YEARS_MAX,
    YEARS_MIN,
    cluster_color,
    downloads,
    layout,
    mo,
    real_prices,
    registry,
    sample_ids,
    sidebar,
    unit_label,
)

from src import clustering, convergence
from src.io import load_yaml

st.set_page_config(page_title="Конвергенция", layout="wide")
side = sidebar()
st.title("Конвергенция")

cc = load_yaml("dynamics.yaml")["convergence"]
reg = registry().set_index("code")
SERIES = {
    "gmp_basic": "ВМП на душу, реальный — базовый метод (единый для всего окна)",
    "gmp_sectoral": "ВМП на душу, реальный — отраслевой метод (2017–2024)",
}
mon = reg["monetary"].fillna(False).astype(bool) if "monetary" in reg.columns else pd.Series(False, index=reg.index)
money = [c for c in reg.index if mon.get(c) and c != "gmp_pc" and reg.at[c, "parent"] == c] + [
    c for c in ["hhi_emp", "density", "old_age_share", "budget_own_share", "living_space_pc"] if c in reg.index
]
opts = list(SERIES) + money
c1, c2 = st.columns([2, 1])
var = c1.selectbox("Показатель", opts, format_func=lambda c: SERIES.get(c) or f"{reg.at[c, 'name']} ({c})")
# окно — из боковой панели (одно на весь сайт); отраслевой ВМП есть только с 2017 г.
sy0, sy1 = side["years"]
lo = max(YEARS_MIN, 2017) if var == "gmp_sectoral" else YEARS_MIN
y0, y1 = c2.slider("Окно", lo, YEARS_MAX, (max(sy0, lo), max(sy1, lo + 1)))
pr = real_prices(side)
if var in SERIES or mon.get(var):
    st.caption(
        f"Денежные показатели — в ценах {pr['base_year']} г. ({unit_label('gmp_pc' if var in SERIES else var, side, True)}). "
        "Конвергенция всегда считается в реальных ценах, даже если в боковой панели выбраны текущие."
    )
if var in SERIES:
    st.caption(
        "ВМП — расчётная оценка команды. Для длинного окна берётся базовый метод: склейка базового (до 2016) и "
        "отраслевого (с 2017) создала бы искусственный скачок."
    )


@st.cache_data(show_spinner=False, max_entries=16)
def series(var: str, pr_key: str) -> pd.DataFrame:
    """territory_id, year, value — МО, действующие в конце окна (с достроенными значениями преемников)."""
    m = mo()
    if var in SERIES:
        spec = {"source": "gmp", "method": "basic" if var == "gmp_basic" else "sectoral", "column": "gmp_pc"}
    else:
        spec = {"source": "indicators", "column": var}
    return convergence.load_series(spec, m.loc[m["active"], "territory_id"], json.loads(pr_key))


d = series(var, json.dumps(pr, sort_keys=True))
d = d[d["year"].between(y0, y1)]
ids = sample_ids(side)
if (y1 - y0 + 1) < cc["min_years_reliable"]:
    st.warning(
        f"Окно {y0}–{y1} — {y1 - y0 + 1} лет. При окне короче ~9–10 лет выводы о конвергенции ненадёжны: σ-тренд "
        "опирается на несколько точек, панельная β-оценка смещена (смещение Никелла)."
    )

samples = {"Выборка": d[d["territory_id"].isin(ids)], "Россия": d}
types = st.session_state.get("types_last")
if types:
    t = pd.Series(types)
    for k in sorted(t.unique()):
        samples[clustering.code(k)] = d[d["territory_id"].isin(set(t.index[t.eq(k)]) & ids)]
    st.caption("Типы — сквозные типы последнего года со страницы «Динамика».")
else:
    st.caption("Чтобы считать конвергенцию по типам, откройте страницу «Динамика» (типы берутся оттуда).")

region = mo().set_index("territory_id")["region_code"]


@st.cache_data(show_spinner="Считаю…", max_entries=64)
def compute(var: str, y0: int, y1: int, key: str, ids_tuple: tuple):
    sd = d[d["territory_id"].isin(ids_tuple)]
    st_, tr = convergence.sigma(sd)
    ba = convergence.beta_absolute(sd, y0, y1)
    bp = convergence.beta_panel(sd, region)
    ps = convergence.log_t(convergence._log_panel(sd), cc["phillips_sul"]["trim"], cc["phillips_sul"]["hp_lambda"])
    return st_, tr, ba, bp, ps


res = {
    name: compute(var, y0, y1, name, tuple(sorted(sd["territory_id"].unique())))
    for name, sd in samples.items()
    if sd["territory_id"].nunique() >= 5
}

st.subheader("σ-конвергенция: разброс ln y по МО")
fig = go.Figure()
for i, (name, (st_, *_)) in enumerate(res.items()):
    color = cluster_color(int(name[1:]) - 1) if name[:1] == "K" and name[1:].isdigit() else CATEGORICAL[i]
    fig.add_trace(
        go.Scatter(
            x=st_["year"],
            y=st_["sd_log"],
            mode="lines+markers",
            name=name,
            line=dict(width=2, color=color, dash="dot" if name[:1] == "K" else "solid"),
            marker=dict(size=8),
            hovertemplate=name + "<br>%{x}: σ = %{y:.3f}<extra></extra>",
        )
    )
st.plotly_chart(layout(fig, 380, yaxis_title="σ(ln y)"), width="stretch")

st.subheader("β-конвергенция: начальный уровень и среднегодовой рост")
pick = st.selectbox("Выборка для диаграммы", list(res))
ba = res[pick][2]
if np.isfinite(ba.get("b", np.nan)):
    names = mo().set_index("territory_id")["name"]
    x, g = ba["level0"], ba["growth"]
    xx = np.linspace(x.min(), x.max(), 50)
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=g,
            mode="markers",
            marker=dict(size=8, color=CATEGORICAL[0], opacity=0.7, line=dict(width=1, color="white")),
            text=[names.get(i, i) for i in x.index],
            name="МО",
            hovertemplate="%{text}<br>ln y₀ = %{x:.2f}<br>рост = %{y:.2%} в год<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xx,
            y=ba["a"] + ba["b"] * xx,
            mode="lines",
            line=dict(width=2, color=CATEGORICAL[1]),
            name=f"β = {ba['b']:.4f} (p = {ba['p']:.3f})",
            hoverinfo="skip",
        )
    )
    st.plotly_chart(
        layout(
            fig,
            420,
            xaxis_title=f"ln y, {y0}",
            yaxis_title=f"среднегодовой рост ln y, {y0}–{y1}",
            yaxis_tickformat=".1%",
        ),
        width="stretch",
    )

st.subheader("Коэффициенты")
rows = []
for name, (_st, tr, ba, bp, ps) in res.items():
    rows.append(
        {
            "выборка": name,
            "МО": ba.get("n"),
            "σ: наклон в год": tr["slope"],
            "σ: p": tr["p"],
            "β абсолютная": ba.get("b"),
            "β: p": ba.get("p"),
            "скорость λ": ba.get("lambda"),
            "полупериод, лет": ba.get("half_life"),
            "β панельная (FE МО и года)": bp.get("b"),
            "β панель: p": bp.get("p"),
            "log t (Phillips–Sul): t": ps.get("t"),
        }
    )
tab = pd.DataFrame(rows)
st.dataframe(tab.style.format({c: "{:.4f}" for c in tab.columns if c not in ("выборка", "МО")}), width="stretch")
st.caption(
    "σ: наклон < 0 — разброс сокращается. β < 0 — МО с низким начальным уровнем растут быстрее; λ — скорость "
    "сближения, полупериод — за сколько лет отставание сокращается вдвое. Панельная β учитывает собственный уровень "
    "каждого МО (условная конвергенция). log t: t < −1,65 — общей конвергенции нет (возможны клубы)."
)
downloads(tab, {"боковая_панель": side, "показатель": var, "окно": [y0, y1]}, "convergence")
