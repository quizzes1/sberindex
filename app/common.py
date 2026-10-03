"""Общие элементы интерфейса: боковая панель, загрузка данных (кэш), выгрузки, карты, палитры.

Все расчёты — в модулях src/; здесь только вызовы и отрисовка.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from src import network  # noqa: E402
from src.io import DATA, GEO, PROCESSED, load_yaml  # noqa: E402

# ---------------------------------------------------------------- палитры (руководство dataviz, проверенный набор)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_GRAY = "#9a9994"  # кластеры сверх 8 — нейтральный серый (не генерируем новые оттенки)
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# расходящаяся: синий ↔ красный, нейтральная серая середина, равные плечи
DIVERGING = [[0.0, "#184f95"], [0.5, "#f0efec"], [1.0, "#e34948"]]
FDS = ["ЦФО", "СЗФО", "ЮФО", "СКФО", "ПФО", "УФО", "СФО", "ДФО"]


def cluster_color(i: int) -> str:
    return CATEGORICAL[i] if 0 <= i < len(CATEGORICAL) else OTHER_GRAY


def layout(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    """Сдержанное оформление: тонкая сетка, без лишних рамок, подписи — текстовыми цветами."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(namelength=-1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **kw,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


# ---------------------------------------------------------------- данные (кэш)
@st.cache_data(show_spinner=False)
def mo() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "mo.parquet")


@st.cache_data(show_spinner=False)
def indicators_wide() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "indicators_wide.parquet")


@st.cache_data(show_spinner=False)
def registry() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "indicator_registry.parquet")


@st.cache_data(show_spinner=False)
def panel_long() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "panel_long.parquet")


@st.cache_data(show_spinner=False)
def geojson() -> dict:
    with open(GEO / "mo_simplified.geojson", encoding="utf-8") as f:
        return json.load(f)


def regions_table() -> pd.DataFrame:
    m = mo()
    return m.groupby(["region_code", "region_name", "federal_district"]).size().rename("МО").reset_index()


# ---------------------------------------------------------------- боковая панель
def sidebar() -> dict:
    """Общие настройки для всех страниц; хранятся в st.session_state."""
    s = st.session_state
    st.sidebar.header("Выборка и предобработка")
    kind = st.sidebar.radio(
        "Выборка",
        ["ДФО", "Россия", "Свой список субъектов"],
        index=["ДФО", "Россия", "Свой список субъектов"].index(s.get("sample_kind", "ДФО")),
        key="sample_kind",
    )
    regions: list[int] = []
    if kind == "Свой список субъектов":
        rt = regions_table().sort_values("region_name")
        names = st.sidebar.multiselect(
            "Субъекты",
            rt["region_name"].tolist(),
            default=s.get("regions_sel", ["Приморский край", "Хабаровский край"]),
            key="regions_sel",
        )
        regions = rt.loc[rt["region_name"].isin(names), "region_code"].astype(int).tolist()
    years = st.sidebar.slider("Годы", 2013, 2024, value=s.get("years_sel", (2017, 2024)), key="years_sel")
    excl = st.sidebar.checkbox("Исключить МО со сменой границ", value=s.get("excl_bc", False), key="excl_bc")
    method = st.sidebar.selectbox(
        "Нормировка",
        ["minmax", "zscore"],
        key="norm_method",
        format_func={"minmax": "min-max в [0, 1]", "zscore": "z-score"}.get,
    )
    scope = st.sidebar.selectbox(
        "Область нормировки",
        ["panel", "year"],
        key="norm_scope",
        format_func={"panel": "вся панель (рекомендуется)", "year": "каждый год отдельно"}.get,
    )
    if scope == "year":
        st.sidebar.caption(
            "⚠️ При нормировке по годам общий рост показателей исчезает, и «переходы» между "
            "кластерами во времени отчасти становятся артефактом нормировки."
        )
    st.sidebar.caption("ВМП — расчётная оценка команды, не официальная статистика.")
    fds = FDS if kind == "Россия" else (["ДФО"] if kind == "ДФО" else [])
    return {
        "kind": kind,
        "federal_districts": fds,
        "regions": regions,
        "years": [int(years[0]), int(years[1])],
        "exclude_boundary_change": bool(excl),
        "method": method,
        "scope": scope,
    }


def sample_ids(side: dict) -> set[int]:
    """territory_id выборки (по ФО или субъектам, с учётом исключения смен границ)."""
    m = mo()
    sel = (
        m["region_code"].isin(side["regions"])
        if side["regions"]
        else m["federal_district"].isin(side["federal_districts"])
    )
    if side["exclude_boundary_change"]:
        sel &= ~m["boundary_change"]
    return set(m.loc[sel, "territory_id"].astype(int))


def sample_rows(side: dict) -> pd.DataFrame:
    """Строки indicators_wide выборки, действующие в году, в выбранных годах."""
    w = indicators_wide()
    y0, y1 = side["years"]
    return w[w["valid_in_year"] & w["year"].between(y0, y1) & w["territory_id"].isin(sample_ids(side))]


def network_params(side: dict, override: dict | None = None) -> dict:
    """Параметры сети = configs/network.yaml + выборка и нормировка из боковой панели + override страницы."""
    p = network.default_params()
    p = network.merge_params(
        p,
        {
            "sample": {
                "federal_districts": side["federal_districts"],
                "regions": side["regions"],
                "years": side["years"],
                "exclude_boundary_change": side["exclude_boundary_change"],
            },
            "preprocess": {"method": side["method"], "scope": side["scope"]},
        },
    )
    return network.merge_params(p, override or {})


# ---------------------------------------------------------------- выгрузки
def downloads(table: pd.DataFrame | None, params: dict, name: str) -> None:
    """Кнопки выгрузки текущей таблицы (CSV) и текущих параметров (YAML) — для воспроизводимости."""
    c1, c2 = st.columns(2)
    if table is not None:
        buf = io.StringIO()
        table.to_csv(buf, index=False)
        c1.download_button(
            "⬇ Таблица (CSV)", buf.getvalue().encode("utf-8-sig"), f"{name}.csv", "text/csv", key=f"dl_csv_{name}"
        )
    p = json.loads(
        json.dumps(
            {"страница": name, **params},
            ensure_ascii=False,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )
    c2.download_button(
        "⬇ Параметры (YAML)",
        yaml.safe_dump(p, allow_unicode=True, sort_keys=False).encode("utf-8"),
        f"{name}_params.yaml",
        "text/yaml",
        key=f"dl_yaml_{name}",
    )


# ---------------------------------------------------------------- карты
def geo_layout(fig: go.Figure, ids: set[int], height: int = 620) -> go.Figure:
    """Азимутальная равновеликая проекция с центром в середине выборки: Россия и Чукотка без разрыва
    на 180-м меридиане (полигоны хранятся в долготах 0…360); масштаб — по угловому размеру выборки."""
    m = mo()
    sub = m[m["territory_id"].isin(ids)]
    lat = sub["lat"].dropna()
    lon = sub["lon360"].dropna()
    if len(lat):
        la0, la1 = float(lat.quantile(0.0)) - 2, float(lat.quantile(1.0)) + 2
        lo0, lo1 = float(lon.min()) - 3, float(lon.max()) + 3
        clat, clon = (la0 + la1) / 2, (lo0 + lo1) / 2
        ext_lon = max((lo1 - lo0) * np.cos(np.radians(clat)), 4)
        ext_lat = max(la1 - la0, 4)
        scale = min(190 / ext_lon, 150 / ext_lat)  # подобрано по виду: область карты plotly почти квадратная
    else:
        clat, clon, scale = 60, 105, 2
    fig.update_geos(
        projection_type="azimuthal equal area",
        projection_rotation=dict(lon=clon, lat=clat),
        center=dict(lon=clon, lat=clat),
        projection_scale=scale,
        visible=False,
        showland=True,
        landcolor="rgba(128,128,128,0.08)",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def features_subset(ids: set[int]) -> dict:
    g = geojson()
    return {
        "type": "FeatureCollection",
        "features": [f for f in g["features"] if int(f["properties"]["territory_id"]) in ids],
    }


def choropleth(values: pd.Series, title: str, log: bool = False, fmt: str = ",.4~s") -> go.Figure:
    """Картограмма показателя (одна синяя шкала). values: индекс territory_id.

    log=True — цвет по lg(значения) для скошенных показателей (иначе несколько крупных значений
    «выжигают» палитру); в подсказке — исходное значение.
    """
    v = values.dropna()
    if log:
        v = v[v > 0]
    ids = set(int(i) for i in v.index)
    names = mo().set_index("territory_id")["name"]
    z = np.log10(v.values) if log else v.values
    cb = dict(title="", thickness=12)
    if log and len(v):
        ticks = np.arange(np.floor(z.min()), np.ceil(z.max()) + 1)
        cb.update(tickvals=ticks, ticktext=[f"{10**t:,.0f}".replace(",", " ") for t in ticks])
    fig = go.Figure(
        go.Choropleth(
            geojson=features_subset(ids),
            featureidkey="properties.territory_id",
            locations=list(v.index),
            z=z,
            customdata=v.values,
            colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
            marker_line_width=0.3,
            marker_line_color="rgba(255,255,255,0.6)",
            text=[names.get(i, i) for i in v.index],
            colorbar=cb,
            hovertemplate="%{text}<br>%{customdata:" + fmt + "}<extra></extra>",
        )
    )
    fig.update_layout(title=title + (" (цвет — логарифмическая шкала)" if log else ""))
    return geo_layout(fig, ids)


def cluster_map(labels: pd.Series, names: dict | None = None, title: str = "") -> go.Figure:
    """Карта кластеров: каждый кластер — свой слой (легенда = расшифровка цвета). labels: индекс territory_id."""
    mnames = mo().set_index("territory_id")["name"]
    fig = go.Figure()
    ids = set(int(i) for i in labels.index)
    for k in sorted(labels.unique()):
        sel = labels[labels.eq(k)]
        lab = (names or {}).get(int(k)) or f"Тип {k}"
        fig.add_trace(
            go.Choropleth(
                geojson=features_subset(set(int(i) for i in sel.index)),
                featureidkey="properties.territory_id",
                locations=list(sel.index),
                z=[1] * len(sel),
                colorscale=[[0, cluster_color(int(k))], [1, cluster_color(int(k))]],
                showscale=False,
                name=f"{lab} ({len(sel)})",
                showlegend=True,
                marker_line_width=0.3,
                marker_line_color="rgba(255,255,255,0.7)",
                text=[mnames.get(i, i) for i in sel.index],
                hovertemplate="%{text}<br>" + lab + "<extra></extra>",
            )
        )
    fig.update_layout(title=title, legend=dict(orientation="h", y=-0.02))
    return geo_layout(fig, ids)


def graph_on_map(edges: pd.DataFrame, labels: pd.Series | None = None, title: str = "") -> go.Figure:
    """Граф: узлы — центры МО на карте, рёбра — линии; цвет узла — кластер (если задан)."""
    m = mo().set_index("territory_id")
    ids = set(int(i) for i in pd.unique(edges[["source", "target"]].values.ravel())) if len(edges) else set()
    if labels is not None:
        ids |= set(int(i) for i in labels.index)
    lat, lon = [], []
    for s, t in zip(edges["source"], edges["target"]):
        lat += [m.at[s, "lat"], m.at[t, "lat"], None]
        lon += [m.at[s, "lon"], m.at[t, "lon"], None]
    fig = go.Figure(
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="lines",
            line=dict(width=0.6, color="rgba(110,110,110,0.35)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    nodes = pd.Index(sorted(ids))
    if labels is None:
        fig.add_trace(
            go.Scattergeo(
                lat=m.loc[nodes, "lat"],
                lon=m.loc[nodes, "lon"],
                mode="markers",
                marker=dict(size=7, color=CATEGORICAL[0], line=dict(width=1, color="white")),
                text=m.loc[nodes, "name"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
    else:
        for k in sorted(labels.unique()):
            sel = labels.index[labels.eq(k)]
            fig.add_trace(
                go.Scattergeo(
                    lat=m.loc[sel, "lat"],
                    lon=m.loc[sel, "lon"],
                    mode="markers",
                    name=f"Тип {k}",
                    marker=dict(size=8, color=cluster_color(int(k)), line=dict(width=1, color="white")),
                    text=m.loc[sel, "name"],
                    hovertemplate="%{text}<br>Тип " + str(k) + "<extra></extra>",
                )
            )
    fig.update_layout(title=title)
    return geo_layout(fig, ids)


def graph_force(edges: pd.DataFrame, labels: pd.Series | None = None, seed: int = 42, title: str = "") -> go.Figure:
    """Силовая раскладка графа (networkx spring_layout, фиксированный seed)."""
    import networkx as nx

    g = nx.Graph()
    g.add_weighted_edges_from(edges[["source", "target", "weight"]].itertuples(index=False, name=None))
    if labels is not None:
        g.add_nodes_from(int(i) for i in labels.index)
    pos = nx.spring_layout(g, weight="weight", seed=seed, k=1.5 / np.sqrt(max(len(g), 1)))
    names = mo().set_index("territory_id")["name"]
    ex, ey = [], []
    for s, t in g.edges():
        ex += [pos[s][0], pos[t][0], None]
        ey += [pos[s][1], pos[t][1], None]
    fig = go.Figure(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            line=dict(width=0.5, color="rgba(110,110,110,0.35)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    groups = (
        {None: list(g.nodes())}
        if labels is None
        else {k: list(labels.index[labels.eq(k)]) for k in sorted(labels.unique())}
    )
    for k, nodes in groups.items():
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in nodes],
                y=[pos[n][1] for n in nodes],
                mode="markers",
                name="МО" if k is None else f"Тип {k}",
                showlegend=k is not None,
                marker=dict(
                    size=8,
                    color=CATEGORICAL[0] if k is None else cluster_color(int(k)),
                    line=dict(width=1, color="white"),
                ),
                text=[names.get(n, n) for n in nodes],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return layout(fig, 560, title=title)


# ---------------------------------------------------------------- прочее
def cluster_dir(net_hash: str) -> Path:
    return DATA / "clusters" / net_hash


NAMES_FILE = DATA / "cluster_names.yaml"


def load_names() -> dict:
    if NAMES_FILE.exists():
        return yaml.safe_load(NAMES_FILE.read_text(encoding="utf-8")) or {}
    return {}


def save_names(key: str, names: dict) -> None:
    allnames = load_names()
    allnames[key] = {int(k): v for k, v in names.items() if v}
    NAMES_FILE.write_text(yaml.safe_dump(allnames, allow_unicode=True, sort_keys=True), encoding="utf-8")


def clustering_cfg() -> dict:
    return load_yaml("clustering.yaml")
