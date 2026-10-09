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

from src import network, prices, update  # noqa: E402
from src.io import DATA, GEO, PROCESSED, load_yaml  # noqa: E402

# ---------------------------------------------------------------- палитры (руководство dataviz, проверенный набор)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_GRAY = "#9a9994"  # кластеры сверх 8 — нейтральный серый (не генерируем новые оттенки)
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# расходящаяся: синий ↔ красный, нейтральная серая середина, равные плечи
DIVERGING = [[0.0, "#184f95"], [0.5, "#f0efec"], [1.0, "#e34948"]]
FDS = ["ЦФО", "СЗФО", "ЮФО", "СКФО", "ПФО", "УФО", "СФО", "ДФО"]


def cluster_color(i: int) -> str:
    """Цвет кластера по номеру (K1 — первый цвет палитры); после восьмого — серый."""
    return CATEGORICAL[i] if 0 <= i < len(CATEGORICAL) else OTHER_GRAY


def layout(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    """Сдержанное оформление: тонкая сетка, без лишних рамок, шрифт сайта (Onest).

    Если есть и заголовок, и легенда, заголовок стоит в самом верху, а легенда — между ним и графиком,
    чтобы они не налезали друг на друга."""
    has_title = bool(kw.get("title"))
    has_legend = sum(1 for t in fig.data if t.showlegend is not False and t.name) > 1
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=86 if has_title and has_legend else 44, b=10),
        font=dict(family="Onest, sans-serif"),
        hoverlabel=dict(namelength=-1, font=dict(family="Onest, sans-serif")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        **kw,
    )
    if has_title:
        fig.update_layout(title_x=0, title_xanchor="left", title_y=0.985, title_yanchor="top", title_font_size=15)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


# ---------------------------------------------------------------- данные (кэш)
@st.cache_resource(show_spinner=False)  # один общий объект на процесс: только чтение
def mo() -> pd.DataFrame:
    """Справочник муниципалитетов (общий кэш, только чтение)."""
    return pd.read_parquet(PROCESSED / "mo.parquet")


@st.cache_resource(show_spinner=False)  # один общий объект на процесс: только чтение
def indicators_wide() -> pd.DataFrame:
    """Показатели в широком виде: строка — муниципалитет × год (общий кэш, только чтение)."""
    return pd.read_parquet(PROCESSED / "indicators_wide.parquet")


@st.cache_resource(show_spinner=False)  # один общий объект на процесс: только чтение
def registry() -> pd.DataFrame:
    """Раскрытый реестр показателей (общий кэш, только чтение)."""
    return pd.read_parquet(PROCESSED / "indicator_registry.parquet")


@st.cache_resource(show_spinner=False)  # один общий объект на процесс: только чтение
def panel_long() -> pd.DataFrame:
    """Панель в длинном виде: муниципалитет × год × ряд (общий кэш, только чтение)."""
    return pd.read_parquet(PROCESSED / "panel_long.parquet")


@st.cache_resource(show_spinner=False)  # один общий объект на процесс: только чтение
def geojson(coarse: bool = False) -> dict:
    """Полигоны для карт: детальные (допуск 0,01°) или облегчённые для всей России (0,03°)."""
    name = (
        "mo_simplified_coarse.geojson"
        if coarse and (GEO / "mo_simplified_coarse.geojson").exists()
        else "mo_simplified.geojson"
    )
    with open(GEO / name, encoding="utf-8") as f:
        return json.load(f)


COARSE_FROM = 800  # с какого числа МО на карте брать облегчённые полигоны


def regions_table() -> pd.DataFrame:
    """Субъекты с федеральным округом и числом муниципалитетов."""
    m = mo()
    return m.groupby(["region_code", "region_name", "federal_district"]).size().rename("МО").reset_index()


# ---------------------------------------------------------------- боковая панель
_SEEN_UPDATE = {"finished": None, "first": True}  # общий для процесса Streamlit (модуль импортируется один раз)


def refresh_after_update() -> dict:
    """После пересчёта со страницы «Обновление данных» один раз сбросить кэши (st.cache_data и lru_cache в src),
    чтобы все страницы читали новые файлы. Возвращает состояние последнего пересчёта."""
    stt = update.read_status()
    fin = stt.get("finished") if stt.get("state") == "done" else None
    if fin and fin != _SEEN_UPDATE["finished"]:
        if not _SEEN_UPDATE["first"]:
            st.cache_data.clear()
            st.cache_resource.clear()
            update.clear_caches()
        _SEEN_UPDATE["finished"] = fin
    _SEEN_UPDATE["first"] = False
    return stt


YEARS_MIN, YEARS_MAX = 2014, 2024  # окно в интерфейсе (2013 — нет темпа роста населения)

# Настройки боковой панели и страницы «Сеть», которые сохраняются при переходе между страницами
SIDEBAR_KEYS = (
    "sample_kind",
    "regions_sel",
    "years_sel",
    "excl_bc",
    "norm_method",
    "norm_scope",
    "price_values",
    "price_base_year",
    "price_scope",
    "price_spatial",
)


def _current_page() -> str | None:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx()
    return getattr(ctx, "page_script_hash", None) if ctx else None


def _mark_page() -> None:
    """Запомнить, открыта ли сейчас другая страница, чем в прошлый запуск (вызывается в начале sidebar())."""
    s = st.session_state
    page = _current_page()
    s["_page_changed"] = s.get("_page") != page
    s["_page"] = page


def keep(key: str, default):
    """Значение виджета с ключом key, сохранённое между страницами.

    Виджет на другой странице Streamlit считает новым и не берёт значение, записанное виджетом прошлой
    страницы, поэтому копия хранится под ключом «_key» и при смене страницы записывается в st.session_state
    заново (на той же странице — нет, иначе откатился бы только что сделанный выбор). Виджет создаётся только
    с key=, без value/index."""
    s = st.session_state
    if s.get("_page_changed", True) and "_" + key in s:
        s[key] = s["_" + key]
    elif key not in s:
        s[key] = s.get("_" + key, default)
    return s[key]


def remember(*keys: str) -> None:
    """Скопировать текущие значения виджетов в постоянные ключи «_key»."""
    s = st.session_state
    for k in keys:
        if k in s:
            s["_" + k] = s[k]


def sidebar() -> dict:
    """Общие настройки для всех страниц; хранятся в st.session_state и не сбрасываются при переходе по страницам."""
    s = st.session_state
    _mark_page()
    upd = refresh_after_update()
    if upd.get("state") == "running":
        st.sidebar.warning("Сейчас идёт пересчёт данных. Пока он не закончится, цифры могут быть неполными.")
    netd = network.default_params()
    pdef = prices.price_params()
    st.sidebar.header("Выборка")
    options = ["Россия", *FDS, "Свой список субъектов"]  # вся Россия по умолчанию; округа — пресеты
    if keep("sample_kind", "Россия") not in options:
        s["sample_kind"] = "Россия"
    kind = st.sidebar.selectbox(
        "Территория", options, key="sample_kind", format_func=lambda x: "Вся Россия" if x == "Россия" else x
    )
    regions: list[int] = []
    if kind == "Свой список субъектов":
        rt = regions_table().sort_values("region_name")
        keep("regions_sel", ["Приморский край", "Хабаровский край"])
        names = st.sidebar.multiselect("Субъекты", rt["region_name"].tolist(), key="regions_sel")
        regions = rt.loc[rt["region_name"].isin(names), "region_code"].astype(int).tolist()
    y_def = tuple(int(y) for y in netd["sample"]["years"])
    yv = keep("years_sel", y_def)
    if not (YEARS_MIN <= yv[0] <= yv[1] <= YEARS_MAX):
        s["years_sel"] = (max(YEARS_MIN, min(yv[0], YEARS_MAX)), max(YEARS_MIN, min(yv[1], YEARS_MAX)))
    years = st.sidebar.slider("Годы", YEARS_MIN, YEARS_MAX, key="years_sel")
    keep("excl_bc", bool(netd["sample"].get("exclude_boundary_change", False)))
    excl = st.sidebar.checkbox("Без муниципалитетов, менявших границы", key="excl_bc")
    keep("norm_method", netd["preprocess"]["method"])
    method = st.sidebar.selectbox(
        "Нормировка",
        ["minmax", "zscore"],
        key="norm_method",
        format_func={"minmax": "от 0 до 1 (min-max)", "zscore": "стандартизация (z-score)"}.get,
    )
    keep("norm_scope", netd["preprocess"]["scope"])
    scope = st.sidebar.selectbox(
        "Нормировать",
        ["panel", "year"],
        key="norm_scope",
        format_func={"panel": "по всем годам сразу", "year": "по каждому году"}.get,
    )
    if scope == "year":
        st.sidebar.caption(
            "При нормировке по каждому году общий рост показателей пропадает, и часть переходов между типами "
            "получается из-за шкалы, а не из-за экономики."
        )
    st.sidebar.header("Цены")
    keep("price_values", pdef["values"])
    values = st.sidebar.radio(
        "Денежные показатели",
        ["real", "nominal"],
        key="price_values",
        horizontal=True,
        format_func={"real": "в ценах одного года", "nominal": "в ценах своего года"}.get,
    )
    keep("price_base_year", int(pdef["base_year"]))
    base_year = st.sidebar.selectbox(
        "Цены какого года", list(range(YEARS_MIN, YEARS_MAX + 1)), key="price_base_year", disabled=values == "nominal"
    )
    keep("price_scope", pdef["deflator_scope"])
    dscope = st.sidebar.selectbox(
        "Индексы цен",
        ["national", "regional"],
        key="price_scope",
        disabled=values == "nominal",
        format_func={"national": "по России", "regional": "по субъектам (приблизительно)"}.get,
    )
    keep("price_spatial", bool(pdef["spatial_price_adjustment"]))
    spatial = st.sidebar.checkbox(
        "Учитывать разницу цен между регионами",
        key="price_spatial",
        disabled=values == "nominal",
        help="Учитывает, что в разных регионах жизнь стоит по-разному: зарплаты, ФОТ, бюджет и розница делятся "
        "на стоимость одинакового набора товаров и услуг в регионе относительно средней по России.",
    )
    if values == "nominal":
        st.sidebar.caption("В ценах своего года годы сравнивать нельзя: мешает инфляция.")
    if st.sidebar.button("Сбросить настройки", help="Вернуть значения по умолчанию на всех страницах"):
        for k in (*SIDEBAR_KEYS, "net_override", "cl_mode", "cl_method", "cl_k", "cl_year", "t2_periods"):
            s.pop(k, None)
            s.pop("_" + k, None)
        st.rerun()
    st.sidebar.caption("ВМП — наша расчётная оценка, а не данные Росстата.")
    remember(*SIDEBAR_KEYS)
    # [] — вся Россия (как в configs/network.yaml, чтобы хэш совпадал с готовыми результатами)
    fds = [kind] if kind in FDS else []
    return {
        "kind": kind,
        "federal_districts": fds,
        "regions": regions,
        "years": [int(years[0]), int(years[1])],
        "exclude_boundary_change": bool(excl),
        "method": method,
        "scope": scope,
        "prices": {
            "values": values,
            "base_year": int(base_year),
            "deflator_scope": dscope,
            "spatial_price_adjustment": bool(spatial),
        },
    }


# ---------------------------------------------------------------- разбиение и сводные таблицы (общие для всех страниц)
def cluster_defaults() -> dict:
    """Режим, метод и k по умолчанию — одни на весь сайт (configs/dynamics.yaml → partition)."""
    part = load_yaml("dynamics.yaml")["partition"]
    return {"mode": part.get("mode", "pooled"), "method": part["method"], "k": int(part["k"])}


@st.cache_data(show_spinner="Кластеризация…", max_entries=16)
def cluster_partition(pj: str, method: str, k: int, mode: str) -> pd.DataFrame:
    """territory_id, year, label (0 → K1) — одно разбиение для страниц «Кластеры», «Динамика», «Сводные таблицы»
    и главной: при одинаковых параметрах состав и номера кластеров везде совпадают."""
    from src import summary

    return summary.partition(json.loads(pj), method, k, mode)


@st.cache_data(show_spinner="Считаю характерные признаки…", max_entries=32)
def cluster_table1(pj: str, method: str, k: int, mode: str, year: int) -> dict:
    """Таблица 1 (характерные признаки кластеров) для разбиения и года, общий кэш."""
    from src import summary

    return summary.characteristic_table(cluster_partition(pj, method, k, mode), json.loads(pj), year)


@st.cache_data(show_spinner=False, max_entries=16)
def cluster_table2(pj: str, method: str, k: int, mode: str, periods: tuple[int, ...]):
    """Таблица 2 (кластеры по периодам): уровень МО, субъекты по числу МО и по населению, сводка."""
    from src import summary

    lb = cluster_partition(pj, method, k, mode)
    p = list(periods)
    mt = summary.periods_table(lb, p)
    return (
        mt,
        summary.subject_table(lb, p, "count"),
        summary.subject_table(lb, p, "pop"),
        summary.transitions_summary(mt),
    )


METHOD_NAMES = {
    "kmeans": "k-средних",
    "ward": "метод Уорда",
    "ward_kmeans": "Уорд + k-means",
    "gmm": "гауссовы смеси",
    "leiden": "Leiden (по сети)",
    "spectral": "спектральный (по сети)",
    "kefrin": "KEFRiN (показатели и сеть)",
    "canus": "CANUS (показатели и сеть, медленный)",
}


def cluster_controls(c_mode, c_method, c_k, exclude: tuple[str, ...] = ()) -> tuple[str, str, int]:
    """Режим, метод и k — одни и те же виджеты (ключи cl_mode, cl_method, cl_k) на страницах «Кластеры»,
    «Динамика», «Сводные таблицы»: выбор на одной странице действует на всех, по умолчанию — configs/dynamics.yaml."""
    from src import clustering

    s = st.session_state
    d = cluster_defaults()
    keep("cl_mode", d["mode"])
    mode = c_mode.selectbox(
        "Режим",
        ["pooled", "per_year"],
        key="cl_mode",
        format_func={"pooled": "одна модель на все годы", "per_year": "каждый год отдельно"}.get,
        help="Одна модель на все годы: тип K2 в 2014 и в 2024 году означает одно и то же, поэтому переходы между "
        "типами видны честно. Каждый год отдельно: типы пересчитываются заново и нумеруются по ВМП внутри года.",
    )
    methods = [m for m in clustering_cfg()["methods"] if clustering.available(m)[0] and m not in exclude]
    if mode == "pooled":
        methods = [m for m in methods if clustering.FAMILIES.get(m) == "attributes"]
    default_m = d["method"] if d["method"] in methods else methods[0]
    if keep("cl_method", default_m) not in methods:
        s["cl_method"] = default_m
    method = c_method.selectbox("Метод", methods, key="cl_method", format_func=lambda m: METHOD_NAMES.get(m, m))
    keep("cl_k", d["k"])
    k = c_k.slider("Число типов", 2, 10, key="cl_k")
    remember("cl_mode", "cl_method", "cl_k")
    return mode, method, int(k)


def cluster_year(col, side: dict, label: str = "Год") -> int:
    """Год просмотра кластеров — общий для страниц «Кластеры» и «Сводные таблицы» (по умолчанию — конец окна)."""
    y0, y1 = side["years"]
    s = st.session_state
    if not (y0 <= keep("cl_year", y1) <= y1):
        s["cl_year"] = y1
    year = col.select_slider(label, list(range(y0, y1 + 1)), key="cl_year")
    remember("cl_year")
    return int(year)


def table2_periods(years_all: list[int], col=None) -> list[int]:
    """Периоды таблицы 2 — общие для страницы «Сводные таблицы» и главной (по умолчанию — 3 равноудалённых года).

    col — куда нарисовать выбор (на главной не рисуется: берётся сохранённый выбор)."""
    from src import summary

    s = st.session_state
    default = [
        y for y in summary.default_periods(min(years_all), max(years_all), summary.cfg()["periods"]["n_default"])
    ]
    saved = [y for y in (s.get("t2_periods") or s.get("_t2_periods") or default) if y in years_all]
    if col is None:
        return sorted(saved) if len(saved) >= 2 else default
    if s.get("_page_changed", True) or "t2_periods" not in s or any(y not in years_all for y in s["t2_periods"]):
        s["t2_periods"] = saved if len(saved) >= 2 else default
    out = col.multiselect(
        "Периоды (годы)",
        years_all,
        key="t2_periods",
        help="По умолчанию — начало, середина и конец выбранных лет.",
    )
    remember("t2_periods")
    return sorted(out)


def plural(n: int, one: str, few: str, many: str) -> str:
    """Склонение по числу: plural(1, "тип", "типа", "типов") → «1 тип», 3 → «3 типа», 5 → «5 типов»."""
    n10, n100 = abs(n) % 10, abs(n) % 100
    word = one if n10 == 1 and n100 != 11 else few if 2 <= n10 <= 4 and not 12 <= n100 <= 14 else many
    return f"{n} {word}"


def partition_badge(h: str, mode: str, method: str, k: int) -> str:
    """Строка параметров разбиения — одна и та же на страницах «Кластеры», «Динамика», «Сводные таблицы»:
    если она совпадает, совпадают и кластеры."""
    m = "одна модель на все годы" if mode == "pooled" else "каждый год отдельно"
    return f"{METHOD_NAMES.get(method, method)}, {plural(k, 'тип', 'типа', 'типов')}, {m}. Сеть `{h}`."


def current_partition_params(side: dict) -> tuple[dict, str, str, str, int]:
    """Параметры сети (боковая панель + страница «Сеть») и выбранные режим, метод и k — как на страницах."""
    override = st.session_state.get("net_override", {})
    params = network_params(side, override)
    if "features" in override:
        params["features"] = override["features"]
    d = cluster_defaults()
    s = st.session_state
    mode = s.get("cl_mode", s.get("_cl_mode", d["mode"]))
    method = s.get("cl_method", s.get("_cl_method", d["method"]))
    if mode == "pooled" and network_family(method) != "attributes":
        method = d["method"]
    k = int(s.get("cl_k", s.get("_cl_k", d["k"])))
    return params, json.dumps(params, sort_keys=True, ensure_ascii=False), mode, method, k


def network_family(method: str) -> str:
    """Семейство метода кластеризации: attributes | graph | attributed_network."""
    from src import clustering

    return clustering.FAMILIES.get(method, "")


def real_prices(side: dict) -> dict:
    """Параметры цен для расчётов, которые всегда в реальных ценах (конвергенция)."""
    return {**side["prices"], "values": "real"}


def to_view(df: pd.DataFrame, codes, side: dict, force_real: bool = False) -> pd.DataFrame:
    """Денежные колонки codes в ценах, выбранных в боковой панели (df — с колонками region_code, year)."""
    pr = real_prices(side) if force_real else side["prices"]
    if pr["values"] == "nominal":
        return df
    return prices.to_real(df, codes, pr["base_year"], pr["deflator_scope"], pr["spatial_price_adjustment"])


def unit_label(code: str, side: dict, force_real: bool = False) -> str:
    """Единица показателя с подписью цен: «руб., в ценах 2023 г.»."""
    reg = registry().set_index("code")
    if code not in reg.index:
        return ""
    r = reg.loc[code]
    mon = bool(r.get("monetary")) if "monetary" in reg.columns and pd.notna(r.get("monetary")) else False
    pr = real_prices(side) if force_real else side["prices"]
    return prices.unit_label(str(r["unit"]), mon, pr)


def sample_ids(side: dict) -> set[int]:
    """territory_id выборки (по ФО или субъектам, с учётом исключения смен границ)."""
    m = mo()
    if side["regions"]:
        sel = m["region_code"].isin(side["regions"])
    elif side["federal_districts"]:
        sel = m["federal_district"].isin(side["federal_districts"])
    else:
        sel = pd.Series(True, index=m.index)  # вся Россия
    if side["exclude_boundary_change"]:
        sel &= ~m["boundary_change"]
    return set(m.loc[sel, "territory_id"].astype(int))


def sample_rows(side: dict) -> pd.DataFrame:
    """Строки indicators_wide выборки, действующие в году, в выбранных годах."""
    w = indicators_wide()
    y0, y1 = side["years"]
    return w[w["valid_in_year"] & w["year"].between(y0, y1) & w["territory_id"].isin(sample_ids(side))]


def network_params(side: dict, override: dict | None = None) -> dict:
    """Параметры сети = configs/network.yaml + выборка и нормировка из боковой панели + override страницы.

    Годы модели — всегда полная панель из configs/network.yaml (2014–2024): нормировка признаков и модель
    кластеризации не зависят от выбранного в боковой панели окна, оно только фильтрует показ. Иначе при
    расширении окна модель обучалась бы на других данных и тот же год получал бы другие кластеры."""
    p = network.default_params()
    p = network.merge_params(
        p,
        {
            "sample": {
                "federal_districts": side["federal_districts"],
                "regions": side["regions"],
                "years": list(p["sample"]["years"]),
                "exclude_boundary_change": side["exclude_boundary_change"],
            },
            "preprocess": {"method": side["method"], "scope": side["scope"]},
            "prices": side["prices"],
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
            "Скачать таблицу (CSV)", buf.getvalue().encode("utf-8-sig"), f"{name}.csv", "text/csv", key=f"dl_csv_{name}"
        )
    p = json.loads(
        json.dumps(
            {"страница": name, **params},
            ensure_ascii=False,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )
    c2.download_button(
        "Скачать параметры (YAML)",
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


def features_subset(ids: set[int], coarse: bool | None = None) -> dict:
    """Полигоны МО из ids. coarse=None — облегчённые, если МО больше COARSE_FROM."""
    g = geojson(len(ids) > COARSE_FROM if coarse is None else coarse)
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
        lab = (names or {}).get(int(k)) or f"K{int(k) + 1}"
        fig.add_trace(
            go.Choropleth(
                geojson=features_subset(set(int(i) for i in sel.index), coarse=len(ids) > COARSE_FROM),
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
                    name=f"K{int(k) + 1}",
                    marker=dict(size=8, color=cluster_color(int(k)), line=dict(width=1, color="white")),
                    text=m.loc[sel, "name"],
                    hovertemplate="%{text}<br>Тип " + str(k) + "<extra></extra>",
                )
            )
    fig.update_layout(title=title)
    return geo_layout(fig, ids)


FORCE_ON_DEMAND = 800  # на сетях больше — силовая раскладка строится по кнопке (на всей России ~8 с)


@st.cache_data(show_spinner="Считаю силовую раскладку…", max_entries=8)
def force_layout(edges: pd.DataFrame, nodes: tuple[int, ...], seed: int = 42) -> dict:
    """Координаты узлов spring_layout (кэш: тот же граф — мгновенно)."""
    import networkx as nx

    g = nx.Graph()
    g.add_weighted_edges_from(edges[["source", "target", "weight"]].itertuples(index=False, name=None))
    g.add_nodes_from(nodes)
    return nx.spring_layout(g, weight="weight", seed=seed, k=1.5 / np.sqrt(max(len(g), 1)))


def force_tab(edges: pd.DataFrame, labels: pd.Series | None = None, title: str = "", key: str = "force") -> None:
    """Вкладка «Силовая раскладка»: на больших сетях — по переключателю, чтобы не тормозить страницу."""
    n = len(labels) if labels is not None else len(set(edges["source"]) | set(edges["target"]))
    if n > FORCE_ON_DEMAND and not st.toggle(f"Показать раскладку ({n} узлов, займёт несколько секунд)", key=key):
        return
    st.plotly_chart(graph_force(edges, labels, title=title), width="stretch")


def graph_force(edges: pd.DataFrame, labels: pd.Series | None = None, seed: int = 42, title: str = "") -> go.Figure:
    """Силовая раскладка графа (networkx spring_layout, фиксированный seed)."""
    import networkx as nx

    g = nx.Graph()
    g.add_weighted_edges_from(edges[["source", "target", "weight"]].itertuples(index=False, name=None))
    if labels is not None:
        g.add_nodes_from(int(i) for i in labels.index)
    pos = force_layout(edges, tuple(int(i) for i in g.nodes()), seed)
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
                name="МО" if k is None else f"K{int(k) + 1}",
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
    """Каталог готовых результатов кластеризации для сети с хэшем h."""
    return DATA / "clusters" / net_hash


NAMES_FILE = DATA / "cluster_names.yaml"


def load_names() -> dict:
    """Названия кластеров, заданные аналитиками (data/cluster_names.yaml)."""
    if NAMES_FILE.exists():
        return yaml.safe_load(NAMES_FILE.read_text(encoding="utf-8")) or {}
    return {}


def save_names(key: str, names: dict) -> None:
    """Сохранить названия кластеров для разбиения key в data/cluster_names.yaml."""
    allnames = load_names()
    allnames[key] = {int(k): v for k, v in names.items() if v}
    NAMES_FILE.write_text(yaml.safe_dump(allnames, allow_unicode=True, sort_keys=True), encoding="utf-8")


def clustering_cfg() -> dict:
    """Параметры методов кластеризации (configs/clustering.yaml)."""
    return load_yaml("clustering.yaml")
