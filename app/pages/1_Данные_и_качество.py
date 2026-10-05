"""Данные и качество: покрытие, пропуски, непривязанные строки, источники."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import CATEGORICAL, SEQUENTIAL, downloads, indicators_wide, layout, mo, registry, sample_ids, sidebar

from src import prices
from src.io import PROCESSED, RAW, load_yaml

st.set_page_config(page_title="Данные и качество", layout="wide")
side = sidebar()
st.title("Данные и качество")

w = indicators_wide()
reg = registry()
ids = sample_ids(side)
y0, y1 = side["years"]
d = w[w["valid_in_year"] & w["territory_id"].isin(ids) & w["year"].between(y0, y1)]
base = reg[~reg["code"].str.contains(r"^emp_share_|^lq_|^gmp_structure_|^spend_share_", regex=True)]
codes = [c for c in base["code"] if c in d]
names = reg.set_index("code")["name"]

st.subheader("Покрытие: доля МО выборки с данными, % (показатель × год)")
cov = (d.groupby("year")[codes].apply(lambda x: x.notna().mean() * 100)).T
fig = go.Figure(
    go.Heatmap(
        z=cov.values,
        x=[str(c) for c in cov.columns],
        y=[names.get(c, c) for c in cov.index],
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        zmin=0,
        zmax=100,
        hovertemplate="%{y}<br>%{x}: %{z:.0f}%<extra></extra>",
        colorbar=dict(title="%", thickness=12),
    )
)
st.plotly_chart(layout(fig, 26 * len(codes) + 80), width="stretch")

st.subheader("Покрытие по субъектам")
c1, c2 = st.columns(2)
ind = c1.selectbox("Показатель", codes, format_func=lambda c: f"{names.get(c, c)} ({c})")
yr = c2.selectbox("Год", list(range(y1, y0 - 1, -1)))
m = mo().set_index("territory_id")
x = d[d["year"].eq(yr)].assign(region=lambda t: t["territory_id"].map(m["region_name"]))
byreg = x.groupby("region")[ind].apply(lambda s: 100 * s.notna().mean()).sort_values()
fig = go.Figure(
    go.Bar(
        x=byreg.values,
        y=byreg.index,
        orientation="h",
        marker_color=SEQUENTIAL[4],
        hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
    )
)
st.plotly_chart(layout(fig, 20 * len(byreg) + 80, xaxis_title="% МО с данными"), width="stretch")

st.subheader("Покрытие: показатель × субъект, % МО с данными за окно")
reg_cov = (
    d.assign(region=d["territory_id"].map(m["region_name"]))
    .groupby("region")[codes]
    .apply(lambda x: x.notna().mean() * 100)
).T
order = sorted(reg_cov.columns, key=lambda r: (m.loc[m["region_name"].eq(r), "federal_district"].iloc[0], r))
reg_cov = reg_cov[order]
fig = go.Figure(
    go.Heatmap(
        z=reg_cov.values,
        x=reg_cov.columns,
        y=[names.get(c, c) for c in reg_cov.index],
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        zmin=0,
        zmax=100,
        hovertemplate="%{y}<br>%{x}: %{z:.0f}%<extra></extra>",
        colorbar=dict(title="%", thickness=12),
    )
)
fig.update_xaxes(tickangle=-60, tickfont=dict(size=9))
st.plotly_chart(layout(fig, 26 * len(codes) + 220), width="stretch")
st.caption(
    f"Субъекты упорядочены по федеральным округам; окно {y0}–{y1}, только МО выборки, действующие в году. Пропуск ≠ "
    "ноль: скрытые Росстатом значения не заполняются. Полный разбор — reports/DATA_GAPS.md."
)

st.subheader("Сверка с признаками научной работы-образца")
sc_path = PROCESSED / "sample_comparison.csv"
if sc_path.exists():
    st.dataframe(pd.read_csv(sc_path), width="stretch", hide_index=True)
    st.caption(
        "Проверено по перечню 603 показателей БДПМО (tochno.st) и архиву хакатона СберИндекса. Чего нет на уровне МО — "
        "так и указано; замены — только из тех же источников. Покрытие — по всей России, 2017–2024."
    )
else:
    st.caption("Нет файла сверки — запустите scripts/build_data_gaps.py.")

st.subheader("МО с пропусками")
sel = st.multiselect(
    "Показатели для проверки",
    codes,
    default=[c for c in ["gmp_pc", "wage", "invest_pc_nobudget", "hhi_emp", "density", "old_age_share"] if c in codes],
    format_func=lambda c: names.get(c, c),
)
gaps = d[d[sel].isna().any(axis=1)][["territory_id", "year", *sel]] if sel else d.iloc[0:0]
gaps = gaps.assign(
    МО=gaps["territory_id"].map(m["name"]),
    регион=gaps["territory_id"].map(m["region_name"]),
    пропущено=gaps[sel].isna().apply(lambda r: ", ".join(r.index[r]), axis=1) if sel else "",
)
st.dataframe(gaps[["МО", "регион", "year", "пропущено"]], width="stretch", height=300)
st.caption(
    "Пропуск — это отсутствие данных у Росстата (часто — скрытые малые значения), а не ноль. "
    "Нули вместо пропусков не подставляются."
)

st.subheader("Непривязанные строки источников")
u = pd.read_csv(PROCESSED / "unmatched.csv")
st.dataframe(u, width="stretch", height=240)
st.caption(
    "Строки БДПМО, которые не удалось привязать к МО справочника СберИндекса (причина — в колонке match). "
    "Подробнее — reports/DATA.md, раздел 2."
)

st.subheader("Источники")
src = load_yaml("sources.yaml")
failed = (RAW / "FAILED.txt").read_text(encoding="utf-8").strip() if (RAW / "FAILED.txt").exists() else ""
rows = []
for g in ("sber", "rosstat", "tochno_regions"):
    for it in src[g]:
        rows.append(
            {
                "источник": it["id"],
                "файл": it["dest"],
                "найден": (RAW / it["dest"]).exists(),
                "назначение": it.get("note", ""),
            }
        )
b = src["tochno_bdmo"]
for it in b["indicators"]:
    url = b["indicator_url"].format(section=it["section"], code=it["code"])
    f = RAW / b["dest_dir"] / url.rsplit("/", 1)[-1]
    rows.append({"источник": f"БДПМО {it['code']}", "файл": f.name, "найден": f.exists(), "назначение": it["role"]})
st.dataframe(pd.DataFrame(rows), width="stretch", height=300)
st.markdown(
    "- **Не найдено в данных:** время в пути между МО (в данных СберИндекса только километры); "
    "доходы местных бюджетов после 2020 г.; занятость и оборот малого бизнеса по МО за всё окно.\n"
    f"- Неудачных загрузок: {len(failed.splitlines()) if failed else 0}."
)

st.subheader("Индексы цен и дефляторы")
pr = side["prices"]
st.markdown(
    f"Денежные показатели хранятся в текущих ценах и пересчитываются в цены **{pr['base_year']} г.**: "
    "real_t = nominal_t × L_base / L_t (`src/prices.py`). Дефлятор показателя — поле `deflator` реестра: "
    "ИПЦ — зарплата, ФОТ, доходы бюджета, розница; дефлятор ВРП — ВМП; индекс цен инвестиционной продукции — "
    "инвестиции; индексы цен производителей — отгрузка (промышленность, обрабатывающие), продукция сельского хозяйства."
)
lv = prices.levels()
ru = lv[lv["region_code"].eq(prices.RUSSIA)].pivot(index="year", columns="deflator", values="level")
gr = (ru / ru.shift(1) * 100).loc[2013:]
fig = go.Figure()
for i, c in enumerate([c for c in prices.DEFLATORS if c in gr]):
    fig.add_trace(
        go.Scatter(
            x=gr.index,
            y=gr[c],
            name=prices.DEFLATORS[c],
            mode="lines+markers",
            line=dict(color=CATEGORICAL[i], width=2),
            hovertemplate="%{x}: %{y:.1f}%<extra>" + prices.DEFLATORS[c] + "</extra>",
        )
    )
st.plotly_chart(
    layout(fig, 340, title="Индексы цен по России, в среднем за год к предыдущему году, %"), width="stretch"
)
ft = prices.coverage_table(pr["base_year"], pr["deflator_scope"])
ft = ft.pivot_table(index="дефлятор", columns="источник", values="n", fill_value=0)
st.caption(
    "Откуда взят коэффициент (число лет × субъектов"
    + (")." if pr["deflator_scope"] == "regional" else "; по России — число лет).")
    + " «Продлено темпами ИПЦ» и «общероссийский» — отметки о подстановке, а не пропуски."
)
st.dataframe(ft, width="stretch")

downloads(cov.reset_index().rename(columns={"index": "показатель"}), {"боковая_панель": side}, "data_quality")
