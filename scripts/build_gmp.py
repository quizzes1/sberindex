"""Расчёт валового муниципального продукта (этап 3, раздел 6.1).

Запуск:  python scripts/build_gmp.py

Результаты:
  data/processed/gmp.parquet              territory_id, year, method, gmp, gmp_pc, gmp_pc_real, gmp_imputed_share,
                                          derived, valid_in_year
  data/processed/gmp_structure.parquet    отраслевая структура ВМП (метод 2)
  data/processed/gmp_sensitivity.parquet  ВМП на душу при разных индикаторах распределения
  data/processed/gmp_agglomerations.parquet
  reports/GMP_CHECKS.md                   проверки: сумма = ВРП, метод 1 vs 2, чувствительность, сверка с ВГП
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import gmp, panel  # noqa: E402
from src.io import PROCESSED, RAW, REPORTS, load_yaml  # noqa: E402

OUT: list[str] = []


def w(s: str = "") -> None:
    OUT.append(s)


def md(df: pd.DataFrame, fmt: str = "{:,.2f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        cells = [("" if pd.isna(v) else fmt.format(v)) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def params(cfg: dict, variant: dict | None = None) -> dict:
    p = {**cfg["sectoral"], "imputation_weight": cfg["imputation_weight"]}
    p["by_section"] = dict(p["by_section"])
    if variant:
        p["by_section"].update(variant.get("by_section", {}))
        if "imputation_weight" in variant:
            p["imputation_weight"] = variant["imputation_weight"]
    return p


def per_capita(d: pd.DataFrame, pop: pd.Series, defl: pd.DataFrame, mo: pd.DataFrame) -> pd.DataFrame:
    d = d.merge(mo[["territory_id", "region_code"]], on="territory_id", how="left", suffixes=("", "_mo"))
    d["region_code"] = d["region_code"].fillna(d.get("region_code_mo"))
    d["pop"] = d.set_index(["territory_id", "year"]).index.map(pop)
    d["gmp_pc"] = d["gmp"] * 1000 / d["pop"]
    d = d.merge(defl, on=["region_code", "year"], how="left")
    d["gmp_pc_real"] = d["gmp_pc"] / d["deflator"]
    d["gmp_imputed_share"] = d["gmp_imputed"] / d["gmp"]
    return d


def extend(d: pd.DataFrame, method: str, cfg_panel: dict) -> pd.DataFrame:
    """Достраивает ВМП преемников объединений суммой предшественников (как в панели)."""
    long = d.melt(id_vars=["territory_id", "year"], value_vars=["gmp", "gmp_imputed"], var_name="indicator")
    long = long.assign(unit="тыс. руб.", source=f"gmp_{method}", flag="")
    c = {
        "series": [{"id": "gmp", "agg": "sum"}, {"id": "gmp_imputed", "agg": "sum"}],
        "boundary": cfg_panel.get("boundary", {}),
    }
    ext = panel.extend_successors(long, c)
    ext = ext.pivot_table(index=["territory_id", "year"], columns="indicator", values="value").reset_index()
    der = panel.extend_successors(long, c).groupby(["territory_id", "year"])["derived"].max().reset_index()
    return ext.merge(der, on=["territory_id", "year"])


def vgp_table(mo: pd.DataFrame) -> pd.DataFrame:
    """ВГП городов ДФО (Росстат) с привязкой к territory_id: territory_id, year, vgp (тыс. руб.)."""
    d = pd.read_excel(RAW / "rosstat/VGP_DFO_2023.xlsx", sheet_name="1", header=None)
    years = [int(v) for v in d.iloc[3, 2:4]]
    rows, region = [], None
    dfo = mo[mo["is_dfo"] & mo["active"]]
    for _, r in d.iloc[4:].iterrows():
        name = str(r[1]).strip()
        if name == "nan":
            continue
        if pd.isna(r[2]):
            region = name
            continue
        stem = re.sub(r"^(г\.|город)\s*|\s*(м\.р\.|м\.о\.|г\.о\.)$", "", name).strip().lower()
        cand = dfo[dfo["region_name"].eq(region)]
        hit = cand[cand["name_short"].str.lower().eq(stem)]
        if len(hit) != 1:
            hit = cand[cand["name"].str.lower().str.contains(stem[:-2], regex=False)]
        tid = int(hit["territory_id"].iloc[0]) if len(hit) == 1 else None
        for j, y in enumerate(years):
            rows.append(
                {"vgp_name": name, "region_name": region, "territory_id": tid, "year": y, "vgp": float(r[2 + j]) * 1000}
            )
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_yaml("gmp.yaml")
    cfg_panel = load_yaml("panel.yaml")
    wide = pd.read_parquet(PROCESSED / "panel_wide.parquet")
    mo = pd.read_parquet(PROCESSED / "mo.parquet")
    data = gmp.GMPInputs.load(wide, mo)
    pop = wide.set_index(["territory_id", "year"])["pop"]
    valid = wide.set_index(["territory_id", "year"])["valid_in_year"]
    defl = gmp.grp_deflator(cfg["base_year"])
    p = params(cfg)
    by = list(range(cfg["basic_years"][0], cfg["basic_years"][1] + 1))
    sy = [y for y in range(cfg["sectoral_years"][0], cfg["sectoral_years"][1] + 1) if y in set(data.gva["year"])]

    print("Метод 1 и метод 2…", flush=True)
    raw = {"basic": gmp.compute(data, "basic", by), "sectoral": gmp.compute(data, "sectoral", sy, p)}

    # --- проверка: сумма по МО = ВРП (метод 1) / ВДС всего (метод 2)
    checks = []
    g = data.grp.set_index(["region_code", "year"])["grp"] * 1000
    t = data.gva[data.gva["section"].eq("TOTAL")].set_index(["region_code", "year"])["gva"]
    for meth, d in raw.items():
        s = d.groupby(["region_code", "year"])["gmp"].sum()
        ref = (g if meth == "basic" else t).reindex(s.index)
        checks.append(
            {
                "метод": meth,
                "регион-лет": len(s),
                "max |Σ ВМП / ВРП − 1|": float((s / ref - 1).abs().max()),
                "max |Σ ВМП / ВРП(сборник) − 1|": float((s / g.reindex(s.index) - 1).abs().max()),
            }
        )

    # --- итоговая таблица
    frames = []
    for meth, d in raw.items():
        e = extend(d, meth, cfg_panel)
        e = per_capita(e, pop, defl, mo).assign(method=meth)
        frames.append(e)
    res = pd.concat(frames, ignore_index=True)
    res["valid_in_year"] = res.set_index(["territory_id", "year"]).index.map(valid).fillna(False).astype(bool)
    res["derived"] = res["derived"].fillna(False).astype(bool)
    cols = [
        "territory_id",
        "year",
        "method",
        "gmp",
        "gmp_pc",
        "gmp_pc_real",
        "gmp_imputed_share",
        "derived",
        "valid_in_year",
        "region_code",
        "deflator",
    ]
    res = res[cols].sort_values(["method", "territory_id", "year"]).reset_index(drop=True)
    res.to_parquet(PROCESSED / "gmp.parquet", index=False)
    st = gmp.structure(raw["sectoral"])
    st.to_parquet(PROCESSED / "gmp_structure.parquet", index=False)

    # --- чувствительность (метод 2)
    print("Чувствительность…", flush=True)
    sens = [raw["sectoral"].assign(variant="baseline")]
    for name, var in cfg["sensitivity"].items():
        sens.append(gmp.compute(data, "sectoral", sy, params(cfg, var)).assign(variant=name))
    sens = pd.concat(sens, ignore_index=True)
    sens["pop"] = sens.set_index(["territory_id", "year"]).index.map(pop)
    sens["gmp_pc"] = sens["gmp"] * 1000 / sens["pop"]
    sens[["variant", "territory_id", "year", "gmp", "gmp_pc"]].to_parquet(
        PROCESSED / "gmp_sensitivity.parquet", index=False
    )

    # --- агломерации
    ag = []
    for name, ids in cfg["agglomerations"].items():
        x = res[res["territory_id"].isin(ids) & res["valid_in_year"]]
        for (meth, y), gg in x.groupby(["method", "year"]):
            if gg["territory_id"].nunique() == len(ids):
                popsum = pop.reindex(list(zip(gg["territory_id"], gg["year"]))).sum()
                ag.append(
                    {
                        "agglomeration": name,
                        "method": meth,
                        "year": y,
                        "gmp": gg["gmp"].sum(),
                        "pop": popsum,
                        "gmp_pc": gg["gmp"].sum() * 1000 / popsum,
                        "n_mo": len(ids),
                    }
                )
    ag = pd.DataFrame(ag)
    ag.to_parquet(PROCESSED / "gmp_agglomerations.parquet", index=False)

    # ================================================================ отчёт
    names = mo.set_index("territory_id")["name"]
    dfo_ids = set(mo.loc[mo["is_dfo"], "territory_id"])
    r2 = res[res["valid_in_year"] & res["territory_id"].isin(dfo_ids)]
    w("# Проверки расчёта ВМП")
    w()
    w(
        "Сгенерировано `scripts/build_gmp.py`. ВМП — **расчётная оценка команды**, не официальная статистика. "
        "Методика — `reports/METHODS.md`, раздел «Валовой муниципальный продукт»."
    )
    w()
    w(
        f"Метод 1 (по общему ФОТ): {by[0]}–{by[-1]}; метод 2 (отраслевой): {sy[0]}–{sy[-1]}. "
        f"Дефлятор — цепной дефлятор ВРП субъекта, базовый год {cfg['base_year']}."
    )
    w()
    w("## 1. Сумма ВМП по МО субъекта равна ВРП субъекта")
    w()
    w(md(pd.DataFrame(checks), "{:.2e}"))
    w()
    w(
        "Метод 2 распределяет ВДС по разделам; её итог совпадает с ВРП из сборника с точностью до округления "
        "Росстата (~1e-6). ВРП распределяется по МО, у которых известен ФОТ; МО без ФОТ (ЗАТО Вилючинск, "
        "Циолковский и др.) остаются без оценки."
    )
    w()
    n = r2[r2["method"].eq("sectoral")].groupby("year")["territory_id"].nunique()
    w("МО ДФО с оценкой (метод 2) по годам: " + ", ".join(f"{y}: {v}" for y, v in n.items()) + " (из 230).")
    w()

    w("## 2. Метод 1 против метода 2 (ДФО)")
    w()
    a = r2.pivot_table(index=["territory_id", "year"], columns="method", values="gmp_pc").dropna()
    rows = []
    for y, x in a.groupby(level="year"):
        rows.append(
            {
                "год": y,
                "МО": len(x),
                "корр. Пирсона (log)": np.corrcoef(np.log(x["basic"]), np.log(x["sectoral"]))[0, 1],
                "корр. Спирмена": x["basic"].corr(x["sectoral"], method="spearman"),
                "медиана метод2/метод1": float((x["sectoral"] / x["basic"]).median()),
            }
        )
    w(md(pd.DataFrame(rows), "{:.3f}"))
    w()
    top_year = max(sy)
    yy = top_year - 1 if top_year - 1 in a.index.get_level_values("year") else top_year
    x = a.xs(yy, level="year").copy()
    x["ratio"] = x["sectoral"] / x["basic"]
    stx = st[st["year"].eq(yy)].pivot_table(index="territory_id", columns="section", values="share")
    x["ведущая отрасль (метод 2)"] = (
        stx.idxmax(axis=1).reindex(x.index)
        + " ("
        + (100 * stx.max(axis=1)).round(0).astype("Int64").astype(str).reindex(x.index)
        + "%)"
    )
    x = x.reindex(x["ratio"].map(lambda v: abs(np.log(v))).sort_values(ascending=False).index).head(20)
    x.insert(0, "МО", x.index.map(names))
    x = x.rename(columns={"basic": "метод 1, руб./чел.", "sectoral": "метод 2, руб./чел.", "ratio": "метод2/метод1"})
    w(f"Топ-20 МО ДФО с наибольшим расхождением методов, {yy} г.:")
    w()
    w(md(x.reset_index(drop=True), "{:,.2f}"))
    w()

    w("## 3. Чувствительность ранжирования МО ДФО к индикаторам распределения (метод 2)")
    w()
    sp = sens[sens["territory_id"].isin(dfo_ids) & sens["year"].eq(yy)].pivot_table(
        index="territory_id", columns="variant", values="gmp_pc"
    )
    rk = sp.rank(ascending=False)
    rows = []
    for v in sp.columns:
        if v == "baseline":
            continue
        d = (rk[v] - rk["baseline"]).abs()
        j = d.idxmax()
        rows.append(
            {
                "вариант": v,
                "описание": str(cfg["sensitivity"][v]),
                "корр. Спирмена с базовым": sp["baseline"].corr(sp[v], method="spearman"),
                "медиана |сдвиг места|": float(d.median()),
                "макс. сдвиг места": float(d.max()),
                "у кого": names.get(j, j),
            }
        )
    w(md(pd.DataFrame(rows), "{:.3f}"))
    w()
    w(f"Ранги — по ВМП на душу среди МО ДФО, {yy} г. (1 — самый высокий).")
    w()

    w("## 4. Внешняя сверка: валовой городской продукт Росстата (города ДФО, 2023–2024)")
    w()
    v = vgp_table(mo)
    m = v.merge(
        res[res["valid_in_year"]]
        .pivot_table(index=["territory_id", "year"], columns="method", values="gmp")
        .reset_index(),
        on=["territory_id", "year"],
        how="left",
    )
    m["метод2/ВГП"] = m["sectoral"] / m["vgp"]
    m["метод1/ВГП"] = m["basic"] / m["vgp"]
    rows = []
    for y, x in m.dropna(subset=["sectoral"]).groupby("year"):
        rows.append(
            {
                "год": y,
                "городов": len(x),
                "Спирмен метод 2": x["vgp"].corr(x["sectoral"], method="spearman"),
                "Спирмен метод 1": x["vgp"].corr(x["basic"], method="spearman"),
                "медиана метод2/ВГП": x["метод2/ВГП"].median(),
                "медиана метод1/ВГП": x["метод1/ВГП"].median(),
                "доля городов в ±25% (м2)": (x["метод2/ВГП"].sub(1).abs() <= 0.25).mean(),
                "доля городов в ±25% (м1)": (x["метод1/ВГП"].sub(1).abs() <= 0.25).mean(),
            }
        )
    w(md(pd.DataFrame(rows), "{:.3f}"))
    w()
    last = m[m["year"].eq(m["year"].max())].copy()
    last["ВГП, млрд руб."] = last["vgp"] / 1e6
    last["метод 2, млрд руб."] = last["sectoral"] / 1e6
    last["метод 1, млрд руб."] = last["basic"] / 1e6
    w(f"По городам, {int(m['year'].max())} г.:")
    w()
    w(
        md(
            last[
                [
                    "vgp_name",
                    "region_name",
                    "ВГП, млрд руб.",
                    "метод 2, млрд руб.",
                    "метод 1, млрд руб.",
                    "метод2/ВГП",
                    "метод1/ВГП",
                ]
            ].rename(columns={"vgp_name": "город", "region_name": "регион"}),
            "{:,.2f}",
        )
    )
    w()
    nm = v[v["territory_id"].isna()]["vgp_name"].unique()
    if len(nm):
        w("Не сопоставлены со справочником: " + ", ".join(nm) + ".")
        w()

    w("## 5. Доля импутированной ВДС (`gmp_imputed_share`), ДФО, метод 2")
    w()
    s = r2[r2["method"].eq("sectoral")]
    tab = s.groupby("year")["gmp_imputed_share"].describe(percentiles=[0.5, 0.9])[["mean", "50%", "90%", "max"]]
    w(
        md(
            tab.reset_index().rename(
                columns={"year": "год", "mean": "среднее", "50%": "медиана", "90%": "90-й перцентиль"}
            ),
            "{:.3f}",
        )
    )
    w()

    w("## 6. Агломерации")
    w()
    if len(ag):
        a2 = ag[ag["method"].eq("sectoral")].pivot_table(index="agglomeration", columns="year", values="gmp_pc") / 1000
        w("ВМП на душу, тыс. руб. (метод 2):")
        w()
        w(md(a2.reset_index(), "{:,.0f}"))
    w()
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "GMP_CHECKS.md").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print(f"gmp.parquet: {len(res):,} строк; отчёт reports/GMP_CHECKS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
