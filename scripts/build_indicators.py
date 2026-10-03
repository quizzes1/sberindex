"""Расчёт показателей из реестра (этап 3, раздел 6.2).

Запуск:  python scripts/build_indicators.py

Результаты:
  data/processed/indicators_wide.parquet   territory_id, year, мета (valid_in_year, derived, region_code, is_dfo,
                                           boundary_change, gmp_method_used) + все показатели
  data/processed/indicators.parquet        long: territory_id, year, indicator, value
  data/processed/indicator_registry.parquet  раскрытый реестр (одна строка на показатель)
  data/processed/indicators_corr_dfo.csv     корреляции Спирмена по ДФО
  data/processed/indicators_variance_dfo.csv разброс показателей по ДФО
  reports/INDICATORS.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import indicators as ind  # noqa: E402
from src.io import PROCESSED, REPORTS  # noqa: E402
from src.normalize import prepare  # noqa: E402

CORE_YEARS = (2017, 2024)


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(("" if pd.isna(v) else fmt.format(v)) if isinstance(v, float) else str(v) for v in r)
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    ctx = ind.Context.load()
    vals = ind.compute_all(ctx)
    meta = ctx.wide[["valid_in_year", "derived"]].copy()
    mo = ctx.mo.set_index("territory_id")
    tid = meta.index.get_level_values("territory_id")
    meta["region_code"] = tid.map(mo["region_code"])
    meta["federal_district"] = tid.map(mo["federal_district"])
    meta["is_dfo"] = tid.map(mo["is_dfo"]).astype(bool)
    meta["boundary_change"] = tid.map(mo["boundary_change"]).astype(bool)
    wide = meta.join(vals)
    y0, y1 = ctx.params["window"]
    wide = wide[wide.index.get_level_values("year").to_series().between(y0, y1).values]
    wide.reset_index().to_parquet(PROCESSED / "indicators_wide.parquet", index=False)
    num = [c for c in vals.columns if c != "gmp_method_used"]
    long = wide[num].stack().rename("value").reset_index().rename(columns={"level_2": "indicator"})
    long = long.dropna(subset=["value"])  # pandas 3: stack() сохраняет пропуски
    long.to_parquet(PROCESSED / "indicators.parquet", index=False)
    reg = ind.expand_registry()
    reg = reg[reg["code"].isin(num)]
    reg.drop(columns=[c for c in reg.columns if reg[c].map(lambda v: isinstance(v, dict | list)).any()]).to_parquet(
        PROCESSED / "indicator_registry.parquet", index=False
    )

    # ---------------- ДФО: покрытие, разброс, корреляции (срез действующих МО, 2017–2024)
    d = wide[wide["valid_in_year"] & wide["is_dfo"]]
    core = d[d.index.get_level_values("year").to_series().between(*CORE_YEARS).values]
    cov = d[num].notna().groupby(level="year").mean().T * 100
    feats = [c for c in num if reg.set_index("code").get("block", pd.Series()).get(c) != "quality"]
    usable = [c for c in feats if core[c].notna().mean() > 0.5]
    logc = reg.loc[reg["log"].fillna(False).astype(bool), "code"].tolist()
    z = prepare(core, usable, log_columns=logc, method="minmax", scope="panel")
    var = pd.DataFrame(
        {
            "среднее": core[usable].mean(),
            "медиана": core[usable].median(),
            "коэф. вариации": core[usable].std() / core[usable].mean().abs(),
            "дисперсия после min-max (с логарифмом по реестру)": z.var(),
            "покрытие 2017–2024, %": 100 * core[usable].notna().mean(),
        }
    )
    var.index.name = "показатель"
    var.to_csv(PROCESSED / "indicators_variance_dfo.csv", encoding="utf-8")
    corr = core[usable].corr(method="spearman", min_periods=100)
    corr.to_csv(PROCESSED / "indicators_corr_dfo.csv", encoding="utf-8")

    names = reg.set_index("code")["name"]
    out = [
        "# Показатели (этап 3)",
        "",
        "Сгенерировано `scripts/build_indicators.py`. Реестр — `configs/indicators.yaml`, "
        "формулы — `src/indicators.py`, раскрытый реестр для интерфейса — `data/processed/indicator_registry.parquet`.",
        "",
        f"Всего колонок-показателей: {len(num)} (включая отраслевые и версии «_cpi» в ценах {ctx.params['base_year']} г.).",
        "",
        "## 1. Покрытие действующих МО ДФО по годам, %",
        "",
    ]
    base = [
        c
        for c in num
        if not c.startswith(("emp_share_", "lq_", "gmp_structure_", "spend_share_")) and not c.endswith("_cpi")
    ]
    t = cov.loc[base].round(0)
    t.insert(0, "показатель", [names.get(c, c) for c in t.index])
    out += [md(t.reset_index().rename(columns={"index": "код"}), "{:.0f}"), ""]
    out += [
        "## 2. Пары сильно связанных показателей (|ρ Спирмена| ≥ 0,8), ДФО, 2017–2024",
        "",
        "Кандидаты на исключение дублей: в сети и кластеризации из такой пары обычно достаточно одного. "
        "Не показаны пары одной величины в разных шкалах (emp_share_k / lq_k / lq_ru_k — внутри года "
        "это одно и то же с точностью до множителя; x / x_cpi / x_real).",
        "",
    ]

    def family(c: str) -> str:
        """emp_share_H, lq_H, lq_ru_H → «H»; wage_cpi → wage: одна величина в разных шкалах."""
        c = c.removesuffix("_cpi").removesuffix("_real")
        for pre in ("emp_share_", "lq_ru_", "lq_"):
            if c.startswith(pre):
                return "sector_" + c[len(pre) :]
        return c

    pairs = []
    for i, a in enumerate(usable):
        for b in usable[i + 1 :]:
            r = corr.at[a, b]
            if np.isfinite(r) and abs(r) >= 0.8 and family(a) != family(b):
                pairs.append({"показатель 1": a, "показатель 2": b, "ρ": r})
    pairs = pd.DataFrame(pairs).sort_values("ρ", key=abs, ascending=False) if pairs else pd.DataFrame()
    out += [md(pairs.head(60), "{:.3f}") if len(pairs) else "Нет.", ""]
    key = [
        "gmp_pc",
        "wage",
        "payroll_pc",
        "shipped_pc",
        "invest_pc",
        "invest_pc_nobudget",
        "retail_pc",
        "budget_own_share",
        "density",
        "pop_growth",
        "hhi_emp",
        "old_age_share",
    ]
    key = [k for k in key if k in corr]
    out += [
        "### Ключевые экономические показатели: матрица ρ Спирмена",
        "",
        md(corr.loc[key, key].round(2).reset_index().rename(columns={"index": ""}), "{:.2f}"),
        "",
    ]
    out += [
        "## 3. Разброс показателей по ДФО (2017–2024)",
        "",
        "Малая дисперсия после нормировки — показатель слабо различает МО (например, коэффициенты локализации "
        "редких отраслей).",
        "",
    ]
    vv = var.sort_values("дисперсия после min-max (с логарифмом по реестру)").reset_index()
    out += [md(vv, "{:.3f}"), ""]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "INDICATORS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"indicators_wide: {wide.shape}; показателей {len(num)}; пар с |ρ|≥0.8: {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
