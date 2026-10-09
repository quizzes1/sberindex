"""Расчёт показателей из реестра.

Запуск:  python scripts/build_indicators.py

Результаты:
  data/processed/indicators_wide.parquet   territory_id, year, мета (valid_in_year, derived, region_code, is_dfo,
                                           boundary_change, gmp_method_used) + все показатели
  data/processed/indicators.parquet        long: territory_id, year, indicator, value
  data/processed/indicator_registry.parquet  раскрытый реестр (одна строка на показатель)
  data/processed/price_levels.parquet      уровни цен (src/prices.py): ИПЦ, дефлятор ВРП, инвест., ИЦП
  data/processed/price_basket.parquet      стоимость фиксированного набора субъекта / Россия
  data/processed/indicators_corr_{ru,dfo}.csv     корреляции Спирмена: вся Россия / ДФО
  data/processed/indicators_variance_{ru,dfo}.csv разброс показателей: вся Россия / ДФО
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
from src import prices  # noqa: E402
from src.io import PROCESSED, REPORTS  # noqa: E402
from src.normalize import prepare  # noqa: E402

CORE_YEARS = (2017, 2024)


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    """Таблица pandas → таблица Markdown для отчёта."""
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
    """Точка входа: расчёт показателей из реестра."""
    prices.save()
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

    # ---------------- покрытие, разброс, корреляции (срез действующих МО, 2017–2024):
    # основная выборка — вся Россия; ДФО — пресет (файлы *_dfo.csv для интерфейса)
    feats = [c for c in num if reg.set_index("code").get("block", pd.Series()).get(c) != "quality"]
    logc = reg.loc[reg["log"].fillna(False).astype(bool), "code"].tolist()
    stats = {}
    for tag, mask in (("ru", wide["valid_in_year"]), ("dfo", wide["valid_in_year"] & wide["is_dfo"])):
        # денежные — в ценах базового года (параметры по умолчанию), как в сети и кластерах
        pp = prices.price_params()
        d = prices.to_real(wide[mask], num, pp["base_year"], pp["deflator_scope"], pp["spatial_price_adjustment"])
        core = d[d.index.get_level_values("year").to_series().between(*CORE_YEARS).values]
        cov = d[num].notna().groupby(level="year").mean().T * 100
        usable = [c for c in feats if core[c].notna().mean() > 0.5]
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
        var.to_csv(PROCESSED / f"indicators_variance_{tag}.csv", encoding="utf-8")
        corr = core[usable].corr(method="spearman", min_periods=100)
        corr.to_csv(PROCESSED / f"indicators_corr_{tag}.csv", encoding="utf-8")
        stats[tag] = (cov, usable, var, corr)
    cov, usable, var, corr = stats["ru"]

    names = reg.set_index("code")["name"]
    out = [
        "# Показатели",
        "",
        "Сгенерировано `scripts/build_indicators.py`. Реестр — `configs/indicators.yaml`, "
        "формулы — `src/indicators.py`, раскрытый реестр для интерфейса — `data/processed/indicator_registry.parquet`.",
        "",
        f"Всего колонок-показателей: {len(num)} (включая отраслевые). Денежные хранятся в текущих ценах; разброс и "
        f"корреляции ниже — в ценах {prices.price_params()['base_year']} г. (src/prices.py, дефлятор — поле deflator реестра).",
        "",
        "## 1. Покрытие действующих МО России по годам, %",
        "",
    ]
    base = [c for c in num if not c.startswith(("emp_share_", "lq_", "gmp_structure_", "spend_share_"))]
    t = cov.loc[base].round(0)
    t.insert(0, "показатель", [names.get(c, c) for c in t.index])
    out += [md(t.reset_index().rename(columns={"index": "код"}), "{:.0f}"), ""]
    out += [
        "## 2. Пары сильно связанных показателей (|ρ Спирмена| ≥ 0,8), Россия, 2017–2024",
        "",
        "Кандидаты на исключение дублей: в сети и кластеризации из такой пары обычно достаточно одного. "
        "Не показаны пары одной величины в разных шкалах (emp_share_k / lq_k / lq_ru_k — внутри года "
        "это одно и то же с точностью до множителя).",
        "",
    ]

    def family(c: str) -> str:
        """emp_share_H, lq_H, lq_ru_H → «H»: одна величина в разных шкалах."""
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
        "## 3. Разброс показателей по России (2017–2024)",
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
