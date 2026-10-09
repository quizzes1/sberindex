"""Полнота данных по всей России и сверка с признаками научной работы-образца.

Запуск:  python scripts/build_data_gaps.py

Результаты: reports/DATA_GAPS.md, data/processed/coverage_indicator_year.csv,
data/processed/coverage_indicator_region.csv, data/processed/sample_comparison.csv (сверка с образцом)
— для страницы «Данные и качество».
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import pandas as pd  # noqa: E402

from src.io import PROCESSED, REPORTS  # noqa: E402

YEARS = (2013, 2024)
KEY = [
    "pop",
    "gmp_pc",
    "wage",
    "payroll_pc",
    "invest_pc",
    "invest_pc_nobudget",
    "shipped_pc",
    "density",
    "budget_own_share",
    "retail_pc",
    "hhi_emp",
    "pop_growth",
]

# Признаки образца → наши показатели. Кодов, которых нет, нет и в источниках (проверено по перечню
# 603 показателей БДПМО и архиву хакатона СберИндекса).
SAMPLE = [
    ("Численность населения", "да", "8112027 (+8112014, 8112013)", ["pop"], "—"),
    (
        "Объём инвестиций",
        "да",
        "8109001 (без МСП), 8109003 (без бюджетных средств, на душу)",
        ["invest_pc", "invest_pc_nobudget"],
        "—",
    ),
    (
        "ВРП и ВРП на душу",
        "нет (только по субъектам)",
        "ВРП и ВДС субъектов — Росстат",
        ["gmp_pc"],
        "ВМП и ВМП на душу — расчётная оценка команды (уже есть)",
    ),
    ("Площадь территории", "да", "8006001", ["density"], "—"),
    ("Плотность населения", "да", "расчёт: P / площадь", ["density"], "—"),
    (
        "Доходы населения",
        "почти нет",
        "8019014 «Объём социальных выплат и налогооблагаемых доходов» — только 2011–2019",
        ["wage", "payroll_pc"],
        "средняя зарплата и ФОТ на жителя (уже есть)",
    ),
    (
        "Уровень безработицы",
        "нет",
        "в БДПМО по МО не публикуется",
        ["employment_ratio"],
        "работники крупных и средних организаций на жителя трудоспособного возраста (новый, грубая замена)",
    ),
    (
        "Больничные койки на 10 тыс.",
        "нет в окне",
        "8018103 — только 2008–2013",
        ["clinics_per_10k"],
        "лечебно-профилактические организации на 10 тыс. жителей (уже есть)",
    ),
    (
        "Затраты на научные исследования",
        "нет",
        "по МО не публикуется",
        ["emp_share_M"],
        "доля занятых в разделе M ОКВЭД2 «Научная и профессиональная деятельность» (уже есть)",
    ),
    ("Обучающиеся в вузах", "нет", "по МО не публикуется", [], "нет замены"),
    (
        "Инновационные технологии, инновационная активность",
        "нет",
        "по МО не публикуется",
        ["emp_share_J"],
        "отчасти — доля занятых в разделе J «Информация и связь» (уже есть)",
    ),
    ("Веб-представительства организаций", "нет", "по МО не публикуется", [], "нет замены"),
    ("Сельскохозяйственное производство", "да", "8007010 (до 2023 г.)", ["agri_output_pc"], "на жителя (новый)"),
    (
        "Оборот сельскохозяйственной продукции",
        "только в натуре",
        "8097016 — реализация в тоннах по продуктам",
        [],
        "используем продукцию с/х в рублях",
    ),
    (
        "Предприятия обрабатывающей промышленности",
        "частично",
        "8942010 (раздел C, без МСП), 8401011 (отгрузка, раздел C)",
        ["manuf_orgs_per_10k", "manuf_shipped_pc"],
        "на 10 тыс. жителей и отгрузка на жителя (новые)",
    ),
    ("Объём торговли", "да", "8401003, 8201003 (оборот розницы без МСП)", ["retail_pc"], "—"),
    (
        "Транспортно-логистическая отрасль",
        "частично",
        "8423005 (занятые, раздел H); отгрузки по H нет",
        ["emp_share_H"],
        "доля занятых в разделе H «Транспортировка и хранение» (уже есть)",
    ),
]


def md(df: pd.DataFrame, fmt: str = "{:.0f}") -> str:
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
    """Точка входа: полнота данных по всей России и сверка с признаками научной работы-образца."""
    w = pd.read_parquet(PROCESSED / "indicators_wide.parquet")
    reg = pd.read_parquet(PROCESSED / "indicator_registry.parquet").set_index("code")
    mo = pd.read_parquet(PROCESSED / "mo.parquet").set_index("territory_id")
    v = w[w["valid_in_year"] & w["year"].between(*YEARS)].copy()
    v["region"] = v["territory_id"].map(mo["region_name"])
    base = [c for c in reg.index if c in v and not c.startswith(("lq_", "gmp_structure_", "spend_share_"))]
    cov_y = (v.groupby("year")[base].apply(lambda x: x.notna().mean() * 100)).T
    cov_y.index.name = "indicator"
    cov_y.to_csv(PROCESSED / "coverage_indicator_year.csv", encoding="utf-8")
    core = v[v["year"].between(2017, 2024)]
    cov_r = (core.groupby("region")[base].apply(lambda x: x.notna().mean() * 100)).T
    cov_r.index.name = "indicator"
    cov_r.to_csv(PROCESSED / "coverage_indicator_region.csv", encoding="utf-8")
    cov_fd = (core.groupby("federal_district")[base].apply(lambda x: x.notna().mean() * 100)).T

    out = [
        "# Полнота данных по всей России и сверка с образцом",
        "",
        "Сгенерировано `scripts/build_data_gaps.py`. Покрытие — доля действующих МО России с данными (пропуск ≠ ноль: "
        "скрытые Росстатом значения не заполняются).",
        "",
        "## 1. Признаки научной работы-образца и что есть у нас на уровне МО",
        "",
    ]
    rows = []
    for name, have, src, codes, repl in SAMPLE:
        cv = [f"{c}: {core[c].notna().mean() * 100:.0f}%" for c in codes if c in core]
        yrs = []
        for c in codes:
            if c in v:
                y = cov_y.loc[c] if c in cov_y.index else None
                if y is not None and (y > 50).any():
                    yrs.append(f"{c}: {y[y > 50].index.min()}–{y[y > 50].index.max()}")
        rows.append(
            {
                "Признак из образца": name,
                "Есть у нас на уровне МО?": have,
                "Источник и код": src,
                "Годы (>50% МО) и покрытие МО России 2017–2024": "; ".join(yrs)
                + (" · " if yrs and cv else "")
                + "; ".join(cv),
                "Замена, если нет": repl,
            }
        )
    pd.DataFrame(rows).to_csv(PROCESSED / "sample_comparison.csv", index=False, encoding="utf-8")
    out += [md(pd.DataFrame(rows), "{}"), ""]
    out += [
        "Проверено по перечню 603 показателей БДПМО (tochno.st, v20250918) и архиву хакатона СберИндекса. "
        "Безработицы, вузов, науки и инноваций, веб-представительств по МО в этих источниках нет; "
        "больничные койки по МО публиковались только в 2008–2013 гг.",
        "",
    ]

    out += ["## 2. Покрытие «показатель × год», % МО России", ""]
    t = cov_y.round(0)
    t.insert(0, "показатель", [reg.at[c, "name"][:60] if c in reg.index else c for c in t.index])
    out += [md(t.reset_index(), "{:.0f}"), ""]

    out += ["## 3. Покрытие «показатель × федеральный округ», % МО, 2017–2024", ""]
    t = cov_fd.round(0)
    t.insert(0, "показатель", [reg.at[c, "name"][:60] if c in reg.index else c for c in t.index])
    out += [
        md(t.reset_index(), "{:.0f}"),
        "",
        "Покрытие «показатель × субъект» — `data/processed/coverage_indicator_region.csv` и тепловая карта на странице "
        "«Данные и качество».",
        "",
    ]

    out += [
        "## 4. Субъекты и годы, где ключевые показатели массово не опубликованы",
        "",
        "Показаны случаи, когда по России в этом году показатель есть у ≥ 80% МО, а у субъекта — меньше чем у 50% МО.",
        "",
    ]
    key = [c for c in KEY if c in v]
    g = v.groupby(["region", "year"])[key].apply(lambda x: x.notna().mean() * 100)
    bad = g.stack().rename("cov").reset_index()
    bad = bad[bad["cov"] < 50]
    # только «провалы» субъекта: по России в этом году показатель есть (≥ 80% МО), а у субъекта — нет;
    # структурные границы рядов (ОКВЭД2 с 2017 г., первый год темпа роста, конец ряда) сюда не попадают
    nat = cov_y.stack().rename("nat").reset_index().rename(columns={"indicator": "level_2"})
    bad = bad.merge(nat, on=["level_2", "year"], how="left")
    bad = bad[bad["nat"] >= 80]
    summ = (
        bad.groupby(["region", "level_2"])["year"]
        .apply(lambda s: ", ".join(str(int(y)) for y in sorted(s)))
        .unstack()
        .fillna("")
    )
    summ.columns = [reg.at[c, "name"][:30] if c in reg.index else c for c in summ.columns]
    out += [md(summ.reset_index().rename(columns={"region": "субъект"}), "{}") if len(summ) else "Нет.", ""]

    out += ["## 5. Окно лет: ключевые показатели покрывают ≥ 80% МО России", ""]
    k2 = [c for c in key if c != "invest_pc"]
    win = cov_y.loc[k2].T.round(0)
    win["все ≥ 80%"] = (win >= 80).all(axis=1).map({True: "да", False: "нет"})
    out += [md(win.reset_index(), "{}"), ""]
    ok = win.index[win["все ≥ 80%"].eq("да")].tolist()
    out += [
        f"Годы, где все ключевые показатели (без invest_pc — ряд 8109001 кончается 2023 г.) покрывают ≥ 80% МО России: "
        f"{', '.join(map(str, ok)) or 'нет'}.",
        "",
    ]
    k3 = [c for c in k2 if c != "budget_own_share"]
    ok2 = win.index[(win[k3] >= 80).all(axis=1)].tolist()
    out += [
        f"Без доли собственных доходов бюджета (в 2024 г. — {win.at[2024, 'budget_own_share']:.0f}% МО; в признаки сети по "
        f"умолчанию не входит): {', '.join(map(str, ok2))}. **Предложение:** окно сети и кластеров — "
        f"{min(ok2)}–{max(ok2)}; в 2024 г. часть субъектов ещё не опубликована целиком (см. раздел 4).",
        "",
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "DATA_GAPS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("reports/DATA_GAPS.md; окно ≥80%:", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
