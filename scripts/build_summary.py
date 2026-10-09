"""Сводные таблицы кластеров для разбиения по умолчанию.

Запуск:  python scripts/build_summary.py [--year 2024] [--periods 2017 2020 2024]

Разбиение — configs/dynamics.yaml → partition (по умолчанию pooled, «Уорд + k-means», k = 5), сеть —
configs/network.yaml (вся Россия, цены базового года). Тексты, исправленные аналитиками в интерфейсе
(data/cluster_descriptions.yaml), подставляются.

Результаты: reports/SUMMARY.md; data/processed/summary_table1.{csv,xlsx} (таблица 1);
data/processed/summary_table2_mo.csv, summary_table2_subjects_{count,pop}.csv, summary_table2.xlsx (с цветами), summary_table2_print.html (таблица 2).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import pandas as pd  # noqa: E402

from src import clustering, network, summary  # noqa: E402
from src.io import PROCESSED, REPORTS, load_yaml  # noqa: E402


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


def partition_key(h: str, method: str, k: int, mode: str, year: int) -> str:
    """Ключ разбиения для сохранённых описаний кластеров: сеть, метод, k, режим, год."""
    return f"{h}|{method}|k{k}|{mode}|{year}"


def main() -> int:
    """Точка входа: сводные таблицы кластеров для разбиения по умолчанию."""
    part = load_yaml("dynamics.yaml")["partition"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None, help="год таблицы 1 (по умолчанию — последний год окна)")
    ap.add_argument(
        "--periods", type=int, nargs="+", default=None, help="годы таблицы 2 (по умолчанию — 3 равноудалённых)"
    )
    a = ap.parse_args()
    cp = clustering.default_params()
    netp = network.merge_params(network.default_params(), cp.get("network_override"))
    h = network.config_hash(netp)
    method, k, mode = part["method"], int(part["k"]), part.get("mode", "pooled")
    labels = summary.partition(netp, method, k, mode)
    res = summary.characteristic_table(labels, netp, a.year)
    year = res["year"]
    key = partition_key(h, method, k, mode, year)
    t = summary.apply_descriptions(res["table"], key)
    pr = netp["prices"]
    title = f"Характерные признаки кластеров, {year} г. ({method}, k = {k}, {mode}; денежные — в ценах {pr['base_year']} г.)"
    cols = ["Кластер", "Число МО", "Характерные признаки", "Примеры МО"]
    t[["label", *cols, "правка"]].to_csv(PROCESSED / "summary_table1.csv", index=False, encoding="utf-8-sig")
    (PROCESSED / "summary_table1.xlsx").write_bytes(summary.table1_excel(res, t, title))

    c = summary.cfg()["characteristic"]
    th = c["thresholds"]
    raw = res["raw"].copy()
    raw.index = [clustering.code(i) for i in raw.index]
    raw.columns = [res["names"][f] for f in raw.columns]
    zt = res["z"].copy()
    zt.index = raw.index
    zt.columns = raw.columns
    out = [
        "# Сводные таблицы кластеров",
        "",
        f"Сгенерировано `scripts/build_summary.py`. Сеть `{h}` (вся Россия, признаки {list(netp['features'])}), "
        f"разбиение — **{method}, k = {k}, режим {mode}** (`configs/dynamics.yaml`). Кластеры пронумерованы K1…Kn "
        f"по убыванию медианы ВМП на душу в ценах {pr['base_year']} г. Правило описаний — `configs/summary.yaml`.",
        "",
        f"## Таблица 1. Характерные признаки кластеров, {year} г.",
        "",
        md(t[cols], "{}"),
        "",
        f"Правило: средний z-score признака по МО кластера (z — по всем МО выборки за {year} г., после пересчёта в цены "
        f"базового года и логарифма по реестру). «Максимальные» — у кластера наибольшее среднее среди кластеров и "
        f"z ≥ {th['extreme']}; «минимальные» — наименьшее и z ≤ −{th['extreme']}; «высокие» — z ≥ {th['high']}; "
        f"«низкие» — z ≤ −{th['high']}; «близкие к среднему» — |z| < {th['high']}. В описание — "
        f"{c['n_features'][0]}–{c['n_features'][1]} признака с наибольшим |z|. Примеры — крупнейшие МО по населению."
        + (" Тексты, исправленные аналитиками, подставлены." if t["правка"].ne("нет").any() else ""),
        "",
        "### Средние значения признаков (денежные — в ценах базового года)",
        "",
        md(raw.reset_index().rename(columns={"index": "кластер"}), "{:,.3f}"),
        "",
        "### Средний z-score",
        "",
        md(zt.reset_index().rename(columns={"index": "кластер"}), "{:+.2f}"),
        "",
    ]

    # ---------------- таблица 2: периоды по умолчанию — 3 равноудалённых года окна
    y0, y1 = int(labels["year"].min()), int(labels["year"].max())
    periods = a.periods or summary.default_periods(y0, y1, summary.cfg()["periods"]["n_default"])
    mt = summary.periods_table(labels, periods)
    sc = summary.subject_table(labels, periods, "count")
    sp = summary.subject_table(labels, periods, "pop")
    sm = summary.transitions_summary(mt)
    title2 = (
        f"Результаты кластеризации по периодам {', '.join(map(str, periods))} ({method}, k = {k}, {mode}; "
        f"K1…K{k} — по убыванию ВМП на душу в ценах {pr['base_year']} г.)"
    )
    csv = mt.drop(columns="territory_id").copy()
    for y in periods:
        csv[y] = csv[y].map(summary.label_text)
    csv.to_csv(PROCESSED / "summary_table2_mo.csv", index=False, encoding="utf-8-sig")
    for tag, st_ in (("count", sc), ("pop", sp)):
        st_.to_csv(PROCESSED / f"summary_table2_subjects_{tag}.csv", index=False, encoding="utf-8")
    (PROCESSED / "summary_table2.xlsx").write_bytes(summary.table2_excel(mt, sc, sp, sm, periods, title2))
    (PROCESSED / "summary_table2_print.html").write_text(
        summary.table2_html(sc, sm, periods, title2, "Доля — по числу МО субъекта."), encoding="utf-8"
    )

    def subj_md(s: pd.DataFrame) -> str:
        v = s.copy()
        for y in periods:
            v[y] = [
                f"{summary.label_text(lab)} ({sh:.0%})" if pd.notna(lab) else "—"
                for lab, sh in zip(v[y], v[f"{y} доля"])
            ]
        return md(v[["№", "Субъект", "ФО", "МО", *periods, "траектория"]], "{}")

    tr_cols = ["территория", "МО", *[c for tr in summary.TRAJECTORIES for c in (tr, f"{tr}, %")]]
    out += [
        f"## Таблица 2. Результаты кластеризации по периодам: {', '.join(map(str, periods))}",
        "",
        "Траектория МО: **стабильный** — один кластер во всех периодах; **рост** — номер кластера только уменьшается "
        "(переход к кластеру с более высоким ВМП на душу); **снижение** — только растёт; **колебание** — и то и другое; "
        "**нет данных** — МО есть в сети меньше чем в двух периодах (не хватает признаков). Таблица уровня МО "
        f"({len(mt)} строк) — `data/processed/summary_table2_mo.csv`, с цветами — `summary_table2.xlsx`, "
        "страница для печати — `summary_table2_print.html`; в интерфейсе — страница «Сводные таблицы».",
        "",
        "### Сводка переходов (число МО и доля, %)",
        "",
        md(sm[tr_cols], "{:.1f}"),
        "",
        "### Уровень субъектов: доминирующий кластер МО субъекта и доля МО в нём",
        "",
        subj_md(sc),
        "",
        "### Уровень субъектов: доминирующий кластер по населению",
        "",
        "Доля — население МО доминирующего кластера к населению МО субъекта в сети этого года.",
        "",
        subj_md(sp),
        "",
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "SUMMARY.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Готово: reports/SUMMARY.md, summary_table1.xlsx ({len(t)} кластеров, {year} г.), summary_table2.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
