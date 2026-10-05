"""σ- и β-конвергенция, клубы Филлипса–Сула (этап 6).

Запуск:  python scripts/build_convergence.py

Ряды и выборки — configs/dynamics.yaml (convergence). Выборки: вся Россия, ДФО (пресет) и каждый тип
(сквозной кластер последнего года) разбиения по умолчанию. Клубы Филлипса–Сула — по ДФО: алгоритм
квадратичен по числу МО, на всей России (~2 300 МО) он слишком долгий.
Состав МО: действующие в конце окна территории (с достроенными значениями прошлых лет для
преемников объединений — derived), т.е. МО в последних неизменных границах.
Результаты: data/processed/convergence_sigma.parquet, convergence_beta.parquet, convergence_clubs.parquet;
reports/CONVERGENCE.md.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from src import clustering, convergence, network  # noqa: E402
from src.io import DATA, PROCESSED, REPORTS, load_yaml  # noqa: E402


def md(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(("" if pd.isna(v) else fmt.format(v)) if isinstance(v, float) else str(v) for v in r)
            + " |"
        )
    return "\n".join(lines)


def series(spec: dict, mo: pd.DataFrame) -> pd.DataFrame:
    """territory_id, year, value для ряда: только МО, действующие в конце окна; денежные — в ценах базового года."""
    return convergence.load_series(spec, mo.loc[mo["active"], "territory_id"])


def main() -> int:
    cfg = load_yaml("dynamics.yaml")
    cc = cfg["convergence"]
    mo = pd.read_parquet(PROCESSED / "mo.parquet")
    region = mo.set_index("territory_id")["region_code"]
    dfo = set(mo.loc[mo["is_dfo"], "territory_id"])
    # типы — сквозные кластеры последнего года разбиения по умолчанию
    cp = clustering.default_params()
    h = network.config_hash(network.merge_params(network.default_params(), cp.get("network_override")))
    part = cfg["partition"]
    tdir = DATA / "clusters" / h / f"dynamics_{part['method']}_k{part['k']}_{part.get('mode', 'per_year')}"
    th = pd.read_parquet(tdir / "through_labels.parquet")
    last = th["year"].max()
    types = th[th["year"].eq(last)].set_index("territory_id")["through"]

    sig_rows, beta_rows, club_rows = [], [], []
    rep = [
        "# Конвергенция (этап 6)",
        "",
        "Сгенерировано `scripts/build_convergence.py`. Параметры — `configs/dynamics.yaml`. Все ряды в реальном "
        "выражении; состав — МО, действующие в конце окна (в последних неизменных границах).",
        "",
        f"Типы МО — сквозные кластеры {last} г. разбиения `{part['method']}`, k = {part['k']} (`reports/DYNAMICS.md`).",
        "",
        "**Предупреждение.** При окне короче ~9–10 лет выводы о конвергенции ненадёжны: σ-тренд опирается на "
        "несколько точек, а панельная β-оценка с фиксированными эффектами на короткой панели смещена вниз "
        "(смещение Никелла).",
        "",
    ]
    for key, spec in cc["series"].items():
        d = series(spec, mo)
        y0, y1 = spec["years"]
        short = (y1 - y0 + 1) < cc["min_years_reliable"]
        samples = {"Россия": d, "ДФО": d[d["territory_id"].isin(dfo)]}
        for t in sorted(types.unique()):
            ids = set(types.index[types.eq(t)])
            samples[f"тип {t}"] = d[d["territory_id"].isin(ids)]
        rep += [f"## {spec['name']} (`{key}`), {y0}–{y1}" + (" — короткое окно, выводы ненадёжны" if short else ""), ""]
        rows = []
        for sname, sd in samples.items():
            st, tr = convergence.sigma(sd, balanced=False)
            st_b, tr_b = convergence.sigma(sd, balanced=True)
            ba = convergence.beta_absolute(sd, y0, y1)
            bp = convergence.beta_panel(sd, region)
            L = convergence._log_panel(sd)
            ps = convergence.log_t(L, cc["phillips_sul"]["trim"], cc["phillips_sul"]["hp_lambda"])
            for r in st.itertuples():
                sig_rows.append(
                    {"series": key, "sample": sname, "year": r.year, "n": r.n, "sd_log": r.sd_log, "cv": r.cv}
                )
            beta_rows.append(
                {
                    "series": key,
                    "sample": sname,
                    **{f"beta_abs_{k}": v for k, v in ba.items() if k not in ("growth", "level0")},
                    **{f"beta_panel_{k}": v for k, v in bp.items()},
                    "sigma_slope": tr["slope"],
                    "sigma_p": tr["p"],
                    "sigma_slope_balanced": tr_b["slope"],
                    "sigma_p_balanced": tr_b["p"],
                    "logt_b": ps["b"],
                    "logt_t": ps["t"],
                    "short_window": short,
                }
            )
            rows.append(
                {
                    "выборка": sname,
                    "МО (β)": ba["n"],
                    "σ: наклон": tr["slope"],
                    "σ: p": tr["p"],
                    "σ сбаланс.: наклон": tr_b["slope"],
                    "β абс.": ba.get("b"),
                    "β абс.: p": ba.get("p"),
                    "λ, %/год": 100 * ba.get("lambda", np.nan),
                    "полупериод, лет": ba.get("half_life"),
                    "β панель (FE)": bp.get("b"),
                    "β панель: p": bp.get("p"),
                    "log t: b": ps["b"],
                    "log t: t": ps["t"],
                }
            )
        rep += [
            md(pd.DataFrame(rows), "{:.4f}"),
            "",
            "σ: наклон < 0 — разброс ln y сокращается (σ-конвергенция); β < 0 — МО с низким начальным уровнем "
            "растут быстрее; log t: t < −1,65 — гипотеза общей конвергенции отвергается (Phillips–Sul).",
            "",
        ]
        # клубы — для ДФО (на всей России алгоритм квадратичен по числу МО и слишком долгий)
        Ld = convergence._log_panel(samples["ДФО"])
        cl = convergence.clubs(Ld, **cc["phillips_sul"])
        for tid, c in cl.items():
            club_rows.append({"series": key, "territory_id": tid, "club": int(c)})
        common = cl.index.intersection(types.index)
        ari = adjusted_rand_score(cl.loc[common], types.loc[common]) if len(common) else np.nan
        ct = pd.crosstab(cl.loc[common].rename("клуб"), types.loc[common].rename("тип"))
        rep += [
            f"Клубы Филлипса–Сула (ДФО): {int((cl > 0).sum())} МО в {int(cl.max()) if (cl > 0).any() else 0} клубах, "
            f"{int((cl < 0).sum())} расходящихся (−1). Согласие клубов с типами {last} г.: ARI = {ari:.2f}.",
            "",
            md(ct.reset_index(), "{:.0f}"),
            "",
        ]
        print(f"{key}: готово", flush=True)
    pd.DataFrame(sig_rows).to_parquet(PROCESSED / "convergence_sigma.parquet", index=False)
    pd.DataFrame(beta_rows).to_parquet(PROCESSED / "convergence_beta.parquet", index=False)
    pd.DataFrame(club_rows).to_parquet(PROCESSED / "convergence_clubs.parquet", index=False)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "CONVERGENCE.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
