"""Сети по умолчанию (этап 4): ДФО и вся Россия, каждый год 2017–2024.

Запуск:  python scripts/build_networks.py

Параметры — configs/network.yaml. Результат: data/networks/{hash}/edges_{год}.parquet
(source, target, weight, distance), nodes_{год}.parquet, params.json, stats.json;
сводка — reports/NETWORKS.md, список сборок — data/networks/index.csv.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import pandas as pd  # noqa: E402

from src import network  # noqa: E402
from src.io import NETWORKS, REPORTS  # noqa: E402

VARIANTS = {
    "ДФО, экономика (по умолчанию)": {},
    "ДФО, экономика 70% + дороги 30%": {"geo": {"alpha": 0.7}},
    "Россия, экономика": {"sample": {"federal_districts": ["ЦФО", "СЗФО", "ЮФО", "СКФО", "ПФО", "УФО", "СФО", "ДФО"]}},
}


def main() -> int:
    base = network.default_params()
    rows, out = (
        [],
        [
            "# Сети (этап 4)",
            "",
            "Сгенерировано `scripts/build_networks.py`. Параметры по умолчанию — "
            "`configs/network.yaml`; любой набор параметров воспроизводится по хэшу "
            "(`data/networks/{hash}/params.json`).",
            "",
        ],
    )
    for name, ov in VARIANTS.items():
        p = network.merge_params(base, ov)
        nets = network.build(p)
        h = network.save(nets, p)
        st = pd.DataFrame([n.stats() for n in nets.values()])
        st["dropped_missing"] = st["dropped_missing"].map(len)
        rows.append({"сборка": name, "hash": h})
        out += [
            f"## {name} — `{h}`",
            "",
            f"Признаки и веса: {p['features']}; метрика {p['metric']}; "
            f"α = {p['geo']['alpha']}; вес ребра {p['edge_weight']}; прореживание {p['sparsify']['method']}, "
            f"k = {p['sparsify']['k']}.",
            "",
        ]
        cols = [
            "year",
            "nodes",
            "edges",
            "components",
            "largest_component",
            "isolated",
            "degree_min",
            "degree_mean",
            "degree_max",
            "dropped_missing",
        ]
        t = st[cols].rename(
            columns={
                "year": "год",
                "nodes": "узлов",
                "edges": "рёбер",
                "components": "компонент",
                "largest_component": "крупнейшая",
                "isolated": "изолированных",
                "degree_min": "степень min",
                "degree_mean": "степень сред.",
                "degree_max": "степень max",
                "dropped_missing": "исключено (пропуск признака)",
            }
        )
        lines = ["| " + " | ".join(t.columns) + " |", "|" + "|".join("---" for _ in t.columns) + "|"]
        for r in t.itertuples(index=False):
            lines.append("| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in r) + " |")
        out += lines + [""]
        geo_keys = [k for k in nets[max(nets)].geo_report]
        for k in geo_keys:
            out += [f"{k} ({max(nets)}): {len(nets[max(nets)].geo_report[k])} МО.", ""]
        print(f"{name}: {h}; рёбер {st['edges'].sum():,}; компонент max {st['components'].max()}")
    pd.DataFrame(rows).to_csv(NETWORKS / "index.csv", index=False, encoding="utf-8")
    out += [
        "## Методическое замечание",
        "",
        "Если ребро строится по **одному** показателю, расстояние |x_i − x_j| упорядочивает МО на прямой, и "
        "кластеризация такой сети сводится к нарезке МО на интервалы значений этого показателя. Поэтому в "
        "сравнение методов включаются комбинации нескольких показателей.",
        "",
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "NETWORKS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
