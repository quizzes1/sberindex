"""Кластеризация и индексы качества на сети по умолчанию (этап 5).

Запуск:  python scripts/build_clusters.py [--quick]   (--quick: без CANUS — он медленный)

Для каждого года сети, каждого метода и k из k_range: метки и индексы SW, CH, DBI, S_Dbw (по
нормированным признакам сети) и AVI, AVU, ANUI, MQ (по весам рёбер разреженной сети).
Результаты: data/clusters/{hash}/labels.parquet, icvi.parquet, params.json; reports/CLUSTERS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from src import clustering, icvi, network  # noqa: E402
from src.io import DATA, REPORTS  # noqa: E402

CLUSTERS = DATA / "clusters"


def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(("" if pd.isna(v) else fmt.format(v)) if isinstance(v, float) else str(v) for v in r)
            + " |"
        )
    return "\n".join(lines)


def consensus_k(tab: pd.DataFrame) -> pd.Series:
    """Средний ранг k по всем индексам (с учётом направления «лучше»); меньше — лучше."""
    ranks = []
    for ind, sign in icvi.BETTER.items():
        if ind in tab and ind != "DBI":
            ranks.append(tab[ind].rank(ascending=sign < 0))
    return pd.concat(ranks, axis=1).mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cp = clustering.default_params()
    netp = network.merge_params(network.default_params(), cp.get("network_override"))
    h = network.config_hash(netp)
    nets = network.build(netp)
    methods = {m: v for m, v in cp["methods"].items() if not (a.quick and m == "canus")}
    skipped = {}
    for m in list(methods):
        ok, why = clustering.available(m)
        if not ok:
            skipped[m] = why
            methods.pop(m)
    ks = list(range(cp["k_range"][0], cp["k_range"][1] + 1))
    labels, scores = [], []
    for y, net in nets.items():
        X = net.X.to_numpy()
        Wsp = net.W * net.A
        for m, mp in methods.items():
            t = time.time()
            for k in ks:
                lab = clustering.fit(X, Wsp, {**mp, "method": m, "k": k, "seed": cp["seed"]})
                labels.append(pd.DataFrame({"year": y, "method": m, "k": k, "territory_id": net.ids, "label": lab}))
                scores.append({"year": y, "method": m, "k": k, **icvi.compute_all(X, Wsp, lab)})
            print(f"{y} {m}: {time.time() - t:.1f} с", flush=True)
    out = CLUSTERS / h
    out.mkdir(parents=True, exist_ok=True)
    L = pd.concat(labels, ignore_index=True)
    S = pd.DataFrame(scores)
    L.to_parquet(out / "labels.parquet", index=False)
    S.to_parquet(out / "icvi.parquet", index=False)
    (out / "params.json").write_text(
        json.dumps(
            {"network": netp, "clustering": cp, "network_hash": h}, ensure_ascii=False, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------- отчёт
    rep = [
        "# Кластеризация и индексы качества (этап 5)",
        "",
        f"Сгенерировано `scripts/build_clusters.py`. Сеть `{h}` (параметры `configs/network.yaml`: ДФО, "
        f"{min(nets)}–{max(nets)}, признаки {list(netp['features'])}). Методы и k — `configs/clustering.yaml`.",
        "",
        "Индексы: SW, CH, S_Dbw (и DBI) — в пространстве нормированных признаков; AVI, AVU, ANUI, MQ — по весам "
        "рёбер разреженной сети. Направление «лучше»: "
        + ", ".join(f"{k} {'↑' if v > 0 else '↓'}" for k, v in icvi.BETTER.items())
        + ".",
        "",
    ]
    if skipped:
        rep += ["Не запущены: " + "; ".join(f"{m} — {w}" for m, w in skipped.items()), ""]
    mean = S.groupby(["method", "k"]).mean(numeric_only=True).drop(columns="year")
    rep += ["## 1. Индексы для k = 2…10 (среднее по годам)", ""]
    best = []
    for m in methods:
        t = mean.loc[m]
        rk = consensus_k(t)
        kbest = int(rk.idxmin())
        best.append(
            {
                "метод": m,
                "k по SW": int(t["SW"].idxmax()),
                "k по CH": int(t["CH"].idxmax()),
                "k по S_Dbw": int(t["S_Dbw"].idxmin()),
                "k по AVI": int(t["AVI"].idxmax()),
                "k по AVU": int(t["AVU"].idxmin()),
                "k по MQ": int(t["MQ"].idxmax()),
                "согласованное k (средний ранг)": kbest,
            }
        )
        tt = t.reset_index()[["k", "K", "SW", "CH", "DBI", "S_Dbw", "AVI", "AVU", "ANUI", "MQ"]]
        tt["средний ранг"] = rk.values
        rep += [f"### {m}", "", md(tt), ""]
    B = pd.DataFrame(best)
    rep += [
        "## 2. Выбор числа кластеров",
        "",
        "«Согласованное k» — k с наименьшим средним рангом по семи индексам (SW, CH, S_Dbw, AVI, AVU, ANUI, MQ). "
        "Индексы часто расходятся: SW, CH и AVI тянут к малым k, MQ — к большим, а S_Dbw на этих данных монотонно убывает с ростом k и выбирает верхнюю границу диапазона (k = 9–10) — поэтому сам по себе он для выбора k не годится; окончательный "
        "выбор k — за аналитиком (страница «Кластеры» интерфейса).",
        "",
        md(B, "{}"),
        "",
    ]
    k0 = int(B["согласованное k (средний ранг)"].mode().iloc[0])
    comp = mean.xs(k0, level="k").reset_index()
    rep += [
        f"## 3. Сравнение методов при k = {k0} (среднее по годам)",
        "",
        md(comp[["method", "K", "SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ"]]),
        "",
    ]
    # согласие разбиений между методами (ARI), k0, среднее по годам
    ms = list(methods)
    ari = pd.DataFrame(np.nan, index=ms, columns=ms)
    for i in ms:
        for j in ms:
            vals = []
            for y in nets:
                a_ = L[(L.year == y) & (L.method == i) & (L.k == k0)].set_index("territory_id")["label"]
                b_ = L[(L.year == y) & (L.method == j) & (L.k == k0)].set_index("territory_id")["label"]
                vals.append(adjusted_rand_score(a_, b_.reindex(a_.index)))
            ari.loc[i, j] = float(np.mean(vals))
    rep += [
        f"### Согласие методов: скорректированный индекс Рэнда (ARI) между разбиениями, k = {k0}",
        "",
        md(ari.reset_index().rename(columns={"index": ""}), "{:.2f}"),
        "",
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "CLUSTERS.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(f"Готово: data/clusters/{h}; согласованное k = {k0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
