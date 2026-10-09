"""Кластеризация и индексы качества на сети по умолчанию.

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


def consensus_k(tab: pd.DataFrame) -> pd.Series:
    """Средний ранг k по всем индексам (с учётом направления «лучше»); меньше — лучше."""
    ranks = []
    for ind, sign in icvi.BETTER.items():
        if ind in tab and ind != "DBI":
            ranks.append(tab[ind].rank(ascending=sign < 0))
    return pd.concat(ranks, axis=1).mean(axis=1)


def add_wcss(S: pd.DataFrame, L: pd.DataFrame, nets: dict) -> pd.DataFrame:
    """WCSS (для метода локтя) по сохранённым меткам — если сетка посчитана до появления WCSS."""
    rows = []
    for (y, m, k), g in L.groupby(["year", "method", "k"]):
        X = nets[y].X.reindex(g["territory_id"].to_numpy()).to_numpy()
        rows.append({"year": y, "method": m, "k": k, "WCSS": icvi.wcss(X, g["label"].to_numpy())})
    return S.drop(columns="WCSS", errors="ignore").merge(pd.DataFrame(rows), on=["year", "method", "k"], how="left")


def main() -> int:
    """Точка входа: кластеризация и индексы качества на сети по умолчанию."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--only",
        nargs="+",
        help="досчитать только эти методы и влить в готовую сетку data/clusters/{hash} (остальные методы не трогать)",
    )
    ap.add_argument("--skip", nargs="+", default=[], help="не считать эти методы (сетка без них, не слияние)")
    ap.add_argument("--report-only", action="store_true", help="только пересобрать отчёт по готовой сетке")
    a = ap.parse_args()
    cp = clustering.default_params()
    netp = network.merge_params(network.default_params(), cp.get("network_override"))
    h = network.config_hash(netp)
    nets = network.build(netp)
    methods = {m: v for m, v in cp["methods"].items() if not (a.quick and m == "canus")}
    if a.only:
        methods = {m: v for m, v in methods.items() if m in a.only}
    methods = {m: v for m, v in methods.items() if m not in a.skip}
    skipped = {}
    for m in list(methods):
        ok, why = clustering.available(m)
        if not ok:
            skipped[m] = why
            methods.pop(m)
    ks = list(range(cp["k_range"][0], cp["k_range"][1] + 1))
    labels, scores = [], []
    for y, net in ({} if a.report_only else nets).items():
        X = net.X.to_numpy()
        Wsp = net.W * net.A
        for m, mp in methods.items():
            t = time.time()
            # медленные методы (CANUS) в автоматической сетке — на фиксированной подвыборке МО года
            # (одна и та же для всех k); на полной выборке они запускаются вручную в интерфейсе
            idx = np.arange(len(X))
            sub_n = mp.get("grid_subsample")
            if sub_n and len(X) > sub_n:
                idx = np.sort(np.random.default_rng(cp["seed"] + int(y)).choice(len(X), sub_n, replace=False))
            Xs, Ws, ids = X[idx], Wsp[np.ix_(idx, idx)], net.ids[idx]
            for k in ks:
                lab = clustering.fit(Xs, Ws, {**mp, "method": m, "k": k, "seed": cp["seed"]})
                labels.append(pd.DataFrame({"year": y, "method": m, "k": k, "territory_id": ids, "label": lab}))
                scores.append({"year": y, "method": m, "k": k, "n_nodes": len(idx), **icvi.compute_all(Xs, Ws, lab)})
            print(f"{y} {m}: {time.time() - t:.1f} с", flush=True)
    out = CLUSTERS / h
    out.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        L, S = pd.read_parquet(out / "labels.parquet"), pd.read_parquet(out / "icvi.parquet")
        methods = {m: v for m, v in cp["methods"].items() if m in set(S["method"])}
    else:
        L = pd.concat(labels, ignore_index=True)
        S = pd.DataFrame(scores)
    if a.only and (out / "labels.parquet").exists():
        # влить в готовую сетку: строки пересчитанных методов заменяются, остальные остаются
        L0, S0 = pd.read_parquet(out / "labels.parquet"), pd.read_parquet(out / "icvi.parquet")
        L = pd.concat([L0[~L0["method"].isin(list(methods))], L], ignore_index=True)
        S = pd.concat([S0[~S0["method"].isin(list(methods))], S], ignore_index=True)
        methods = {m: v for m, v in cp["methods"].items() if m in set(S["method"])}
    if "WCSS" not in S or S["WCSS"].isna().any():
        S = add_wcss(S, L, nets)
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
        "# Кластеризация и индексы качества",
        "",
        f"Сгенерировано `scripts/build_clusters.py`. Сеть `{h}` (параметры `configs/network.yaml`: "
        f"{', '.join(netp['sample']['federal_districts']) or 'вся Россия'}, "
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
    subs = {m: v["grid_subsample"] for m, v in methods.items() if v.get("grid_subsample")}
    if subs:
        rep += [
            "На подвыборке МО (одна и та же для всех k в пределах года; полная выборка — вручную в интерфейсе): "
            + ", ".join(f"{m} — {n} МО" for m, n in subs.items())
            + ". Сравнение с другими методами (ARI) — на общих МО.",
            "",
        ]
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
                "k по локтю (WCSS)": icvi.elbow(t.index, t["WCSS"]),
            }
        )
        tt = t.reset_index()[["k", "K", "SW", "CH", "DBI", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "WCSS"]]
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
        "**Метод локтя.** WCSS — внутрикластерная сумма квадратов в пространстве нормированных признаков; с ростом k "
        "она всегда убывает. «Локоть» — k, после которого убывание резко замедляется: точка кривой WCSS(k), наиболее "
        "удалённая от прямой между первой и последней точкой (обе оси приведены к [0, 1], как в методе Kneedle). "
        "В средний ранг WCSS не входит — сама по себе она всегда «за» большее k.",
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
                common = a_.index.intersection(b_.index)  # CANUS — на подвыборке
                vals.append(adjusted_rand_score(a_.loc[common], b_.loc[common]))
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
