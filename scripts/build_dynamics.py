"""Динамика кластеров (этап 6): сквозные номера, переходы, мигранты, ARI, бутстрэп.

Запуск:  python scripts/build_dynamics.py [--method kefrin --k 6] [--boot 50]

Берёт разбиение из data/clusters/{hash сети}/labels.parquet (scripts/build_clusters.py).
Результаты: data/clusters/{hash}/dynamics_{method}_k{k}/ и reports/DYNAMICS.md.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import clustering, dynamics, network, prices  # noqa: E402
from src.io import DATA, PROCESSED, REPORTS, load_yaml  # noqa: E402


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


def main() -> int:
    cfg = load_yaml("dynamics.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=cfg["partition"]["method"])
    ap.add_argument("--k", type=int, default=cfg["partition"]["k"])
    ap.add_argument("--boot", type=int, default=cfg["bootstrap"]["n"])
    ap.add_argument("--mode", default=cfg["partition"].get("mode", "per_year"), choices=["per_year", "pooled"])
    a = ap.parse_args()
    cp = clustering.default_params()
    netp = network.merge_params(network.default_params(), cp.get("network_override"))
    h = network.config_hash(netp)
    L = pd.read_parquet(DATA / "clusters" / h / "labels.parquet")
    mp = {**cp["methods"][a.method], "method": a.method, "k": a.k, "seed": cfg["seed"]}
    Xp = network.features(netp).dropna()
    if a.mode == "pooled":
        lab = clustering.fit_pooled(Xp, mp).reset_index()
    else:
        lab = L[L["method"].eq(a.method) & L["k"].eq(a.k)][["territory_id", "year", "label"]]
    # сквозные типы K1…Kn (0 → K1 — самый высокий ВМП на душу в ценах базового года)
    th = dynamics.through_labels(lab, a.mode, cfg["matching"]["min_jaccard"], netp["prices"])
    tr = dynamics.transitions(th)
    mig = dynamics.migrants(th)
    ys = dynamics.year_summary(th)

    # бутстрэп по узлам для каждого года
    nets = network.build(netp)
    boot = []
    if a.mode == "pooled":
        # бутстрэп по МО: с возвращением берутся МО (со всеми их годами), модель обучается заново
        rng = np.random.default_rng(cfg["seed"])
        ids = Xp.index.get_level_values("territory_id").unique()
        base = lab.set_index(["territory_id", "year"])["label"]
        from sklearn.metrics import adjusted_rand_score

        ari = []
        for b in range(a.boot):
            s_ = rng.choice(ids, len(ids), replace=True)
            m = Xp.index.get_level_values("territory_id").isin(s_)
            l2 = clustering.fit(Xp[m].to_numpy(), None, {**mp, "seed": cfg["seed"] + b + 1})
            ari.append(adjusted_rand_score(base[Xp.index[m]], l2))
        boot.append(
            {
                "год": "все",
                "повторов": len(ari),
                "ARI медиана": float(np.median(ari)),
                "ARI 5%": float(np.quantile(ari, 0.05)),
                "ARI 95%": float(np.quantile(ari, 0.95)),
            }
        )
    for y, net in nets.items() if a.mode == "per_year" else []:
        base = lab[lab["year"].eq(y)].set_index("territory_id")["label"].reindex(net.ids).to_numpy()
        ari = dynamics.bootstrap_stability(
            lambda X, W: clustering.fit(X, W, mp),
            net.X.to_numpy(),
            net.W * net.A,
            base,
            n=a.boot,
            frac=cfg["bootstrap"]["frac"],
            seed=cfg["seed"],
        )
        boot.append(
            {
                "год": y,
                "повторов": len(ari),
                "ARI медиана": float(np.median(ari)),
                "ARI 5%": float(np.quantile(ari, 0.05)),
                "ARI 95%": float(np.quantile(ari, 0.95)),
            }
        )
        print(f"{y}: бутстрэп ARI медиана {np.median(ari):.2f}", flush=True)
    boot = pd.DataFrame(boot)

    out = DATA / "clusters" / h / f"dynamics_{a.method}_k{a.k}_{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    th.to_parquet(out / "through_labels.parquet", index=False)
    tr.to_parquet(out / "transitions.parquet", index=False)
    mig.to_parquet(out / "migrants.parquet", index=False)
    ys.to_csv(out / "year_summary.csv", index=False, encoding="utf-8")
    boot.to_csv(out / "bootstrap.csv", index=False, encoding="utf-8")

    # паспорта сквозных типов: медианы исходных показателей за последний год
    mo = pd.read_parquet(PROCESSED / "mo.parquet").set_index("territory_id")
    ind = pd.read_parquet(PROCESSED / "indicators_wide.parquet")
    pr = netp["prices"]  # денежные — в тех же ценах, что и признаки сети
    if pr.get("values", "real") == "real":
        ind = prices.to_real(
            ind, list(ind.columns), pr["base_year"], pr["deflator_scope"], pr["spatial_price_adjustment"]
        )
    last = max(nets)
    t_last = th[th["year"].eq(last)].merge(ind[ind["year"].eq(last)], on=["territory_id", "year"])
    feats = list(netp["features"]) + ["pop", "wage", "gmp_imputed_share"]
    prof = t_last.groupby("through")[feats].median()
    prof.insert(0, "МО", t_last.groupby("through").size())
    examples = (
        t_last.assign(name=t_last["territory_id"].map(mo["name_short"]), pop_=t_last["pop"])
        .sort_values("pop_", ascending=False)
        .groupby("through")["name"]
        .apply(lambda s: ", ".join(s.head(4)))
    )
    prof["крупнейшие МО"] = examples

    K = clustering.code
    tm = tr[tr["year_from"].eq(last - 1)].pivot_table(index="from", columns="to", values="n", fill_value=0)
    tm.index = [K(c) for c in tm.index]
    tm.columns = [K(c) for c in tm.columns]
    tm.index.name = "from"
    prof.index = [K(c) for c in prof.index]
    prof.index.name = "through"
    rep = [
        "# Динамика кластеров (этап 6)",
        "",
        f"Сгенерировано `scripts/build_dynamics.py`. Сеть `{h}`, метод **{a.method}**, k = {a.k}, режим **{a.mode}** "
        f"(`configs/dynamics.yaml`). "
        + (
            "Одна модель на все МО-годы: тип одинаково определён во всех годах, сопоставление не нужно. "
            if a.mode == "pooled"
            else f"Кластеры соседних лет сопоставлены венгерским алгоритмом по мере Жаккара "
            f"(порог {cfg['matching']['min_jaccard']}). "
        )
        + "Типы пронумерованы K1…Kn по медиане ВМП на душу в ценах базового года, по убыванию (K1 — самый высокий).",
        "",
        "## 1. Сводка по годам",
        "",
        md(ys),
        "",
        "ARI — скорректированный индекс Рэнда между разбиениями соседних лет на общих МО (1 — совпадают, 0 — "
        "как случайные). «Доля сменивших тип» — по сквозным номерам.",
        "",
        "## 2. Устойчивость к составу выборки (бутстрэп по узлам)",
        "",
        md(boot),
        "",
        f"## 3. Матрица переходов {last - 1} → {last} (число МО)",
        "",
        md(tm.reset_index().rename(columns={"from": "из \\ в"}), "{:.0f}"),
        "",
        f"## 4. Паспорта типов, {last} г. (медианы исходных показателей)",
        "",
        md(prof.reset_index().rename(columns={"through": "тип"}), "{:,.3f}"),
        "",
        f"## 5. «Мигранты»: всего переходов {len(mig)}, МО хотя бы раз сменивших тип — "
        f"{mig['territory_id'].nunique()} из {th['territory_id'].nunique()}",
        "",
    ]
    top = mig.assign(
        МО=mig["territory_id"].map(mo["name"]),
        регион=mig["territory_id"].map(mo["region_name"]),
        **{"from": mig["from"].map(K), "to": mig["to"].map(K)},
    )
    rep += [md(top[top["year_to"].eq(last)][["МО", "регион", "year_from", "year_to", "from", "to"]].head(40), "{}"), ""]
    # сравнение режимов: каждый год заново против одной модели на всю панель
    comp = []
    for m in ("kmeans", "ward", "ward_kmeans", "gmm", "leiden", "spectral", "kefrin", "canus"):
        lm = L[L["method"].eq(m) & L["k"].eq(a.k)][["territory_id", "year", "label"]]
        if len(lm):
            s_ = dynamics.year_summary(dynamics.match_years(lm, cfg["matching"]["min_jaccard"]))
            comp.append(
                {
                    "метод": m,
                    "режим": "per_year",
                    "ARI соседних лет": s_["ARI"].mean(),
                    "доля сменивших тип за год": s_["доля сменивших тип"].mean(),
                }
            )
        if clustering.FAMILIES.get(m) == "attributes":
            lp = clustering.fit_pooled(
                Xp, {**cp["methods"][m], "method": m, "k": a.k, "seed": cfg["seed"]}
            ).reset_index()
            s_ = dynamics.year_summary(dynamics.match_years(lp, cfg["matching"]["min_jaccard"]))
            comp.append(
                {
                    "метод": m,
                    "режим": "pooled",
                    "ARI соседних лет": s_["ARI"].mean(),
                    "доля сменивших тип за год": s_["доля сменивших тип"].mean(),
                }
            )
    rep += [
        f"## 6. Сравнение режимов (k = {a.k}, среднее по парам соседних лет)",
        "",
        "per_year — каждый год кластеризуется заново и кластеры сопоставляются; pooled — одна модель на все "
        "МО-годы (нормировка по всей панели), типы общие для всех лет. Если при per_year доля «сменивших тип» "
        "сопоставима с неустойчивостью из бутстрэпа, заметная часть переходов — шум перекластеризации.",
        "",
        md(pd.DataFrame(comp)),
        "",
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "DYNAMICS.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(f"Готово: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
