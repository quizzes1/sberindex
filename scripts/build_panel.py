"""Сборка годовой панели.

Запуск:  python scripts/build_panel.py

Результаты:
  data/processed/panel_long.parquet   territory_id, year, indicator, value, unit, source, flag, derived, valid_in_year
  data/processed/panel_wide.parquet   одна строка на territory_id × year
  data/processed/mo.parquet           справочник МО (название, тип, ОКТМО по годам, регион, ФО, координаты, флаги)
  data/processed/unmatched.csv        строки источников, не привязанные к territory_id, с причиной
  data/processed/rejected.csv         отбракованные заведомо ошибочные значения (например, население = 0)
  data/geo/mo.gpkg                    полные полигоны + атрибуты
  data/geo/mo_simplified.geojson      упрощённые полигоны для карты (долготы 0…360)
  data/geo/mo_simplified_coarse.geojson  сильнее упрощённые — для карты всей России
  data/geo/adjacency.parquet          пары смежных МО (общая граница, допуск 200 м, с учётом 180-го меридиана)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import geo, panel  # noqa: E402
from src.io import GEO, PROCESSED, load_yaml  # noqa: E402


def main() -> int:
    """Точка входа: сборка годовой панели."""
    t = time.time()
    cfg = load_yaml("panel.yaml")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    GEO.mkdir(parents=True, exist_ok=True)

    print("Читаю ряды и собираю long-панель…", flush=True)
    long, unmatched, rejected = panel.build(cfg)

    print("Геометрия…", flush=True)
    polys = geo.load_polygons()
    cent = geo.centroids(polys)
    mo = panel.mo_table(long).merge(cent, on="territory_id", how="left")
    # координаты центра: из справочника; если нет (города федерального значения) — центроид полигона
    mo["coord_source"] = mo["lat"].notna().map({True: "центр МО (справочник)", False: "центроид полигона"})
    mo["lat"] = mo["lat"].fillna(mo["poly_lat"])
    mo["lon"] = mo["lon"].fillna(mo["poly_lon"])
    mo["lon360"] = mo["lon"].where(mo["lon"] >= 0, mo["lon"] + 360)
    mo["has_polygon"] = mo["territory_id"].isin(polys["territory_id"])

    long.to_parquet(PROCESSED / "panel_long.parquet", index=False)
    wide = panel.to_wide(long)
    wide.to_parquet(PROCESSED / "panel_wide.parquet", index=False)
    mo.to_parquet(PROCESSED / "mo.parquet", index=False)
    unmatched.to_csv(PROCESSED / "unmatched.csv", index=False, encoding="utf-8")
    rejected.to_csv(PROCESSED / "rejected.csv", index=False, encoding="utf-8")

    attrs = mo[["territory_id", "name", "type", "region_name", "federal_district", "is_dfo", "active"]]
    full = polys[["territory_id", "geometry"]].merge(attrs, on="territory_id", how="left")
    full.to_file(GEO / "mo.gpkg", layer="mo", driver="GPKG")
    simp = geo.simplified(full)
    out = GEO / "mo_simplified.geojson"
    if out.exists():
        out.unlink()
    simp.to_file(out, driver="GeoJSON")
    # облегчённый слой для карты всей России (~2,7 МБ вместо ~5,3 МБ)
    coarse = GEO / "mo_simplified_coarse.geojson"
    if coarse.exists():
        coarse.unlink()
    geo.simplified(full, tolerance=0.03).to_file(coarse, driver="GeoJSON")
    geo.adjacency(polys).to_parquet(GEO / "adjacency.parquet", index=False)

    # краткая сводка
    w = wide[wide["valid_in_year"]]
    dfo = set(mo.loc[mo["is_dfo"], "territory_id"])
    print(
        f"panel_long: {len(long):,} строк, {long['indicator'].nunique()} рядов; "
        f"panel_wide: {len(wide):,} строк; МО: {len(mo)} (ДФО {len(dfo)})"
    )
    print("МО с населением по годам (действующие, РФ / ДФО):")
    s = w.dropna(subset=["pop"]).groupby("year")["territory_id"].agg(ru="nunique", dfo=lambda x: x.isin(dfo).sum())
    print(s.T.to_string())
    print(
        f"непривязанных групп строк: {len(unmatched)}; geojson {out.stat().st_size / 1e6:.1f} МБ; "
        f"{time.time() - t:.0f} с"
    )
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    raise SystemExit(main())
