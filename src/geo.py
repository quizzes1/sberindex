"""Геометрия МО: полигоны СберИндекса, 180-й меридиан, центроиды, упрощённый слой для карт."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union

from src.dictionary import DICT_GPKG

# Равновеликая азимутальная проекция с центром в России: долготы 19…190° без разрыва.
LAEA_RUSSIA = "+proj=laea +lat_0=60 +lon_0=105 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"


def load_polygons() -> gpd.GeoDataFrame:
    """Полигоны справочника (одна строка на territory_id), EPSG:4326."""
    g = gpd.read_file(DICT_GPKG)
    g["territory_id"] = g["territory_id"].astype("int64")
    g = g.rename_geometry("geometry") if g.geometry.name != "geometry" else g
    g["geometry"] = shapely.make_valid(g.geometry.values)
    return g


def shift_lon_360(geom):
    """Переносит отрицательные долготы в 180…360 (Чукотка за 180-м меридианом) и сшивает части."""
    if geom is None or geom.is_empty:
        return geom
    xy = shapely.get_coordinates(geom)
    if not (xy[:, 0] < 0).any():
        return geom
    g2 = shapely.transform(geom, lambda c: np.column_stack([np.where(c[:, 0] < 0, c[:, 0] + 360, c[:, 0]), c[:, 1]]))
    return unary_union(shapely.make_valid(g2))


def shifted(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Копия слоя с долготами 0…360 (Россия непрерывна от 19° до ~190° в. д.)."""
    out = g.copy()
    out["geometry"] = [shift_lon_360(x) for x in g.geometry]
    return out


def centroids(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Центроиды полигонов, посчитанные в равновеликой проекции с учётом 180-го меридиана.

    Возвращает territory_id, poly_lat, poly_lon (−180…180), poly_lon360 (0…360).
    """
    s = shifted(g).set_crs(4326, allow_override=True).to_crs(LAEA_RUSSIA)
    c = s.geometry.centroid.to_crs(4326)
    lon360 = np.where(c.x < 0, c.x + 360, c.x)
    return gpd.GeoDataFrame(
        {
            "territory_id": g["territory_id"].values,
            "poly_lat": c.y.values,
            "poly_lon": np.where(lon360 > 180, lon360 - 360, lon360),
            "poly_lon360": lon360,
        },
    )


def simplified(g: gpd.GeoDataFrame, tolerance: float = 0.01) -> gpd.GeoDataFrame:
    """Упрощённый слой для карты интерфейса: долготы 0…360, допуск упрощения в градусах."""
    s = shifted(g)
    s["geometry"] = s.geometry.simplify(tolerance, preserve_topology=True)
    s["geometry"] = s.geometry.set_precision(1e-4)
    return s


def adjacency(g: gpd.GeoDataFrame, tol_m: float = 200.0) -> pd.DataFrame:
    """Пары смежных МО (общая граница), с учётом 180-го меридиана.

    Полигоны OSM соседей не всегда совпадают по границе точно, поэтому МО считаются смежными,
    если расстояние между полигонами не больше tol_m метров (по умолчанию 200 м).
    Возвращает territory_id_x < territory_id_y.
    """
    s = shifted(g)[["territory_id", "geometry"]].set_crs(4326, allow_override=True).to_crs(LAEA_RUSSIA)
    b = s.copy()
    b["geometry"] = b.geometry.buffer(tol_m / 2)
    j = gpd.sjoin(b, b, predicate="intersects")
    pairs = j[j["territory_id_left"] < j["territory_id_right"]][["territory_id_left", "territory_id_right"]]
    pairs.columns = ["territory_id_x", "territory_id_y"]
    return (
        pd.DataFrame(pairs).drop_duplicates().sort_values(["territory_id_x", "territory_id_y"]).reset_index(drop=True)
    )
