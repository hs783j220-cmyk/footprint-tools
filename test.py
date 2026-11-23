"""Measure instance-aligned raster precision, recall, F1 and IoU."""

import json
import math

import numpy as np
from shapely.ops import unary_union

from utils.metrics import overlap_scores, polis


def run(prediction, reference, output, resolution, id_field):
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Raster resolution must be positive")
    pred = gpd.read_file(prediction)
    truth = gpd.read_file(reference)
    crs = pred.crs
    if crs is None or not crs.is_projected or any(axis.unit_conversion_factor != 1 for axis in crs.axis_info[:2]):
        raise ValueError("Prediction requires a projected CRS in metres")
    if truth.crs is None or truth.empty:
        raise ValueError("Reference requires nonempty polygons with a CRS")
    truth = truth.to_crs(crs)

    def by_instance(frame):
        if id_field not in frame or frame[id_field].isna().any():
            raise ValueError(f"Both files must contain nonempty {id_field} values")
        result = {}
        for key, rows in frame.groupby(frame[id_field].astype(str)):
            polygons = [g if g.is_valid else g.buffer(0) for g in rows.geometry if g is not None and not g.is_empty]
            if not polygons or any(g.geom_type not in {"Polygon", "MultiPolygon"} for g in polygons):
                raise ValueError(f"Instance {key} requires polygonal geometry")
            geometry = unary_union(polygons)
            if geometry.is_empty or geometry.area <= 0 or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                raise ValueError(f"Instance {key} has no valid polygon area")
            result[key] = geometry
        return result

    a, b = by_instance(pred), by_instance(truth)
    counts = np.zeros(3, dtype=np.int64)
    ious, distances = [], []
    for key in sorted(a.keys() | b.keys()):
        ga, gb = a.get(key), b.get(key)
        extent = unary_union([g for g in (ga, gb) if g is not None]).bounds
        left, bottom = (math.floor(v / resolution) for v in extent[:2])
        right, top = (math.ceil(v / resolution) for v in extent[2:])
        shape = (top - bottom, right - left)
        transform = from_origin(left * resolution, top * resolution, resolution, resolution)

        def mask(geom):
            if geom is None:
                return np.zeros(shape, dtype=bool)
            return rasterize([(geom, 1)], out_shape=shape, transform=transform, dtype="uint8").astype(bool)

        predicted, actual = mask(ga), mask(gb)
        tp = np.count_nonzero(predicted & actual)
        fp = np.count_nonzero(predicted & ~actual)
        fn = np.count_nonzero(~predicted & actual)
        counts += (tp, fp, fn)
        if tp + fp + fn:
            ious.append(overlap_scores(tp, fp, fn)["iou"])
        if ga is not None and gb is not None:
            distances.append(polis(ga, gb))

    tp, fp, fn = map(int, counts)
    if tp + fp + fn == 0:
        raise ValueError("No foreground pixels at the selected resolution")
    metrics = {"tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn,
               **overlap_scores(tp, fp, fn), "miou": float(np.mean(ious)),
               "polis_m": float(np.mean(distances)) if distances else None,
               "polis_matched_instances": len(distances),
               "resolution_m": resolution, "crs": crs.to_string()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
