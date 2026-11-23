"""Raster overlap and vertex-to-boundary footprint metrics."""

import numpy as np
from shapely.geometry import Point


def overlap_scores(tp, fp, fn):
    union = tp + fp + fn
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if union else 0.0,
        "iou": tp / union if union else 0.0,
    }


def polis(prediction, reference):
    """Symmetric mean exterior-vertex distance to the opposite exterior (metres)."""
    from shapely.ops import unary_union

    def exteriors(geometry):
        polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else (geometry,)
        return [polygon.exterior for polygon in polygons]

    def directional(source, target):
        boundary = unary_union(target)
        return float(np.mean([Point(xy).distance(boundary) for ring in source for xy in list(ring.coords)[:-1]]))

    a, b = exteriors(prediction), exteriors(reference)
    return (directional(a, b) + directional(b, a)) / 2
