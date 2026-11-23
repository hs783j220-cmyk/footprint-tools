"""Extract footprints for accepted candidates using equations 23–25."""

import hashlib
import json
import math

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from datasets.footprint import load_pair, read_manifest
from utils.geometry import estimate_main_directions_from_contours, snap_ring


def clean_region(mask, occupancy, step, occupancy_threshold, component_overlap,
                 min_area, dilation, closing, max_hole):
    import cv2
    from scipy.ndimage import binary_fill_holes, label

    def kernel(distance):
        radius = int(round(distance / step))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    allowed = cv2.dilate(mask.astype(np.uint8), kernel(dilation)).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((occupancy >= occupancy_threshold).astype(np.uint8), connectivity=8)
    overlap = np.bincount(labels[allowed], minlength=count)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = (areas * step**2 >= min_area) & (overlap / np.maximum(areas, 1) >= component_overlap)
    keep[0] = False
    cleaned = cv2.morphologyEx(keep[labels].astype(np.uint8), cv2.MORPH_CLOSE, kernel(closing)).astype(bool)
    holes, n = label(binary_fill_holes(cleaned) & ~cleaned)
    small = np.bincount(holes.ravel(), minlength=n + 1) * step**2 <= max_hole
    small[0] = False
    return cleaned | small[holes]


def regularize(geometry, directions, simplify, snap_angle, max_shift):
    polygon = geometry.simplify(simplify, preserve_topology=True)
    if not snap_angle or not directions:
        return polygon
    rings = [polygon.exterior, *polygon.interiors]
    corrected = [snap_ring(np.asarray(ring.coords)[:-1], directions, snap_angle, max_shift) for ring in rings]
    oriented = Polygon(corrected[0], corrected[1:])
    return oriented if oriented.is_valid and oriented.area > 0 else polygon


def run(manifest, confidence, output, occupancy_threshold, component_overlap, min_area,
        dilation, closing, max_hole, simplify, snap_angle, max_shift):
    import cv2
    import geopandas as gpd
    from rasterio.features import shapes
    from rasterio.transform import from_origin

    if not (0 < occupancy_threshold <= 1 and 0 <= component_overlap <= 1 and 0 <= snap_angle <= 45):
        raise ValueError("Invalid occupancy, component overlap or direction threshold")
    if any(not math.isfinite(v) or v < 0 for v in (min_area, dilation, closing, max_hole, simplify, max_shift)):
        raise ValueError("Areas, distances and tolerances must be finite and nonnegative")
    if output.suffix.lower() != ".gpkg" or output.exists():
        raise ValueError("Specify a new .gpkg file")
    data = read_manifest(manifest)
    report = json.loads(confidence.read_text(encoding="utf-8-sig"))
    if report["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("Confidence report does not match this manifest")
    accepted = {row["group_id"]: row["confidence"] for row in report["candidates"] if row["accepted"]}
    records = []
    for row in data["samples"]:
        group = row["group_id"]
        if group not in accepted:
            continue
        mask, occupancy = load_pair(manifest.parent, row)
        grid = row["grid"]
        step = grid["step"]
        binary = clean_region(mask, occupancy, step, occupancy_threshold, component_overlap,
                              min_area, dilation, closing, max_hole)
        if not binary.any():
            print(f"Candidate {group}: no supported region survived")
            continue
        pixels = binary.astype(np.uint8)
        contours, _ = cv2.findContours(pixels, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        directions = estimate_main_directions_from_contours(contours)
        transform = from_origin(grid["min_x"], grid["max_y"], step, step)
        polygons = [shape(geometry) for geometry, _ in shapes(pixels, mask=binary, transform=transform)]
        base = unary_union([regularize(poly, directions, simplify, 0, max_shift) for poly in polygons])
        adjusted = unary_union([regularize(poly, directions, simplify, snap_angle, max_shift) for poly in polygons])
        parts = lambda geom: sorted(len(poly.interiors) for poly in getattr(geom, "geoms", (geom,)))
        geometry = adjusted if adjusted.is_valid and parts(adjusted) == parts(base) else base
        records.append({"group_id": group, "confidence": accepted[group], "geometry": geometry})
    if not records:
        raise ValueError("No accepted candidate yielded a footprint")
    output.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(records, crs=data["crs"]).to_file(output, driver="GPKG", layer="footprints", index=False)
    print(f"Exported {len(records)}/{len(accepted)} accepted candidates")
