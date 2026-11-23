"""Build binary 2D masks and continuous 3D occupancy maps (equations 8–13)."""

import json
import math
import re

import numpy as np

from datasets.annotations import extract_polygons_all_groups_from_json
from utils.raster import convert_pixel_polygons_to_world


def occupancy_grid(points, rows, cols, shape, interval, max_slices, min_points):
    result = np.zeros(shape, dtype=np.float32)
    if not len(points):
        return result
    z = points[:, 2]
    bottom = z.min()
    count = max(1, min(max_slices, int(np.ceil((z.max() - bottom) / interval))))
    levels = np.floor((z - bottom) / interval).astype(np.int64)
    # Include points on the top boundary when the complete height range fits.
    in_range = (levels < count) | ((levels == count) & (z <= bottom + count * interval))
    levels = np.minimum(levels[in_range], count - 1)
    voxels, population = np.unique(
        np.column_stack((levels, rows[in_range], cols[in_range])), axis=0, return_counts=True,
    )
    supported = voxels[population >= min_points]
    np.add.at(result, (supported[:, 1], supported[:, 2]), 1.0 / count)
    return np.minimum(result, 1)


def remove_ground(local, dem, clearance):
    if not len(local):
        return local
    elevation = np.ma.vstack(list(dem.sample(local[:, :2], indexes=1, masked=True)))[:, 0]
    if np.ma.getmaskarray(elevation).any() or not np.isfinite(elevation.data).all():
        raise ValueError("Ground DEM must cover the candidate and have valid elevations")
    height = local[:, 2] - elevation.data
    return local[height > clearance]


def run(ply, annotations, geotiff, output, min_points, buffer, step, interval, max_slices,
        ground_dem=None, ground_clearance=None, preview=False):
    from contextlib import ExitStack

    import open3d as o3d
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely import intersects_xy
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if min_points < 1 or max_slices < 1 or any(not math.isfinite(v) or v <= 0 for v in (buffer, step, interval)):
        raise ValueError("Use positive finite grid parameters and positive count thresholds")
    if (ground_dem is None) != (ground_clearance is None):
        raise ValueError("Provide --ground-dem and --ground-clearance together")
    if ground_clearance is not None and (not math.isfinite(ground_clearance) or ground_clearance < 0):
        raise ValueError("Ground clearance must be finite and nonnegative")
    cloud = o3d.io.read_point_cloud(str(ply))
    points = np.asarray(cloud.points)
    if not points.size or not np.isfinite(points).all():
        raise ValueError("PLY must contain finite, registered XYZ points")
    groups = extract_polygons_all_groups_from_json(annotations)
    polygons, info = convert_pixel_polygons_to_world(groups, str(geotiff))
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with ExitStack() as resources:
        dem = resources.enter_context(rasterio.open(ground_dem)) if ground_dem is not None else None
        if dem is not None and dem.crs != rasterio.crs.CRS.from_string(info["crs"]):
            raise ValueError("Ground DEM must use the same CRS as the GeoTIFF")
        for key, rings in polygons.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
                raise ValueError(f"Invalid candidate ID {key!r}")
            polygon = unary_union([Polygon(ring).buffer(0) for ring in rings])
            if polygon.is_empty:
                continue
            region = polygon.buffer(buffer)
            xmin, ymin, xmax, ymax = region.bounds
            width, height = math.ceil((xmax - xmin) / step), math.ceil((ymax - ymin) / step)
            transform = from_origin(xmin, ymax, step, step)
            mask = rasterize([(polygon, 1)], out_shape=(height, width), transform=transform, dtype="uint8")
            if not mask.any():
                continue
            local = points[
                (points[:, 0] >= xmin) & (points[:, 0] < xmax)
                & (points[:, 1] > ymin) & (points[:, 1] <= ymax)
            ]
            local = local[intersects_xy(region, local[:, 0], local[:, 1])]
            if dem is not None:
                local = remove_ground(local, dem, ground_clearance)
            cols = np.floor((local[:, 0] - xmin) / step).astype(int)
            rows = np.floor((ymax - local[:, 1]) / step).astype(int)
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            occupancy = occupancy_grid(local[valid], rows[valid], cols[valid], mask.shape, interval, max_slices, min_points)
            archive = f"candidate_{len(records):06d}.npz"
            np.savez_compressed(output / archive, mask=mask, occupancy=occupancy)
            if preview:
                from ..utils.visualize import save_candidate_preview

                save_candidate_preview((output / archive).with_suffix(".png"), mask, occupancy)
            records.append({"group_id": key, "arrays": archive,
                            "grid": {"min_x": xmin, "max_y": ymax, "step": step}})
    if not records:
        raise ValueError("No valid candidate generated a nonempty mask")
    manifest = {"crs": info["crs"], "parameters": {
        "buffer_m": buffer, "step_m": step, "slice_interval_m": interval,
        "max_slices": max_slices, "min_points_per_voxel": min_points,
        "ground_clearance_m": ground_clearance,
    }, "samples": records}
    (output / "samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} candidate pairs to {output}")
