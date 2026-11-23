"""Fit a robust planar similarity transform from paired control points."""

import json
from pathlib import Path

import numpy as np


def fit_similarity(source, target, threshold):
    import cv2

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("Provide at least three paired XY control points")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Control points must be finite")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("RANSAC threshold must be positive")
    src_origin = source.mean(axis=0)
    dst_origin = target.mean(axis=0)
    affine, inliers = cv2.estimateAffinePartial2D(
        source - src_origin, target - dst_origin, ransacReprojThreshold=threshold,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 3:
        raise ValueError("Could not fit a similarity transform with three inliers")
    linear = affine[:, :2]
    if not np.isfinite(affine).all() or np.linalg.det(linear) <= 0:
        raise ValueError("The fitted similarity transform is degenerate")

    matrix = np.eye(4)
    matrix[:2, :2] = linear
    matrix[:2, 3] = dst_origin + affine[:, 2] - linear @ src_origin
    residuals = np.linalg.norm(source @ linear.T + matrix[:2, 3] - target, axis=1)
    return matrix, residuals, inliers.ravel().astype(bool)


def run(ply: Path, controls: Path, output: Path, threshold: float):
    import open3d as o3d
    from pyproj import CRS, Transformer

    data = json.loads(controls.read_text(encoding="utf-8-sig"))
    output_crs = CRS.from_user_input(data["output_crs"])
    if not output_crs.is_projected or any(axis.unit_conversion_factor != 1 for axis in output_crs.axis_info[:2]):
        raise ValueError("output_crs must be a projected CRS in metres")
    target = np.asarray(data["target_xy"], dtype=float)
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target_xy must contain pairs of coordinates")
    projector = Transformer.from_crs(data["target_crs"], output_crs, always_xy=True)
    x, y = projector.transform(target[:, 0], target[:, 1], errcheck=True)
    matrix, residuals, inliers = fit_similarity(data["source_xy"], np.column_stack((x, y)), threshold)

    if ply.resolve() == output.resolve() or output.suffix.lower() != ".ply":
        raise ValueError("Specify a different output path ending in .ply")
    cloud = o3d.io.read_point_cloud(str(ply))
    if not cloud.has_points():
        raise ValueError("Input PLY contains no readable points")
    cloud.transform(matrix)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud):
        raise OSError(f"Could not write {output}")
    report = {
        "output_crs": output_crs.to_string(),
        "matrix": matrix.tolist(),
        "inliers": inliers.tolist(),
        "residuals_m": residuals.tolist(),
        "inlier_rmse_m": float(np.sqrt(np.mean(residuals[inliers] ** 2))),
    }
    output.with_suffix(".registration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
