import math

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def get_polygon_union(polygons_world):
    valid_polys = []

    for poly in polygons_world:
        if poly is None or len(poly) < 3:
            continue

        try:
            p = Polygon(poly)
            if not p.is_valid:
                p = p.buffer(0)
            if p is not None and not p.is_empty and p.area > 0:
                valid_polys.append(p)
        except Exception:
            continue

    if len(valid_polys) == 0:
        return None

    geom = unary_union(valid_polys)

    if not geom.is_valid:
        geom = geom.buffer(0)

    return geom


def compute_roi_bounds_from_polygon(poly_geom, buffer_m):
    minx, miny, maxx, maxy = poly_geom.bounds

    return {
        "min_x": float(minx - buffer_m),
        "min_y": float(miny - buffer_m),
        "max_x": float(maxx + buffer_m),
        "max_y": float(maxy + buffer_m)
    }


def build_soft_roi_weight_map(binary_mask, step, tau_m=0.8, sigma_m=0.35):
    if binary_mask is None or binary_mask.size == 0:
        return None

    mask = (binary_mask > 0).astype(np.uint8)

    if np.sum(mask) == 0:
        return None

    inv = (1 - mask).astype(np.uint8)

    dist_px = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dist_m = dist_px * step

    weights = np.exp(-(dist_m ** 2) / max(sigma_m ** 2, 1e-6))
    weights[mask == 1] = 1.0
    weights[dist_m > tau_m] = 0.0

    return weights.astype(np.float32)


def build_mask_prior_buffer(original_mask, step, buffer_m=0.8):
    original_mask = (original_mask > 0).astype(np.uint8)

    if np.sum(original_mask) == 0:
        return np.ones_like(original_mask, dtype=np.uint8)

    buffer_px = max(1, int(round(buffer_m / step)))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * buffer_px + 1, 2 * buffer_px + 1)
    )

    prior_buffer = cv2.dilate(
        original_mask,
        kernel,
        iterations=1
    )

    return prior_buffer.astype(np.uint8)


def compute_layer_consistency(points,
                              min_x,
                              max_y,
                              step,
                              width,
                              height,
                              layer_thickness=0.5,
                              weights=None):
    if points.size == 0:
        return (
            np.zeros((height, width), dtype=np.uint16),
            np.zeros((height, width), dtype=np.float32)
        )

    x_idx = np.floor((points[:, 0] - min_x) / step).astype(np.int32)
    y_idx = np.floor((max_y - points[:, 1]) / step).astype(np.int32)

    z_min = np.min(points[:, 2])
    z_idx = np.floor((points[:, 2] - z_min) / layer_thickness).astype(np.int32)

    valid = (
        (x_idx >= 0) & (x_idx < width) &
        (y_idx >= 0) & (y_idx < height)
    )

    x_valid = x_idx[valid]
    y_valid = y_idx[valid]
    z_valid = z_idx[valid]

    if x_valid.size == 0:
        return (
            np.zeros((height, width), dtype=np.uint16),
            np.zeros((height, width), dtype=np.float32)
        )

    voxel_coords = np.vstack((y_valid, x_valid, z_valid)).T
    unique_voxels = np.unique(voxel_coords, axis=0)

    xy_coords = unique_voxels[:, :2]
    unique_xy, layer_counts = np.unique(
        xy_coords,
        axis=0,
        return_counts=True
    )

    consistency_map = np.zeros((height, width), dtype=np.uint16)
    consistency_map[unique_xy[:, 0], unique_xy[:, 1]] = layer_counts.astype(np.uint16)

    density_map = np.zeros((height, width), dtype=np.float32)

    if weights is None:
        np.add.at(density_map, (y_valid, x_valid), 1.0)
    else:
        w_valid = weights[valid].astype(np.float32)
        np.add.at(density_map, (y_valid, x_valid), w_valid)

    return consistency_map, density_map


def angle_180_from_vector(dx, dy):
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    if ang < 0:
        ang += 180.0
    if ang >= 180.0:
        ang -= 180.0
    return float(ang)


def angle_diff_180(a, b):
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def estimate_main_directions_from_contours(contours,
                                           min_len_px=8,
                                           bin_deg=5.0,
                                           peak_ratio=0.25,
                                           min_votes=2,
                                           max_dirs=8):
    """
    每个建筑单独估计主方向。
    从轮廓点中提取线段方向，按长度加权投票。
    """
    angles = []
    weights = []

    for contour in contours:
        pts = contour.reshape(-1, 2).astype(np.float32)

        if len(pts) < 2:
            continue

        # 闭合轮廓，最后一点连回第一点
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]

            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            length = math.hypot(dx, dy)

            if length < min_len_px:
                continue

            a = angle_180_from_vector(dx, dy)
            angles.append(a)
            weights.append(length)

    if len(angles) == 0:
        return [0.0, 90.0]

    angles = np.asarray(angles, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    bin_count = int(round(180.0 / bin_deg))
    hist = np.zeros(bin_count, dtype=np.float32)

    for a, w in zip(angles, weights):
        idx = int(round(a / bin_deg)) % bin_count
        hist[idx] += w

    max_vote = float(hist.max())

    candidate_bins = []

    for i in range(bin_count):
        left = hist[(i - 1) % bin_count]
        mid = hist[i]
        right = hist[(i + 1) % bin_count]

        if mid >= left and mid >= right:
            if mid >= max(max_vote * peak_ratio, min_votes):
                candidate_bins.append((i, mid))

    candidate_bins = sorted(candidate_bins, key=lambda x: x[1], reverse=True)

    dirs = []

    for idx, vote in candidate_bins:
        a = (idx * bin_deg) % 180.0

        too_close = False

        for existing in dirs:
            if angle_diff_180(a, existing) < bin_deg:
                too_close = True
                break

        if not too_close:
            dirs.append(float(round(a, 4)))

        if len(dirs) >= max_dirs:
            break

    if len(dirs) == 0:
        dirs = [0.0, 90.0]

    dirs = sorted(dirs)

    return dirs
