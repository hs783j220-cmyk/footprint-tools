import math

import numpy as np


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


def snap_ring(coordinates, directions, angle_limit, shift_limit):
    """Rotate eligible edge lines about their midpoints and intersect adjacent lines."""
    edges = np.roll(coordinates, -1, axis=0) - coordinates
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths == 0):
        return coordinates
    units = edges / lengths[:, None]
    angles = np.degrees(np.arctan2(units[:, 1], units[:, 0]))
    # Occupancy directions use image coordinates where y increases downward.
    candidates = -np.asarray(directions, dtype=float)
    deltas = (candidates[None, :] - angles[:, None] + 90) % 180 - 90
    closest = deltas[np.arange(len(angles)), np.argmin(np.abs(deltas), axis=1)]
    use_snap = (np.abs(closest) <= angle_limit) & (np.abs(closest) > 1e-8)
    headings = np.radians(angles + np.where(use_snap, closest, 0.0))
    snapped = np.column_stack((np.cos(headings), np.sin(headings)))
    midpoints = coordinates + edges * 0.5
    result = coordinates.copy()

    for i in range(len(coordinates)):
        previous = (i - 1) % len(coordinates)
        if not (use_snap[previous] or use_snap[i]):
            continue
        matrix = np.column_stack((snapped[previous], -snapped[i]))
        if abs(np.linalg.det(matrix)) < 1e-6:
            return coordinates
        offsets = np.linalg.solve(matrix, midpoints[i] - midpoints[previous])
        corner = midpoints[previous] + offsets[0] * snapped[previous]
        if np.linalg.norm(corner - coordinates[i]) > shift_limit:
            return coordinates
        result[i] = corner
    return result


