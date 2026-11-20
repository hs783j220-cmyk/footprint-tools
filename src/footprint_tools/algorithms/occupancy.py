import os
import cv2
import json
import math
import numpy as np
import re


from ..utils.geometry import (
    get_polygon_union,
    compute_roi_bounds_from_polygon,
    build_soft_roi_weight_map,
    build_mask_prior_buffer,
    compute_layer_consistency,
    estimate_main_directions_from_contours,
)
from ..datasets.raster import align_roi_bounds_to_tif_window, save_tif_crop_by_window
from ..utils.visualization import save_gray_preview, write_image


def polygons_to_mask(polygons, min_x, max_y, width, height, step):
    """
    将世界坐标建筑物多边形栅格化成与 occupancy grid 对齐的 mask。
    mask 内部为 1，外部为 0。

    当前版本采用 GeoTIFF / QGIS 一致的图像方向：
        x_pixel = (X - min_x) / step
        y_pixel = (max_y - Y) / step

    因此输出 PNG 的方向与 tif 显示方向一致。
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if not polygons:
        return mask

    contours = []

    for poly in polygons:
        pts_xy = np.array(poly, dtype=np.float64)

        px = np.round((pts_xy[:, 0] - min_x) / step).astype(np.int32)
        py = np.round((max_y - pts_xy[:, 1]) / step).astype(np.int32)

        pts = np.vstack((px, py)).T

        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

        contours.append(pts.reshape((-1, 1, 2)).astype(np.int32))

    if contours:
        cv2.fillPoly(mask, contours, 1)

    return mask


def generate_mask_dominant_occupancy(original_mask,
                                     consistency_map,
                                     density_map,
                                     step,
                                     points,
                                     layer_thickness=0.5,
                                     mask_buffer_m=0.8,
                                     weak_consistency_thresh=0.02,
                                     strong_consistency_thresh=0.06,
                                     density_min=1,
                                     close_m=0.3):
    original_mask = (original_mask > 0).astype(np.uint8)

    if np.sum(original_mask) == 0:
        raise RuntimeError("original_mask 为空，无法进行 mask 主导生成。")

    if points.size == 0:
        occupancy_mask = original_mask.astype(np.uint8) * 255
        prior_buffer = original_mask.copy()
        lidar_weak = np.zeros_like(original_mask, dtype=np.uint8)
        lidar_strong = np.zeros_like(original_mask, dtype=np.uint8)
        consistency_norm = np.zeros_like(original_mask, dtype=np.float32)

        return occupancy_mask, prior_buffer, lidar_weak, lidar_strong, consistency_norm

    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    num_layers = int(np.ceil((z_max - z_min) / layer_thickness)) + 1
    num_layers = max(num_layers, 1)

    consistency_norm = consistency_map.astype(np.float32) / float(num_layers)

    prior_buffer = build_mask_prior_buffer(
        original_mask=original_mask,
        step=step,
        buffer_m=mask_buffer_m
    )

    lidar_weak = (
        (consistency_norm >= weak_consistency_thresh) &
        (density_map >= density_min) &
        (prior_buffer > 0)
    ).astype(np.uint8)

    lidar_strong = (
        (consistency_norm >= strong_consistency_thresh) &
        (density_map >= density_min) &
        (prior_buffer > 0)
    ).astype(np.uint8)

    occupancy_bin = original_mask.copy()

    occupancy_bin = np.clip(
        occupancy_bin | lidar_strong,
        0,
        1
    ).astype(np.uint8)

    if close_m > 0:
        close_px = max(3, int(round(close_m / step)))

        if close_px % 2 == 0:
            close_px += 1

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_px, close_px)
        )

        occupancy_bin = cv2.morphologyEx(
            occupancy_bin,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=1
        )

    occupancy_bin = occupancy_bin & prior_buffer.astype(np.uint8)

    occupancy_mask = occupancy_bin.astype(np.uint8) * 255

    return occupancy_mask, prior_buffer, lidar_weak, lidar_strong, consistency_norm


# ============================================================
# ===== 4. 轮廓与主方向工具函数 =====
# ============================================================

def extract_contours_with_holes(binary_mask,
                                close_kernel_size=3,
                                close_iterations=1,
                                min_component_area=80,
                                outline_thickness=1):
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    if close_iterations > 0 and close_kernel_size > 1:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (close_kernel_size, close_kernel_size)
        )

        clean_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=close_iterations
        )
    else:
        clean_mask = binary_mask.copy()

    contours, hierarchy = cv2.findContours(
        clean_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or len(contours) == 0:
        raise RuntimeError("未提取到轮廓。")

    outline_all = np.ones_like(clean_mask, dtype=np.uint8) * 255
    filled_all = np.zeros_like(clean_mask, dtype=np.uint8)

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_component_area:
            continue

        valid_contours.append(contour)

    if len(valid_contours) == 0:
        raise RuntimeError(
            f"未找到面积大于 {min_component_area} 的有效轮廓。"
        )

    cv2.drawContours(
        filled_all,
        valid_contours,
        -1,
        255,
        thickness=cv2.FILLED
    )

    cv2.drawContours(
        outline_all,
        valid_contours,
        -1,
        0,
        thickness=outline_thickness
    )

    return clean_mask, filled_all, outline_all, valid_contours


def save_contours_npz_and_txt(contours, save_dir, prefix):
    contours_npz_path = os.path.join(
        save_dir,
        f"{prefix}_final_contours_with_holes.npz"
    )

    contours_txt_dir = os.path.join(
        save_dir,
        f"{prefix}_final_contours_with_holes_txt"
    )

    os.makedirs(contours_txt_dir, exist_ok=True)

    contour_dict = {}

    for i, contour in enumerate(contours):
        pts = contour.reshape(-1, 2)

        contour_name = f"contour_{i + 1}"
        contour_dict[contour_name] = pts

        txt_path = os.path.join(
            contours_txt_dir,
            f"{contour_name}.txt"
        )

        np.savetxt(
            txt_path,
            pts,
            fmt="%d"
        )

    if len(contour_dict) > 0:
        np.savez(
            contours_npz_path,
            **contour_dict
        )

    return contours_npz_path, contours_txt_dir


# ============================================================
# ===== 5. 单 group 处理函数 =====
# ============================================================

def process_one_group(group_id,
                      polygons_world,
                      points,
                      tif_path,
                      tif_info,
                      run_save_dir, parameters, *, ply_file, json_file):
    step = parameters["step"]
    layer_thickness = parameters["layer_thickness"]
    roi_crop_buffer_m = parameters["roi_crop_buffer_m"]
    weak_consistency_thresh = parameters["weak_consistency_thresh"]
    strong_consistency_thresh = parameters["strong_consistency_thresh"]
    density_min = parameters["density_min"]
    mask_buffer_m = parameters["mask_buffer_m"]
    soft_tau_m = parameters["soft_tau_m"]
    soft_sigma_m = parameters["soft_sigma_m"]
    close_m = parameters["close_m"]
    min_contour_area = parameters["min_contour_area"]
    outline_thickness = parameters["outline_thickness"]
    close_kernel_size = parameters["close_kernel_size"]
    close_iterations = parameters["close_iterations"]
    main_direction_min_segment_len_px = parameters["main_direction_min_segment_len_px"]
    main_direction_angle_bin_deg = parameters["main_direction_angle_bin_deg"]
    main_direction_peak_ratio = parameters["main_direction_peak_ratio"]
    main_direction_min_votes = parameters["main_direction_min_votes"]
    max_main_directions = parameters["max_main_directions"]
    save_tif_crop = parameters["save_tif_crop"]
    print("\n" + "=" * 80)
    print(f"开始处理 group: {group_id}")
    print("=" * 80)

    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(group_id)):
        raise ValueError(f"Unsafe building group ID: {group_id!r}")
    prefix = f"group_{group_id}"

    group_save_dir = os.path.join(run_save_dir, prefix)
    os.makedirs(group_save_dir, exist_ok=False)

    poly_geom = get_polygon_union(polygons_world)

    if poly_geom is None or poly_geom.is_empty:
        print(f"[跳过] group {group_id}: 无有效 polygon。")
        return None

    raw_roi_bounds = compute_roi_bounds_from_polygon(
        poly_geom,
        buffer_m=roi_crop_buffer_m
    )

    aligned_bounds, tif_window_info = align_roi_bounds_to_tif_window(
        raw_roi_bounds,
        tif_path=tif_path
    )

    min_x = aligned_bounds["min_x"]
    min_y = aligned_bounds["min_y"]
    max_x = aligned_bounds["max_x"]
    max_y = aligned_bounds["max_y"]

    width = int(np.ceil((max_x - min_x) / step)) + 1
    height = int(np.ceil((max_y - min_y) / step)) + 1

    if width <= 0 or height <= 0:
        print(f"[跳过] group {group_id}: 无效 ROI 尺寸。")
        return None

    # 保存 tif 裁剪图
    tif_crop_path = None

    if save_tif_crop:
        tif_crop_path = os.path.join(
            group_save_dir,
            f"{prefix}_tif_crop.tif"
        )

        try:
            save_tif_crop_by_window(
                tif_path,
                tif_window_info,
                tif_crop_path
            )
            print(f"已保存 tif 裁剪图: {tif_crop_path}")
        except Exception as e:
            print(f"保存 tif 裁剪图失败: {e}")
            tif_crop_path = None

    # 根据 ROI bbox 裁剪点云
    point_mask = (
        (points[:, 0] >= min_x) &
        (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) &
        (points[:, 1] <= max_y)
    )

    points_crop = points[point_mask]

    print(f"group {group_id} ROI 占据图尺寸: {width} x {height}")
    print(f"group {group_id} 裁剪点数: {points_crop.shape[0]}")

    if points_crop.shape[0] == 0:
        print(f"[警告] group {group_id}: ROI 内无点云，仍将基于 mask 输出轮廓。")

    # 生成该 group 的 original_mask
    original_mask = polygons_to_mask(
        polygons_world,
        min_x,
        max_y,
        width,
        height,
        step
    )

    if np.sum(original_mask) == 0:
        print(f"[跳过] group {group_id}: original_mask 为空，请检查 tif/json/点云坐标。")
        return None

    original_mask_path = os.path.join(
        group_save_dir,
        f"{prefix}_image_original_mask.png"
    )

    write_image(
        original_mask_path,
        255 - original_mask.astype(np.uint8) * 255
    )

    # 软 ROI 权重
    weight_map = build_soft_roi_weight_map(
        original_mask,
        step,
        tau_m=soft_tau_m,
        sigma_m=soft_sigma_m
    )

    if weight_map is None or np.max(weight_map) <= 0:
        weight_map = np.ones((height, width), dtype=np.float32)

    weight_path = os.path.join(
        group_save_dir,
        f"{prefix}_roi_soft_weight.png"
    )

    write_image(
        weight_path,
        (np.clip(weight_map, 0, 1) * 255).astype(np.uint8)
    )

    # 对该 group 的点云应用软 ROI
    if points_crop.shape[0] > 0:
        x_idx_all = np.floor((points_crop[:, 0] - min_x) / step).astype(np.int32)
        y_idx_all = np.floor((max_y - points_crop[:, 1]) / step).astype(np.int32)

        valid_idx = (
            (x_idx_all >= 0) & (x_idx_all < width) &
            (y_idx_all >= 0) & (y_idx_all < height)
        )

        keep_mask = np.zeros(points_crop.shape[0], dtype=bool)
        weights = np.zeros(points_crop.shape[0], dtype=np.float32)

        idxs = np.where(valid_idx)[0]

        if idxs.size > 0:
            px = x_idx_all[idxs]
            py = y_idx_all[idxs]

            w = weight_map[py, px]
            keep = w > 0.0

            keep_mask[idxs] = keep
            weights[idxs] = w

        points_roi = points_crop[keep_mask]
        weights_roi = weights[keep_mask]

        if points_roi.shape[0] == 0:
            points_roi = points_crop.copy()
            weights_roi = None
    else:
        points_roi = np.zeros((0, 3), dtype=np.float64)
        weights_roi = None

    print(f"group {group_id} 软 ROI 后点数: {points_roi.shape[0]}")

    # 一致性和密度图
    consistency_map, density_map = compute_layer_consistency(
    points_roi,
    min_x,
    max_y,
    step,
    width,
    height,
    layer_thickness=layer_thickness,
    weights=weights_roi
    )

    consistency_path = os.path.join(
        group_save_dir,
        f"{prefix}_consistency_map.png"
    )

    save_gray_preview(consistency_path, consistency_map)

    density_path = os.path.join(
        group_save_dir,
        f"{prefix}_density_map.png"
    )

    save_gray_preview(density_path, density_map)

    # mask 主导 occupancy
    occupancy_mask, prior_buffer, lidar_weak, lidar_strong, consistency_norm = generate_mask_dominant_occupancy(
        original_mask=original_mask,
        consistency_map=consistency_map,
        density_map=density_map,
        step=step,
        points=points_roi,
        layer_thickness=layer_thickness,
        mask_buffer_m=mask_buffer_m,
        weak_consistency_thresh=weak_consistency_thresh,
        strong_consistency_thresh=strong_consistency_thresh,
        density_min=density_min,
        close_m=close_m
    )

    occupancy_path = os.path.join(
        group_save_dir,
        f"{prefix}_occupancy_mask_mask_dominant.png"
    )

    write_image(
        occupancy_path,
        255 - occupancy_mask
    )

    prior_buffer_path = os.path.join(
        group_save_dir,
        f"{prefix}_prior_buffer.png"
    )

    write_image(
        prior_buffer_path,
        255 - prior_buffer.astype(np.uint8) * 255
    )

    lidar_weak_path = os.path.join(
        group_save_dir,
        f"{prefix}_lidar_weak_evidence.png"
    )

    write_image(
        lidar_weak_path,
        255 - lidar_weak.astype(np.uint8) * 255
    )

    lidar_strong_path = os.path.join(
        group_save_dir,
        f"{prefix}_lidar_strong_evidence.png"
    )

    write_image(
        lidar_strong_path,
        255 - lidar_strong.astype(np.uint8) * 255
    )

    final_mask = occupancy_mask.copy()

    final_preview = os.path.join(
        group_save_dir,
        f"{prefix}_final_occupancy_mask_mask_dominant.png"
    )

    write_image(
        final_preview,
        255 - final_mask
    )

    # 提取轮廓
    try:
        clean_mask, filled_all, outline_with_holes, contours_all = extract_contours_with_holes(
            final_mask,
            close_kernel_size=close_kernel_size,
            close_iterations=close_iterations,
            min_component_area=min_contour_area,
            outline_thickness=outline_thickness
        )

        clean_path = os.path.join(
            group_save_dir,
            f"{prefix}_clean_mask.png"
        )

        filled_path = os.path.join(
            group_save_dir,
            f"{prefix}_final_filled_with_holes_check.png"
        )

        outline_path = os.path.join(
            group_save_dir,
            f"{prefix}_final_outline_with_holes.png"
        )

        write_image(clean_path, 255 - clean_mask)
        write_image(filled_path, 255 - filled_all)
        write_image(outline_path, outline_with_holes)

    except Exception as e:
        print(f"[失败] group {group_id} 轮廓提取失败: {e}")
        return None

    # 每个 group 单独估计主方向
    main_dirs = estimate_main_directions_from_contours(
        contours_all,
        min_len_px=main_direction_min_segment_len_px,
        bin_deg=main_direction_angle_bin_deg,
        peak_ratio=main_direction_peak_ratio,
        min_votes=main_direction_min_votes,
        max_dirs=max_main_directions
    )

    main_dir_path = os.path.join(
        group_save_dir,
        f"{prefix}_multi_main_directions.txt"
    )

    with open(main_dir_path, "w", encoding="utf-8") as f:
        f.write("# Multi main directions estimated separately for this building group\n")
        f.write("# Unit: degree, range: [0, 180)\n")
        f.write(str(main_dirs))

    print(f"group {group_id} 主方向: {main_dirs}")

    # 保存轮廓点
    contours_npz_path, contours_txt_dir = save_contours_npz_and_txt(
        contours_all,
        group_save_dir,
        prefix
    )

    # 保存 ROI 参数，用于后续误差评价
    roi_params = {
        "group_id": str(group_id),
        "paths": {
            "ply_file": ply_file,
            "json_file": json_file,
            "tif_file": tif_path,
            "group_save_dir": group_save_dir,
            "tif_crop_path": tif_crop_path,
            "original_mask_path": original_mask_path,
            "occupancy_path": occupancy_path,
            "final_outline_path": outline_path,
            "contours_npz_path": contours_npz_path,
            "main_direction_txt_path": main_dir_path
        },
        "coordinate_rule": {
            "description": "ROI pixel coordinate to map coordinate, consistent with GeoTIFF/QGIS display direction",
            "x_map": "min_x + x_pixel * step",
            "y_map": "max_y - y_pixel * step"
        },
        "roi_grid": {
            "min_x": float(min_x),
            "min_y": float(min_y),
            "max_x": float(max_x),
            "max_y": float(max_y),
            "step": float(step),
            "width": int(width),
            "height": int(height)
        },
        "tif_window": tif_window_info,
        "tif_info": tif_info,
        "raw_polygon_bounds_with_buffer": raw_roi_bounds,
        "parameters": {
            "layer_thickness": layer_thickness,
            "weak_consistency_thresh": weak_consistency_thresh,
            "strong_consistency_thresh": strong_consistency_thresh,
            "density_min": density_min,
            "mask_buffer_m": mask_buffer_m,
            "soft_tau_m": soft_tau_m,
            "soft_sigma_m": soft_sigma_m,
            "close_m": close_m,
            "min_contour_area": min_contour_area,
            "outline_thickness": outline_thickness,
            "close_kernel_size": close_kernel_size,
            "close_iterations": close_iterations,
            "main_direction_min_segment_len_px": main_direction_min_segment_len_px,
            "main_direction_angle_bin_deg": main_direction_angle_bin_deg,
            "main_direction_peak_ratio": main_direction_peak_ratio,
            "main_direction_min_votes": main_direction_min_votes,
            "max_main_directions": max_main_directions
        },
        "statistics": {
            "points_crop_count": int(points_crop.shape[0]),
            "points_roi_count": int(points_roi.shape[0]),
            "original_mask_pixel_count": int(np.sum(original_mask > 0)),
            "final_occupancy_pixel_count": int(np.sum(final_mask > 0)),
            "contour_count": int(len(contours_all)),
            "main_directions": main_dirs
        }
    }

    roi_params_path = os.path.join(
        group_save_dir,
        f"{prefix}_roi_params.json"
    )

    with open(roi_params_path, "w", encoding="utf-8") as f:
        json.dump(roi_params, f, ensure_ascii=False, indent=2)

    print(f"已保存 ROI 参数: {roi_params_path}")
    print(f"group {group_id} 处理完成。输出目录: {group_save_dir}")

    return roi_params


# ============================================================
# ===== 6. 主程序 =====
# ============================================================
