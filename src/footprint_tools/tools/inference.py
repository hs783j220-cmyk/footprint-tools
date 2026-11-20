import json
import os
from datetime import datetime

import numpy as np

from ..datasets.annotations import extract_polygons_all_groups_from_json
from ..datasets.raster import convert_pixel_polygons_to_world


def run(ply_file, json_file, tif_file, base_save_dir, *, method, process_one_group, parameters):
    """Run the selected occupancy method and return its batch summary."""
    import open3d as o3d
    from uuid import uuid4

    ply_file, json_file, tif_file = map(os.fspath, (ply_file, json_file, tif_file))
    base_save_dir = os.fspath(base_save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{method}_{uuid4().hex[:8]}"
    run_save_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(run_save_dir, exist_ok=False)

    print("本次批处理输出目录:", run_save_dir)

    # 读取点云
    print(f"读取点云: {ply_file}")
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    print(f"点云读取完成，原始点数: {points.shape[0]}")

    if points.shape[0] == 0:
        raise RuntimeError("点云为空，请检查 PLY 文件路径或文件内容。")

    points = points.copy()

    print("已跳过法向量计算，直接使用原始点云。")

    # 读取 JSON 所有 building group
    print(f"读取 JSON: {json_file}")
    group_to_polygons_pixel = extract_polygons_all_groups_from_json(json_file)

    print(f"检测到 building group 数量: {len(group_to_polygons_pixel)}")
    print("group 列表:", list(group_to_polygons_pixel.keys()))

    # 使用 tif transform 将 JSON 像素坐标转换为地图坐标
    group_to_polygons_world, tif_info = convert_pixel_polygons_to_world(
        group_to_polygons_pixel,
        tif_file
    )

    print("已根据 tif transform 将 JSON 像素坐标转换为世界坐标。")
    print("tif CRS:", tif_info["crs"])
    print("tif bounds:", tif_info["bounds"])

    # 保存总配置
    run_config = {
        "run_name": run_name,
        "ply_file": ply_file,
        "json_file": json_file,
        "tif_file": tif_file,
        "base_save_dir": base_save_dir,
        "run_save_dir": run_save_dir,
        "detected_groups": list(group_to_polygons_world.keys()),
        "tif_info": tif_info,
        "coordinate_rule": {
            "description": "Per group ROI pixel coordinate to map coordinate, consistent with GeoTIFF/QGIS display direction",
            "x_map": "min_x + x_pixel * step",
            "y_map": "max_y - y_pixel * step"
        },
        "method": method,
        "global_parameters": parameters
    }

    if method in {"pointcloud", "pointcloud-postprocessed"}:
        run_config["mode"] = "pointcloud_dominant"

    run_config_path = os.path.join(
        run_save_dir,
        "run_config_all_groups.json"
    )

    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # 逐 group 处理
    all_results = []
    unsuccessful_groups = {}

    for group_id in sorted(group_to_polygons_world.keys(), key=lambda x: str(x)):
        try:
            result = process_one_group(
                group_id=group_id,
                polygons_world=group_to_polygons_world[group_id],
                points=points,
                tif_path=tif_file,
                tif_info=tif_info,
                run_save_dir=run_save_dir,
                parameters=parameters,
                ply_file=ply_file,
                json_file=json_file,
            )

            if result is not None:
                all_results.append(result)
            else:
                unsuccessful_groups[group_id] = "Skipped by occupancy method"

        except Exception as e:
            unsuccessful_groups[group_id] = str(e)
            print(f"[错误] group {group_id} 处理失败: {e}")

    # 保存汇总
    summary = {
        "run_name": run_name,
        "processed_group_count": len(all_results),
        "processed_groups": [r["group_id"] for r in all_results],
        "failed_or_skipped_group_count": len(group_to_polygons_world) - len(all_results),
        "unsuccessful_groups": unsuccessful_groups,
        "results": all_results
    }

    summary_path = os.path.join(
        run_save_dir,
        "summary_all_groups.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("全部 group 处理完成")
    print(f"成功处理 group 数量: {len(all_results)} / {len(group_to_polygons_world)}")
    print("汇总文件:", summary_path)
    print("总配置文件:", run_config_path)
    print("输出目录:", run_save_dir)
    print("=" * 80)
    return summary

