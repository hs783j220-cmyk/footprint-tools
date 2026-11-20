import math
import os

import numpy as np


def convert_pixel_polygons_to_world(group_to_polygons_pixel, tif_path):
    """
    将 JSON / SAM mask 中的像素坐标转换为真实地图坐标。
    以 tif transform 为准。
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("需要安装 rasterio: pip install rasterio")

    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"未找到 tif 文件: {tif_path}")

    with rasterio.open(tif_path) as src:
        T = src.transform
        crs = src.crs
        tif_width = src.width
        tif_height = src.height
        tif_bounds = src.bounds
        tif_transform_tuple = tuple(T)

        group_to_polygons_world = {}

        for group_id, polys in group_to_polygons_pixel.items():
            world_polys = []

            for poly in polys:
                world_coords = []

                for x, y in poly:
                    X, Y = T * (float(x), float(y))
                    world_coords.append([X, Y])

                world_polys.append(np.array(world_coords, dtype=np.float64))

            group_to_polygons_world[group_id] = world_polys

    tif_info = {
        "crs": str(crs),
        "width": int(tif_width),
        "height": int(tif_height),
        "bounds": {
            "left": float(tif_bounds.left),
            "bottom": float(tif_bounds.bottom),
            "right": float(tif_bounds.right),
            "top": float(tif_bounds.top)
        },
        "transform": list(map(float, tif_transform_tuple))
    }

    return group_to_polygons_world, tif_info


def align_roi_bounds_to_tif_window(roi_bounds, tif_path):
    """
    以 tif 为准计算裁剪窗口，并根据裁剪窗口反推真实世界范围。
    这样每个建筑的输出 ROI 与 tif 像素坐标可追溯。
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        raise ImportError("需要安装 rasterio: pip install rasterio")

    with rasterio.open(tif_path) as src:
        raw_window = from_bounds(
            roi_bounds["min_x"],
            roi_bounds["min_y"],
            roi_bounds["max_x"],
            roi_bounds["max_y"],
            transform=src.transform
        )

        # 向外取整，保证完整覆盖
        col_off = int(math.floor(raw_window.col_off))
        row_off = int(math.floor(raw_window.row_off))
        col_end = int(math.ceil(raw_window.col_off + raw_window.width))
        row_end = int(math.ceil(raw_window.row_off + raw_window.height))

        col_off = max(0, col_off)
        row_off = max(0, row_off)
        col_end = min(src.width, col_end)
        row_end = min(src.height, row_end)

        width = max(1, col_end - col_off)
        height = max(1, row_end - row_off)

        window = rasterio.windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=width,
            height=height
        )

        win_bounds = rasterio.windows.bounds(window, src.transform)

        tif_window_info = {
            "col_off": int(col_off),
            "row_off": int(row_off),
            "width": int(width),
            "height": int(height),
            "bounds": {
                "left": float(win_bounds[0]),
                "bottom": float(win_bounds[1]),
                "right": float(win_bounds[2]),
                "top": float(win_bounds[3])
            }
        }

    # 注意：占据图仍然用 step=0.1 m 建图；
    # 但 min_x/min_y/max_x/max_y 是由 tif 裁剪窗口对应的真实地理范围决定的。
    aligned_bounds = {
        "min_x": float(tif_window_info["bounds"]["left"]),
        "min_y": float(tif_window_info["bounds"]["bottom"]),
        "max_x": float(tif_window_info["bounds"]["right"]),
        "max_y": float(tif_window_info["bounds"]["top"])
    }

    return aligned_bounds, tif_window_info


def save_tif_crop_by_window(tif_path, tif_window_info, out_path):
    try:
        import rasterio
    except ImportError:
        raise ImportError("需要安装 rasterio: pip install rasterio")

    with rasterio.open(tif_path) as src:
        from rasterio.windows import Window

        window = Window(
            col_off=tif_window_info["col_off"],
            row_off=tif_window_info["row_off"],
            width=tif_window_info["width"],
            height=tif_window_info["height"]
        )

        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update({
            "height": int(tif_window_info["height"]),
            "width": int(tif_window_info["width"]),
            "transform": src.window_transform(window)
        })

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
