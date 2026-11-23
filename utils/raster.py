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
        if crs is None or not crs.is_projected or not math.isclose(crs.linear_units_factor[1], 1.0):
            raise ValueError("GeoTIFF must have a projected CRS in metres")
        if T.b != 0 or T.d != 0 or T.a <= 0 or T.e >= 0:
            raise ValueError("GeoTIFF must be north-up and unrotated")
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
