import json

import numpy as np

ALLOWED_LABELS = {"building"}


def normalize_group_id(group, fallback_idx):
    """
    group 可能是 int / str / None。
    如果 group 为空，则自动分配一个 group。
    """
    if group is None:
        return f"nogroup_{fallback_idx}"

    try:
        if isinstance(group, float) and group.is_integer():
            return str(int(group))
        if isinstance(group, int):
            return str(group)
        return str(group)
    except Exception:
        return f"nogroup_{fallback_idx}"


def extract_polygons_all_groups_from_json(json_path):
    """
    兼容 LabelMe(shapes) 和 Custom(objects/segmentation) 格式。
    不再只提取单个 target_group，而是提取所有 label/category/name 为 building 的 group。
    返回：
        group_to_polygons_pixel: dict[group_id] = [N x 2 pixel polygon, ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    group_to_polygons = {}
    fallback_idx = 1

    if "shapes" in data:
        for shape in data["shapes"]:
            label = shape.get("label", "").lower()
            group = shape.get("group_id")
            if group is None:
                group = shape.get("group")

            if label not in ALLOWED_LABELS:
                continue

            if "points" not in shape:
                continue

            group_id = normalize_group_id(group, fallback_idx)
            fallback_idx += 1

            poly = np.array(shape["points"], dtype=np.float64)

            if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
                continue

            group_to_polygons.setdefault(group_id, []).append(poly)

    elif "objects" in data:
        for obj in data["objects"]:
            label = (
                obj.get("category") or
                obj.get("label") or
                obj.get("name") or
                ""
            ).lower()

            group = obj.get("group")

            if label not in ALLOWED_LABELS:
                continue

            if "segmentation" not in obj:
                continue

            seg = obj["segmentation"]

            if not isinstance(seg, list) or len(seg) == 0:
                continue

            if isinstance(seg[0], (int, float)):
                poly = np.array(seg, dtype=np.float64).reshape(-1, 2)
            else:
                poly = np.array(seg, dtype=np.float64)

            if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
                continue

            group_id = normalize_group_id(group, fallback_idx)
            fallback_idx += 1

            group_to_polygons.setdefault(group_id, []).append(poly)

    if len(group_to_polygons) == 0:
        raise RuntimeError(
            f"在 JSON 文件 {json_path} 中未找到 label/category/name 为 building 的多边形。"
        )

    return group_to_polygons
