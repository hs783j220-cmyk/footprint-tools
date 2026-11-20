from pathlib import Path
import math

import numpy as np
from PIL import Image


BLACK_LINE_THRESHOLD = 80


def load_white_bg_black_line(path: Path, threshold: int = BLACK_LINE_THRESHOLD) -> np.ndarray:
    """
    读取白底黑线 PNG，提取黑色线条。

    返回：
        line_mask=True 表示黑色轮廓线像素。

    不做：
    - 去噪
    - 闭运算
    - 膨胀腐蚀
    - 最大连通区域筛选
    """
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img)

    rgb = arr[..., :3].astype(np.uint8)
    alpha = arr[..., 3]

    visible = alpha > 0

    # 白底黑线：三个通道都比较暗，认为是线条
    black_line = (
        (rgb[..., 0] <= threshold) &
        (rgb[..., 1] <= threshold) &
        (rgb[..., 2] <= threshold)
    )

    line_mask = visible & black_line

    total_pixels = line_mask.size
    line_pixels = int(line_mask.sum())
    ratio = line_pixels / total_pixels if total_pixels > 0 else 0.0

    print(
        f"Loaded line: {path.name} | "
        f"line_pixels={line_pixels}, line_ratio={ratio:.6f}"
    )

    if line_pixels == 0:
        print(f"WARNING: {path.name} 没有读到黑色线条，可能阈值太小或图像不是白底黑线。")

    return line_mask


def resize_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    最近邻缩放二值 mask。
    """
    h, w = target_shape
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    im = im.resize((w, h), Image.Resampling.NEAREST)
    return np.asarray(im) > 0


def check_or_align_shape(pred: np.ndarray, gt: np.ndarray, pred_name: str, *, allow_resize: bool = False) -> tuple[np.ndarray, bool]:
    """
    检查预测图和 GT 尺寸是否一致。
    """
    if pred.shape == gt.shape:
        return pred, False

    if not allow_resize:
        raise ValueError(
            f"尺寸不一致，已停止评价。\n"
            f"Prediction file: {pred_name}\n"
            f"Prediction shape: {pred.shape[1]}x{pred.shape[0]}\n"
            f"GT shape: {gt.shape[1]}x{gt.shape[0]}\n"
            f"如果确认两者范围一致，只是尺寸不同，可以把 使用 --allow-resize。"
        )

    return resize_nearest(pred, gt.shape), True


def line_to_area_mask(line_mask: np.ndarray) -> np.ndarray:
    """
    将白底黑线轮廓转换为面积 mask。

    只做：
    - findContours
    - drawContours 填充闭合区域

    不做：
    - 最大连通区域筛选
    - 删除小区域
    - 开闭运算
    - 膨胀腐蚀
    - 断线修补

    注意：
    如果轮廓线不闭合，填充结果可能不正确。
    """
    try:
        import cv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "缺少 opencv-python，请运行: python -m pip install opencv-python"
        )

    line_u8 = (line_mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        line_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filled = np.zeros_like(line_u8, dtype=np.uint8)

    if len(contours) > 0:
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)

    area_mask = filled > 0

    area_pixels = int(area_mask.sum())
    ratio = area_pixels / area_mask.size if area_mask.size > 0 else 0.0

    print(
        f"Filled area mask | "
        f"contours={len(contours)}, area_pixels={area_pixels}, area_ratio={ratio:.6f}"
    )

    return area_mask


def get_mask_boundary(mask: np.ndarray) -> np.ndarray:
    """
    提取面积 mask 的边界。
    不做容差、不做膨胀。
    """
    try:
        import cv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "缺少 opencv-python，请运行: python -m pip install opencv-python"
        )

    mask_u8 = mask.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)

    boundary = mask_u8.astype(bool) & (~eroded.astype(bool))
    return boundary


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def evaluate_area_and_boundary(pred_area: np.ndarray, gt_area: np.ndarray) -> dict:
    """
    计算面积指标和边界指标。
    """
    pred = pred_area.astype(bool)
    gt = gt_area.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    area_iou = safe_div(tp, tp + fp + fn)

    pred_area_pixels = int(pred.sum())
    gt_area_pixels = int(gt.sum())
    intersection_pixels = tp
    union_pixels = int(np.logical_or(pred, gt).sum())
    total_pixels = int(pred.size)

    pred_boundary = get_mask_boundary(pred)
    gt_boundary = get_mask_boundary(gt)

    b_tp = int(np.logical_and(pred_boundary, gt_boundary).sum())
    b_fp = int(np.logical_and(pred_boundary, ~gt_boundary).sum())
    b_fn = int(np.logical_and(~pred_boundary, gt_boundary).sum())

    boundary_precision = safe_div(b_tp, b_tp + b_fp)
    boundary_recall = safe_div(b_tp, b_tp + b_fn)
    boundary_f1 = safe_div(
        2 * boundary_precision * boundary_recall,
        boundary_precision + boundary_recall,
    )
    boundary_iou = safe_div(b_tp, b_tp + b_fp + b_fn)

    strict_score = area_iou * boundary_iou

    return {
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Area_IoU": area_iou,

        "Boundary_Precision": boundary_precision,
        "Boundary_Recall": boundary_recall,
        "Boundary_F1": boundary_f1,
        "Boundary_IoU": boundary_iou,

        "Strict_Score": strict_score,

        "TP_pixels": tp,
        "FP_pixels": fp,
        "FN_pixels": fn,
        "TN_pixels": tn,

        "Pred_Area_Pixels": pred_area_pixels,
        "GT_Area_Pixels": gt_area_pixels,
        "Intersection_Pixels": intersection_pixels,
        "Union_Pixels": union_pixels,
        "Total_Pixels": total_pixels,

        "Pred_Boundary_Pixels": int(pred_boundary.sum()),
        "GT_Boundary_Pixels": int(gt_boundary.sum()),
        "Boundary_TP_pixels": b_tp,
        "Boundary_FP_pixels": b_fp,
        "Boundary_FN_pixels": b_fn,
        "Boundary_Union_Pixels": int(np.logical_or(pred_boundary, gt_boundary).sum()),
    }


def save_debug_line_mask(mask: np.ndarray, path: Path) -> None:
    """
    保存调试线图：白底黑线。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    out = np.ones(mask.shape, dtype=np.uint8) * 255
    out[mask] = 0

    Image.fromarray(out, mode="L").save(path)


def save_debug_area_mask(mask: np.ndarray, path: Path) -> None:
    """
    保存调试面积图：黑底白色区域。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    out = np.zeros(mask.shape, dtype=np.uint8)
    out[mask] = 255

    Image.fromarray(out, mode="L").save(path)


def write_xlsx(rows: list[dict], path: Path) -> None:
    """
    只输出 Excel。
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "缺少 openpyxl，请运行: python -m pip install openpyxl"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Area Metrics"

    headers = list(rows[0].keys())
    ws.append(headers)

    for row in rows:
        ws.append([row.get(h, "") for h in headers])

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    avg_fill = PatternFill("solid", fgColor="FFF2CC")
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

    float_metric_cols = {
        "Precision",
        "Recall",
        "F1_Score",
        "Area_IoU",
        "Boundary_Precision",
        "Boundary_Recall",
        "Boundary_F1",
        "Boundary_IoU",
        "Strict_Score",
    }

    for row_idx in range(2, ws.max_row + 1):
        is_avg_row = str(ws.cell(row_idx, 1).value) == "Average"

        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row_idx, col_idx)
            header = ws.cell(1, col_idx).value

            cell.border = border
            cell.alignment = Alignment(horizontal="center", vertical="center")

            if is_avg_row:
                cell.fill = avg_fill
                cell.font = Font(bold=True)

            if header in float_metric_cols:
                cell.number_format = "0.0000"
            elif isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                cell.number_format = "0"

    for col_idx, header in enumerate(headers, start=1):
        max_len = len(str(header))

        for r in range(2, ws.max_row + 1):
            value = ws.cell(r, col_idx).value
            max_len = max(max_len, len(str(value)))

        ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 10), 34)

    ws.freeze_panes = "A2"
    wb.save(path)


def build_average_row(rows: list[dict]) -> dict:
    """
    生成平均值行。
    只对数值列求平均。
    """
    avg_row = {}

    for key in rows[0].keys():
        if key == "id":
            avg_row[key] = "Average"
            continue

        if key in {
            "Prediction_File",
            "GT_File",
            "Resized_To_GT",
        }:
            avg_row[key] = ""
            continue

        values = []

        for row in rows:
            value = row.get(key, "")

            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                if not math.isnan(float(value)):
                    values.append(float(value))

        avg_row[key] = float(np.mean(values)) if values else ""

    return avg_row
