from pathlib import Path

from ..utils.metrics import (
    build_average_row, check_or_align_shape, evaluate_area_and_boundary,
    get_mask_boundary, line_to_area_mask, load_white_bg_black_line,
    save_debug_area_mask, save_debug_line_mask, write_xlsx,
)


def evaluate(gt_path: Path, predictions: list[Path], output: Path, *, threshold: int = 80, allow_resize: bool = False, debug_dir: Path | None = None) -> None:
    print("Starting area evaluation from white-background black-line PNGs...")
    print(f"GT line PNG: {gt_path}")
    print(f"Predictions: {len(predictions)}")
    print(f"Output Excel: {output}")
    print(f"Black line threshold: {threshold}")
    print(f"Allow resize to GT: {allow_resize}")

    # GT：白底黑线 -> 线条 mask -> 面积 mask
    gt_line = load_white_bg_black_line(gt_path, threshold=threshold)
    gt_area = line_to_area_mask(gt_line)

    print(
        f"GT final area: {gt_path.name}, "
        f"size={gt_area.shape[1]}x{gt_area.shape[0]}, "
        f"area_pixels={int(gt_area.sum())}"
    )

    if debug_dir is not None:
        save_debug_line_mask(gt_line, debug_dir / "GT_01_extracted_black_line.png")
        save_debug_area_mask(gt_area, debug_dir / "GT_02_filled_area_mask.png")
        save_debug_area_mask(get_mask_boundary(gt_area), debug_dir / "GT_03_boundary_mask.png")

    rows = []

    for idx, pred_path in enumerate(predictions, start=1):
        name = pred_path.name

        # Pred：白底黑线 -> 线条 mask
        pred_line = load_white_bg_black_line(pred_path, threshold=threshold)
        original_shape = pred_line.shape

        pred_line, resized = check_or_align_shape(pred_line, gt_line, name, allow_resize=allow_resize)

        # 线条 mask -> 面积 mask
        pred_area = line_to_area_mask(pred_line)

        metrics = evaluate_area_and_boundary(pred_area, gt_area)

        row = {
            "id": idx,
            "Prediction_File": name,
            "GT_File": gt_path.name,
            "GT_Height": gt_area.shape[0],
            "GT_Width": gt_area.shape[1],
            "Pred_Height_Original": original_shape[0],
            "Pred_Width_Original": original_shape[1],
            "Resized_To_GT": resized,
        }

        row.update(metrics)
        rows.append(row)

        if debug_dir is not None:
            stem = f"{idx:03d}_{pred_path.stem}"
            save_debug_line_mask(pred_line, debug_dir / f"{stem}_01_extracted_black_line.png")
            save_debug_area_mask(pred_area, debug_dir / f"{stem}_02_filled_area_mask.png")
            save_debug_area_mask(get_mask_boundary(pred_area), debug_dir / f"{stem}_03_boundary_mask.png")

        print(
            f"[{idx:02d}/{len(predictions):02d}] {name} | "
            f"P={metrics['Precision']:.4f}, "
            f"R={metrics['Recall']:.4f}, "
            f"F1={metrics['F1_Score']:.4f}, "
            f"AreaIoU={metrics['Area_IoU']:.4f}, "
            f"BoundaryIoU={metrics['Boundary_IoU']:.4f}, "
            f"Strict={metrics['Strict_Score']:.4f}, "
            f"TP={metrics['TP_pixels']}, "
            f"FP={metrics['FP_pixels']}, "
            f"FN={metrics['FN_pixels']}"
        )

    rows.append(build_average_row(rows))

    write_xlsx(rows, output)

    print("Evaluation completed successfully.")
    print(f"Excel saved: {output}")

    if debug_dir is not None:
        print(f"Debug masks saved: {debug_dir}")
