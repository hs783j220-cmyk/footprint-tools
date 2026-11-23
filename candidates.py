"""Generate and refine automatic SAM prompts from paper §3.2."""

import json
import math

import cv2
import numpy as np


def prompts(width, height, spacing, stride, scales):
    points = [(x, y) for y in range(0, height, spacing) for x in range(0, width, spacing)]
    boxes = []
    for size in sorted({(min(s, width), min(s, height)) for s in scales}):
        w, h = size
        xs = sorted(set(range(0, width - w + 1, stride)) | {width - w})
        ys = sorted(set(range(0, height - h + 1, stride)) | {height - h})
        boxes.extend((x, y, x + w, y + h) for y in ys for x in xs)
    return points, boxes


def mask_iou(a, b):
    ax, ay, mask_a = a
    bx, by, mask_b = b
    left, top = max(ax, bx), max(ay, by)
    right = min(ax + mask_a.shape[1], bx + mask_b.shape[1])
    bottom = min(ay + mask_a.shape[0], by + mask_b.shape[0])
    if right <= left or bottom <= top:
        return 0.0
    patch_a = mask_a[top-ay:bottom-ay, left-ax:right-ax]
    patch_b = mask_b[top-by:bottom-by, left-bx:right-bx]
    intersection = np.count_nonzero(patch_a & patch_b)
    return intersection / (mask_a.sum() + mask_b.sum() - intersection)


def refine(proposals, pixel_area, dedup_iou, min_area, max_area, max_aspect, min_compactness):
    unique = []
    for score, mask in sorted(proposals, key=lambda entry: -entry[0]):
        if not any(mask_iou(mask, existing) >= dedup_iou for existing in unique):
            unique.append(mask)

    refined = []
    for x, y, cropped in unique:
        contours, _ = cv2.findContours(cropped.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        dimensions = cv2.minAreaRect(contour)[1]
        if min(dimensions) == 0 or perimeter == 0:
            continue
        if not min_area <= area * pixel_area <= max_area:
            continue
        if max(dimensions) / min(dimensions) > max_aspect or 4 * math.pi * area / perimeter**2 < min_compactness:
            continue
        contour = cv2.approxPolyDP(contour, 0.5, True).reshape(-1, 2)
        if len(contour) >= 3:
            refined.append((contour + [x, y]).tolist())
    return refined


def run(geotiff, checkpoint, output, model_type, device, point_spacing, box_stride,
        box_sizes, prompt_batch_size, dedup_iou, min_area, max_area, max_aspect, min_compactness):
    import rasterio
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    if min(point_spacing, box_stride, min(box_sizes), prompt_batch_size) < 1:
        raise ValueError("Prompt parameters must be positive")
    if not (0 < dedup_iou <= 1 and 0 < min_area <= max_area and max_aspect >= 1 and 0 <= min_compactness <= 1):
        raise ValueError("Invalid candidate geometry thresholds")
    with rasterio.open(geotiff) as raster:
        crs = raster.crs
        if crs is None or not crs.is_projected or not math.isclose(crs.linear_units_factor[1], 1.0):
            raise ValueError("The SAM tile must use a projected CRS in metres")
        if raster.count < 3 or any(dtype != "uint8" for dtype in raster.dtypes[:3]):
            raise ValueError("SAM expects three 8-bit RGB bands")
        height, width = raster.height, raster.width
        pixel_area = abs(raster.transform.determinant)
        image = np.ascontiguousarray(np.moveaxis(raster.read([1, 2, 3]), 0, -1))

    model = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device=device).eval()
    predictor = SamPredictor(model)
    predictor.set_image(image)
    point_prompts, box_prompts = prompts(width, height, point_spacing, box_stride, box_sizes)
    proposals = []
    with torch.inference_mode():
        for is_box, values in ((False, point_prompts), (True, box_prompts)):
            for start in range(0, len(values), prompt_batch_size):
                coords = np.asarray(values[start:start + prompt_batch_size], dtype=np.float32)
                if is_box:
                    boxes = predictor.transform.apply_boxes(coords, (height, width))
                    kwargs = {"boxes": torch.as_tensor(boxes, device=device)}
                else:
                    transformed = predictor.transform.apply_coords(coords, (height, width))
                    kwargs = {"point_coords": torch.as_tensor(transformed[:, None, :], device=device),
                              "point_labels": torch.ones((len(coords), 1), dtype=torch.int64, device=device)}
                masks, scores, _ = predictor.predict_torch(
                    kwargs.pop("point_coords", None), kwargs.pop("point_labels", None), **kwargs,
                    multimask_output=not is_box,
                )
                masks, scores = masks.cpu().numpy(), scores.cpu().numpy()
                for variants, confidence in zip(masks, scores, strict=True):
                    best = int(confidence.argmax())
                    binary = variants[best]
                    ys, xs = np.nonzero(binary)
                    if len(xs):
                        x, y = int(xs.min()), int(ys.min())
                        cropped = binary[y:int(ys.max())+1, x:int(xs.max())+1].copy()
                        proposals.append((float(confidence[best]), (x, y, cropped)))

    contours = refine(proposals, pixel_area, dedup_iou, min_area, max_area, max_aspect, min_compactness)
    payload = {"imagePath": geotiff.name, "imageWidth": width, "imageHeight": height,
               "imageData": None, "flags": {}, "shapes": [
        {"label": "building", "group_id": i, "shape_type": "polygon",
         "points": contour, "flags": {}, "description": "Unverified SAM candidate"}
        for i, contour in enumerate(contours, 1)
    ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(contours)} SAM candidates")
