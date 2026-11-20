"""Generate synthetic inputs for the README quick start."""

import argparse
import json
from pathlib import Path


def write_outlines(output):
    from PIL import Image, ImageDraw

    for name, offset in (("reference.png", 0), ("prediction.png", 3)):
        image = Image.new("L", (64, 64), 255)
        ImageDraw.Draw(image).rectangle((16 + offset, 16, 48 + offset, 48), outline=0)
        image.save(output / name)


def write_occupancy_inputs(output):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(500000, 3500000, 0.1, 0.1)
    with rasterio.open(output / "orthophoto.tif", "w", driver="GTiff",
                       width=128, height=128, count=1, dtype="uint8",
                       crs="EPSG:32650", transform=transform) as raster:
        raster.write(np.zeros((128, 128), dtype=np.uint8), 1)

    shape = {"label": "building", "group_id": 1,
             "points": [[24, 24], [104, 24], [104, 104], [24, 104]]}
    (output / "buildings.json").write_text(json.dumps({"shapes": [shape]}), encoding="utf-8")

    cells = range(24, 104)
    heights = (1.0, 1.5, 2.0, 2.5, 3.0)
    with (output / "scene.ply").open("w", encoding="ascii") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(cells)**2 * len(heights)}\n"
            "property double x\nproperty double y\nproperty double z\nend_header\n"
        )
        for row in cells:
            for col in cells:
                x, y = transform * (col + 0.5, row + 0.5)
                for height in heights:
                    stream.write(f"{x:.3f} {y:.3f} {height:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--outlines-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outlines(args.output_dir)
    if not args.outlines_only:
        write_occupancy_inputs(args.output_dir)


if __name__ == "__main__":
    main()
