# Building Footprint Extraction

Generate building footprint contours from a georeferenced point cloud and building annotations, then evaluate predicted outlines against a reference raster. The occupancy methods combine a rasterized polygon prior with point density and vertical layer consistency.

## Installation

Use Python 3.10 or newer and run these commands from the repository root. Python 3.12 is the version used for the package build and CLI checks.

Install the runtime dependencies listed in [requirements.txt](requirements.txt), then install the project to register the `footprint` command:

```sh
python -m pip install -r requirements.txt
python -m pip install .
```

To install only one workflow and its dependencies, use the corresponding package extra:

```sh
python -m pip install ".[occupancy]"
python -m pip install ".[metrics]"
```

The package uses `setuptools>=68` as its build backend. With the default pip build isolation, pip installs this build dependency automatically. Confirm the CLI is available with `footprint --help`. You can also use `python -m footprint_tools --help`.

## Extract building footprints

```sh
footprint occupancy --ply scene.ply --annotations buildings.json --geotiff orthophoto.tif --output-dir output --method mask
```

- The PLY and GeoTIFF must use the same projected coordinate system in metres. Use a north-up, unrotated raster. The tool uses the coordinates stored in the inputs without reprojection.
- Annotations must reference pixel coordinates in the full GeoTIFF. The reader accepts LabelMe-style `shapes` with `label`, `points` and optional `group_id` (or `group`), or `objects` with a building label, `segmentation` and optional `group`. Accepted shapes use the label `building`. Group IDs may contain ASCII letters, digits, underscores and hyphens.
- Each run writes a timestamped subdirectory containing per-group raster and contour outputs, `run_config_all_groups.json`, and `summary_all_groups.json`. Check the summary for skipped groups. The command returns a nonzero status when any group fails or is skipped.
- ROI coordinates use `x = min_x + col * step` and `y = max_y - row * step`. Contour outputs retain the original workflow's pixel coordinate convention. Keep the ROI metadata with the outputs.

| Method | Source implementation | Description |
| --- | --- | --- |
| `mask` | `GPT_occmap_whole.py` | Mask-dominant occupancy with outer and hole contours |
| `mask-baseline` | `occmap_duibi1.py` | Mask-dominant comparison variant |
| `pointcloud` | `occmap_duibi2.py` | Point-cloud-dominant occupancy with outer contours |
| `pointcloud-postprocessed` | `occmap_duibi3.py` | Point-cloud-dominant occupancy with additional contour postprocessing |

### Experiment configuration

Each method has its own defaults in [configs/default.json](src/footprint_tools/configs/default.json). Pass a JSON object containing only the fields to override:

```sh
footprint occupancy --ply scene.ply --annotations buildings.json --geotiff orthophoto.tif --output-dir output --method mask --config examples/mask-config.json
```

The example changes whether per-building GeoTIFF crops are exported. Distances such as `step` and `layer_thickness` use metres. The effective parameters are recorded in the run configuration.

## Evaluate outline predictions

```sh
footprint evaluate --ground-truth reference.png --prediction model-a.png model-b.png --output output/metrics.xlsx --debug-dir output/debug
```

Use white-background PNGs with closed black outlines on the same pixel grid. The evaluator fills the outlines, computes overlap and boundary metrics, and writes an XLSX report with one row per prediction and an average row. The boundary IoU measures pixel overlap. Use `--allow-resize` only when both images represent the same extent and differ in resolution. See `footprint evaluate --help` for the threshold and debug options.

## Try the synthetic example

Run from the repository root after installing both sets of dependencies:

```sh
python examples/generate_demo.py output/demo
footprint occupancy --ply output/demo/scene.ply --annotations output/demo/buildings.json --geotiff output/demo/orthophoto.tif --output-dir output/demo/occupancy --method mask --config examples/mask-config.json
footprint evaluate --ground-truth output/demo/reference.png --prediction output/demo/prediction.png --output output/demo/metrics.xlsx
```

The generator creates a projected GeoTIFF, a synthetic PLY, a building annotation, and two aligned outline images. If you only installed the evaluation dependencies, add `--outlines-only` to the generator command and run the evaluation command.

## Project layout

```text
footprint-tools/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── examples/
│   ├── generate_demo.py
│   └── mask-config.json
└── src/footprint_tools/
    ├── algorithms/  # Four occupancy and contour methods
    ├── configs/     # Packaged method defaults and override validation
    ├── datasets/    # Annotation and GeoTIFF readers
    ├── tools/       # Extraction and evaluation workflows
    ├── utils/       # Geometry, image output and metrics
    └── cli.py
```

## Release scope

This package contains the grouped occupancy methods and PNG outline evaluation. Point-cloud registration, GIS evaluation, and other site-specific experiments remain in the source workspace. Annotation software, datasets, and model weights are external inputs.

## License

Released under the [MIT License](LICENSE).
