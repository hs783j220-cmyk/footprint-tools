"""Command-line entry points for building footprint experiments."""

import argparse
from importlib import import_module
from pathlib import Path


METHODS = {
    "mask": "occupancy",
    "mask-baseline": "mask_baseline",
    "pointcloud": "pointcloud",
    "pointcloud-postprocessed": "pointcloud_postprocessed",
}


def input_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and evaluate building footprints.")
    commands = parser.add_subparsers(dest="command", required=True)

    occupancy = commands.add_parser("occupancy", help="Generate contours from point clouds and annotations.")
    occupancy.add_argument("--ply", required=True, type=input_file)
    occupancy.add_argument("--annotations", required=True, type=input_file)
    occupancy.add_argument("--geotiff", required=True, type=input_file)
    occupancy.add_argument("--output-dir", required=True, type=Path)
    occupancy.add_argument("--method", choices=METHODS, default="mask")

    occupancy.add_argument('--config', type=input_file, help='JSON file overriding the selected method defaults.')

    metrics = commands.add_parser("evaluate", help="Compare outline images and write an Excel report.")
    metrics.add_argument("--ground-truth", required=True, type=input_file)
    metrics.add_argument("--prediction", required=True, nargs="+", type=input_file, metavar="PNG")
    metrics.add_argument("--output", required=True, type=Path, metavar="XLSX")
    metrics.add_argument("--threshold", type=int, default=80, help="Maximum RGB channel value for black lines (default: 80).")
    metrics.add_argument("--allow-resize", action="store_true", help="Resize mismatched predictions to the reference grid.")
    metrics.add_argument("--debug-dir", type=Path, help="Save extracted lines, areas and boundaries.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if not 0 <= args.threshold <= 255:
            parser.error("--threshold must be between 0 and 255")
        if args.output.suffix.lower() != ".xlsx":
            parser.error("--output must have the .xlsx suffix")

    try:
        if args.command == "occupancy":
            from .configs import load_parameters
            from .tools.inference import run

            parameters = load_parameters(args.method, args.config)
            method = import_module(f".algorithms.{METHODS[args.method]}", package=__package__)
            summary = run(
                args.ply, args.annotations, args.geotiff, args.output_dir,
                method=args.method,
                process_one_group=method.process_one_group,
                parameters=parameters,
            )
            return int(summary["failed_or_skipped_group_count"] > 0)

        from .tools.evaluation import evaluate

        evaluate(
            args.ground_truth,
            args.prediction,
            args.output,
            threshold=args.threshold,
            allow_resize=args.allow_resize,
            debug_dir=args.debug_dir,
        )
        return 0
    except ImportError as exc:
        extra = "occupancy" if args.command == "occupancy" else "metrics"
        parser.exit(1, f"Missing dependency: {exc}. Install with python -m pip install '.[{extra}]'.\n")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
