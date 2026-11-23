"""Command-line entry points for the staged footprint workflow."""

import argparse
from importlib import import_module
from importlib.resources import files
from pathlib import Path

import yaml


# Module, required inputs, and experimental parameter types.
STAGES = {
    "register": ("registration", "ply controls", "threshold"),
    "sam-candidates": ("candidates", "geotiff checkpoint",
        "model_type:str device:str box_sizes:list point_spacing:int box_stride:int prompt_batch_size:int "
        "dedup_iou min_area max_area max_aspect min_compactness"),
    "prepare-samples": ("prepare_samples", "ply annotations geotiff",
        "ground_dem:path preview:bool min_points:int max_slices:int buffer step interval ground_clearance:optional"),
    "train": ("train", "manifest assignments",
        "device:str tau_positive tau_negative eta_positive eta_negative epochs:int batch_size:int "
        "input_size:int seed:int model_width:int token_grid:int learning_rate dropout"),
    "predict": ("predict", "manifest checkpoint", "assignments:path device:str batch_size:int"),
    "extract": ("extract_footprints", "manifest confidence",
        "occupancy_threshold component_overlap min_area closing max_hole snap_angle max_shift dilation simplify"),
    "evaluate": ("test", "prediction reference", "resolution id_field:str"),
}


def load_config(path):
    config = yaml.safe_load(files("configs").joinpath("default.yaml").read_text(encoding="utf-8"))
    if path is not None:
        overrides = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if not isinstance(overrides, dict):
            raise ValueError("Configuration must be a YAML mapping")
        for stage, values in overrides.items():
            if stage not in STAGES or not isinstance(values, dict):
                raise ValueError(f"Invalid configuration section: {stage}")
            config.setdefault(stage, {}).update(values)
    return config


def input_file(value):
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def build_parser(config):
    parser = argparse.ArgumentParser(prog="footprint", description="Extract and evaluate building footprints.")
    parser.add_argument("--config", type=Path, help="YAML overrides, before the subcommand.")
    commands = parser.add_subparsers(dest="command", required=True)
    types = {"str": str, "int": int, "float": float, "optional": float, "list": int, "path": input_file}
    for name, (_, inputs, parameters) in STAGES.items():
        stage = commands.add_parser(name)
        stage.add_argument("--output", required=True, type=Path)
        for key in inputs.split():
            stage.add_argument(f"--{key}", required=True, type=input_file)
        overrides = config.get(name, {})
        keys = set()
        for parameter in parameters.split():
            key, _, kind = parameter.partition(":")
            kind = kind or "float"
            keys.add(key)
            option = "--" + key.replace("_", "-")
            if kind == "bool":
                default = overrides.get(key, False)
                if type(default) is not bool:
                    raise ValueError(f"{name}.{key} must be boolean")
                stage.add_argument(option, action="store_true", default=default)
                continue
            default = overrides.get(key)
            convert = types[kind]
            if kind == "list" and default is not None:
                if not isinstance(default, list) or not default:
                    raise ValueError(f"{name}.{key} must be a nonempty list")
                default = [convert(str(item)) for item in default]
            elif default is not None:
                if kind == "path":
                    raise ValueError(f"Pass {option} as a command-line path")
                default = convert(str(default))
            kwargs = {"nargs": "+"} if kind == "list" else {}
            if key == "model_type":
                kwargs["choices"] = ("vit_b", "vit_l", "vit_h")
                if default is not None and default not in kwargs["choices"]:
                    raise ValueError("Invalid SAM model_type")
            stage.add_argument(option, type=convert, default=default,
                               required=default is None and kind not in {"path", "optional"}, **kwargs)
        if overrides.keys() - keys:
            raise ValueError(f"Unknown parameters for {name}: {sorted(overrides.keys() - keys)}")
    return parser


def main(argv=None):
    preview = argparse.ArgumentParser(prog="footprint", add_help=False)
    preview.add_argument("--config", type=Path)
    settings, _ = preview.parse_known_args(argv)
    try:
        parser = build_parser(load_config(settings.config))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        preview.error(str(exc))
    args = parser.parse_args(argv)
    command = args.command
    options = {key: value for key, value in vars(args).items() if key not in {"command", "config"}}
    try:
        output = options["output"]
        inputs = {value.resolve() for key, value in options.items() if key != "output" and isinstance(value, Path)}
        if args.config is not None:
            inputs.add(args.config.resolve())
        outputs = {output.resolve()}
        if command == "register":
            outputs.add(output.with_suffix(".registration.json").resolve())
        elif command == "train" and output.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("--output must end in .pt or .pth")
        elif command not in {"prepare-samples", "train", "extract"} and output.suffix.lower() != ".json":
            raise ValueError("--output must end in .json")
        if outputs & inputs:
            raise ValueError("Output files must differ from input files")
        import_module(f"tools.{STAGES[command][0]}").run(**options)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
