"""Resolve method defaults and validate experiment overrides."""

import json
import math
from importlib.resources import files
from pathlib import Path


def load_parameters(method: str, config_path: Path | None = None) -> dict:
    defaults = json.loads(files(__package__).joinpath("default.json").read_text(encoding="utf-8"))
    parameters = defaults[method]
    if config_path is None:
        return parameters

    overrides = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(overrides, dict):
        raise ValueError("The configuration must be a JSON object of parameter overrides.")
    for key, value in overrides.items():
        if key not in parameters:
            raise ValueError(f"Unknown parameter for {method}: {key}")
        expected = type(parameters[key])
        valid_type = type(value) is expected or (expected is float and type(value) is int)
        if not valid_type:
            raise ValueError(f"{key} must have type {expected.__name__}")
        if expected in (int, float) and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be finite and nonnegative")

    parameters.update(overrides)
    for key in ("step", "layer_thickness", "soft_sigma_m", "outline_thickness", "main_direction_angle_bin_deg"):
        if parameters[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= parameters["weak_consistency_thresh"] <= parameters["strong_consistency_thresh"] <= 1:
        raise ValueError("Consistency thresholds must be ordered and within [0, 1]")
    if parameters["main_direction_peak_ratio"] > 1:
        raise ValueError("main_direction_peak_ratio must be within [0, 1]")
    for key in ("close_kernel_size", "open_kernel_size"):
        if key in parameters and parameters[key] < 1:
            raise ValueError(f"{key} must be at least 1")
    if method == "pointcloud-postprocessed":
        if parameters["simplify_epsilon_min_px"] > parameters["simplify_epsilon_max_px"]:
            raise ValueError("Simplification epsilon bounds must be ordered")
    return parameters
