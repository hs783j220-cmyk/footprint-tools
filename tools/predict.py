"""Score unseen candidates with a fixed checkpoint and validation threshold."""

import hashlib
import json

from datasets.footprint import read_manifest, split_training
from models.fusion import CandidateNetwork
from datasets.footprint import CandidateDataset


def run(manifest, checkpoint, output, batch_size, device, assignments):
    import torch
    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data = read_manifest(manifest)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["occupancy_parameters"] != data["parameters"]:
        raise ValueError("Use the training occupancy configuration for inference")
    settings = state.get("model_settings")
    if settings is None:
        from cli import load_config

        defaults = load_config(None)["train"]
        settings = dict(width=defaults["model_width"], dropout=defaults["dropout"], token_grid=defaults["token_grid"])
    model = CandidateNetwork(**settings).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    campus = manifest_hash == state["campus_manifest_sha256"]
    records = [(row, None) for row in data["samples"]]
    if campus:
        if assignments is None:
            raise ValueError("Supply the original campus split with --assignments")
        assignment_bytes = assignments.read_bytes()
        if hashlib.sha256(assignment_bytes).hexdigest() != state["campus_assignments_sha256"]:
            raise ValueError("Campus split differs from the split used for training")
        records = split_training(manifest.parent, data["samples"], json.loads(assignment_bytes))["test"]
    elif assignments is not None:
        raise ValueError("Provide --assignments only with the original campus manifest")
    if not records:
        raise ValueError("No candidates selected for prediction")
    loader = DataLoader(CandidateDataset(manifest.parent, records, state["input_size"]), batch_size=batch_size)
    scores = []
    with torch.no_grad():
        for image, lidar, _ in loader:
            scores.extend(model(image.to(device), lidar.to(device)).sigmoid().cpu().tolist())
    threshold = state["threshold"]
    report = {"manifest_sha256": manifest_hash,
              "threshold": threshold, "candidates": [
        {"group_id": row["group_id"], "confidence": score, "accepted": score >= threshold}
        for (row, _), score in zip(records, scores, strict=True)
    ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
