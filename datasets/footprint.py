"""Candidate tensors and pseudo-labels from equations 19–21."""

import json

import numpy as np


def read_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = data["samples"]
    ids = [row["group_id"] for row in rows]
    if len(ids) != len(set(ids)) or not rows:
        raise ValueError("Manifest must contain samples and unique group IDs")
    return data


def load_pair(root, row):
    path = (root / row["arrays"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Candidate array path must be inside the manifest directory")
    with np.load(path, allow_pickle=False) as archive:
        mask = archive["mask"].astype(np.float32)
        occupancy = archive["occupancy"].astype(np.float32)
    if mask.ndim != 2 or mask.shape != occupancy.shape or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"Invalid mask/occupancy shape or binary mask in {path}")
    if not np.isfinite(occupancy).all() or np.any((occupancy < 0) | (occupancy > 1)):
        raise ValueError(f"Invalid occupancy range in {path}")
    return mask, occupancy


def split_training(root, rows, assignments):
    """Use building instance membership, including assignments for false candidates."""
    groups = {"train": [], "validation": [], "test": []}
    instances = {}
    archives = {}
    for row in rows:
        key = row["group_id"]
        record = assignments[key]
        instance = record["instance_id"]
        split = record["split"]
        if not isinstance(instance, str) or not instance or split not in groups:
            raise ValueError(f"Invalid instance_id or split for candidate {key}")
        if instances.setdefault(instance, split) != split:
            raise ValueError(f"Instance {instance} appears in multiple splits")
        archive = (root / row["arrays"]).resolve()
        if not archive.is_relative_to(root.resolve()) or archives.setdefault(archive, split) != split:
            raise ValueError(f"Candidate {key} has an invalid or cross-split archive path")
        groups[split].append((row, record.get("label")))
    if not groups["train"] or not groups["validation"]:
        raise ValueError("Provide both training and validation candidates")
    return groups


def tensor_pair(mask, occupancy, size):
    import torch
    from torch.nn import functional as F

    h, w = mask.shape
    ratio = size / max(h, w)
    shape = max(1, round(h * ratio)), max(1, round(w * ratio))
    image = torch.from_numpy(mask)[None, None]
    lidar = torch.from_numpy(occupancy)[None, None]
    image = F.interpolate(image, size=shape, mode="nearest")[0]
    lidar = F.interpolate(lidar, size=shape, mode="bilinear", align_corners=False)[0]
    dh, dw = size - shape[0], size - shape[1]
    pad = (dw // 2, (dw + 1) // 2, dh // 2, (dh + 1) // 2)
    return F.pad(image, pad), F.pad(lidar, pad)


class CandidateDataset:
    def __init__(self, root, records, size):
        self.root, self.records, self.size = root, records, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row, label = self.records[index]
        image, lidar = tensor_pair(*load_pair(self.root, row), self.size)
        return image, lidar, float(label or 0)
