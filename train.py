"""Train and score the confidence network using a fixed building split."""

import hashlib
import json

import numpy as np

from datasets.footprint import CandidateDataset, load_pair, read_manifest, split_training
from losses import classification_loss, pseudo_label


def choose_threshold(scores, targets):
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if scores.shape != targets.shape or set(targets.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Threshold calibration requires finite scores and both reference classes")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    true_positives = np.cumsum(targets[order])
    # A valid operating point includes all candidates tied at its score.
    boundaries = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    f1 = 2 * true_positives[boundaries] / (boundaries + 1 + targets.sum())
    index = int(np.argmax(f1))
    return float(sorted_scores[boundaries[index]]), float(f1[index])


def run(manifest, assignments, output, tau_positive, tau_negative, eta_positive, eta_negative,
        epochs, batch_size, learning_rate, input_size, seed, device, model_width, dropout, token_grid):
    import torch
    from torch.utils.data import DataLoader

    from ..models.fusion import CandidateNetwork

    if not (0 <= tau_negative < tau_positive <= 1 and 0 <= eta_negative < eta_positive <= 1):
        raise ValueError("Pseudo-label negative thresholds must be below positive thresholds in [0, 1]")
    if epochs < 1 or batch_size < 1 or input_size < 32 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid training hyperparameters")
    data = read_manifest(manifest)
    assignment_bytes = assignments.read_bytes()
    groups = split_training(manifest.parent, data["samples"], json.loads(assignment_bytes))
    campus_manifest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    campus_assignments = hashlib.sha256(assignment_bytes).hexdigest()
    training = []
    for row, _ in groups["train"]:
        label = pseudo_label(*load_pair(manifest.parent, row), tau_positive, tau_negative,
                             eta_positive, eta_negative)
        if label is not None:
            training.append((row, label))
    validation = groups["validation"]
    if {value for _, value in training} != {0, 1}:
        raise ValueError("Training pseudo-labels must include both classes")
    if any(type(value) is not int or value not in (0, 1) for _, value in validation) or {v for _, v in validation} != {0, 1}:
        raise ValueError("Provide independent 0/1 reference labels for both classes in validation")

    if model_width < 1 or token_grid < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid model hyperparameters")
    model_settings = dict(width=model_width, dropout=dropout, token_grid=token_grid)
    torch.manual_seed(seed)
    model = CandidateNetwork(**model_settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = classification_loss()
    train_loader = DataLoader(CandidateDataset(manifest.parent, training, input_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CandidateDataset(manifest.parent, validation, input_size), batch_size=batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for image, lidar, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(image.to(device), lidar.to(device))
            loss = loss_fn(logits, labels.to(device=device, dtype=logits.dtype))
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(labels)

        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for image, lidar, target in val_loader:
                scores.extend(model(image.to(device), lidar.to(device)).sigmoid().cpu().tolist())
                labels.extend(target.tolist())
        threshold, f1 = choose_threshold(scores, labels)
        print(f"Epoch {epoch}/{epochs} BCE={loss_total / len(training):.5f} validation_F1={f1:.4f}")
        if f1 > best:
            best = f1
            state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            checkpoint = {"state_dict": state, "threshold": threshold,
                          "model_settings": model_settings, "input_size": input_size, "validation_f1": f1, "epoch": epoch,
                          "occupancy_parameters": data["parameters"],
                          "campus_manifest_sha256": campus_manifest,
                          "campus_assignments_sha256": campus_assignments,
                          "pseudo_label_thresholds": [tau_positive, tau_negative, eta_positive, eta_negative]}
            temporary = output.with_suffix(output.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(output)
    print(f"Saved validation-selected weights and threshold to {output}")
