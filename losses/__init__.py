"""Candidate pseudo-labels and binary classification loss."""

import numpy as np


def pseudo_label(mask, occupancy, tau_positive, tau_negative, eta_positive, eta_negative):
    mask_area = mask.sum(dtype=np.float64)
    if mask_area == 0:
        raise ValueError("Candidate mask is empty")
    shared = np.minimum(mask, occupancy).sum(dtype=np.float64)
    iou = shared / np.maximum(mask, occupancy).sum(dtype=np.float64)
    support = (mask * occupancy).sum(dtype=np.float64) / mask_area
    if iou >= tau_positive and support >= eta_positive:
        return 1
    if iou <= tau_negative or support <= eta_negative:
        return 0
    return None


def classification_loss():
    from torch.nn import BCEWithLogitsLoss

    return BCEWithLogitsLoss()
