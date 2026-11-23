"""Optional preview of a candidate mask beside its continuous LiDAR occupancy."""

import numpy as np


def save_candidate_preview(path, mask, occupancy):
    import cv2

    panel = np.concatenate((mask.astype(np.uint8) * 255,
                            np.rint(np.clip(occupancy, 0, 1) * 255).astype(np.uint8)), axis=1)
    ok, encoded = cv2.imencode(".png", panel)
    if not ok:
        raise OSError(f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())
