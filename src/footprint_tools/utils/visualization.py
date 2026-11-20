import os

import cv2
import numpy as np


def save_gray_preview(path, image):
    image = np.asarray(image)

    if image.max() > 0:
        out = image.astype(np.float32) / image.max()
    else:
        out = image.astype(np.float32)

    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    write_image(path, out)


def write_image(path, image):
    """Encode with OpenCV and write through Python for Unicode path support."""
    ok, encoded = cv2.imencode(os.path.splitext(path)[1], image)
    if not ok:
        raise OSError(f"Could not encode image: {path}")
    with open(path, "wb") as stream:
        stream.write(encoded.tobytes())
