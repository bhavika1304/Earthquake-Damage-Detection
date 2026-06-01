import numpy as np
import cv2

import numpy as np
import cv2

def create_overlay(image, mask):

    # resize image to match mask size
    h, w = mask.shape[:2]
    image = cv2.resize(image, (w, h))

    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    overlay = image.copy().astype(np.float32)

    colors = {
        1: np.array([255, 255,   0], dtype=np.float32),
        2: np.array([255,   0,   0], dtype=np.float32),
    }

    alpha = 0.55

    for cls, color in colors.items():
        where = mask == cls
        if not where.any():
            continue
        overlay[where] = alpha * color + (1 - alpha) * overlay[where]

    return overlay.clip(0, 255).astype(np.uint8)
