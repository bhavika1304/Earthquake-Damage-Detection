import cv2
import numpy as np


def apply_gaussian_blur(image, kernel_size=5):

    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

    return blurred