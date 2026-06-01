import cv2
import numpy as np


def register_images(pre_img, post_img):

    # Convert to grayscale

    pre_gray = cv2.cvtColor(
        pre_img,
        cv2.COLOR_RGB2GRAY
    )

    post_gray = cv2.cvtColor(
        post_img,
        cv2.COLOR_RGB2GRAY
    )

    # Convert to float32

    pre_gray = pre_gray.astype(np.float32) / 255.0
    post_gray = post_gray.astype(np.float32) / 255.0

    # Define motion model

    warp_mode = cv2.MOTION_EUCLIDEAN

    # Initialize warp matrix

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Number of iterations

    number_of_iterations = 5000

    # Termination criteria

    termination_eps = 1e-8

    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps
    )

    try:

        # ECC registration

        cc, warp_matrix = cv2.findTransformECC(
            pre_gray,
            post_gray,
            warp_matrix,
            warp_mode,
            criteria
        )

        height, width = pre_img.shape[:2]

        aligned = cv2.warpAffine(
            post_img,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        return aligned

    except cv2.error:

        # If registration fails,
        # return original image

        return post_img