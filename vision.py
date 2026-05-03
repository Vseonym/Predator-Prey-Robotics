import cv2
import numpy as np


def clean_mask(mask):
    """
    Reduce small noisy blobs before contour detection.
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def extract_features(frame):
    """
    Input:
        BGR image from OpenCV.

    Output:
        [
            prey_visible,
            prey_x,
            prey_area,
            red_visible,
            red_x,
            red_area
        ]

    prey_x / red_x:
        normalized horizontal offset in [-1, 1]
        -1 = far left
         0 = center
        +1 = far right

    prey_area / red_area:
        normalized blob area in [0, 1]
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -------- GREEN prey --------
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = clean_mask(mask_green)

    prey_visible, prey_x, prey_area = extract_blob(mask_green)

    # -------- RED predators --------
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = mask_red1 + mask_red2
    mask_red = clean_mask(mask_red)

    red_visible, red_x, red_area = extract_blob(mask_red)

    return [
        prey_visible,
        prey_x,
        prey_area,
        red_visible,
        red_x,
        red_area,
    ]


def extract_blob(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return 0.0, 0.0, 0.0

    # Largest blob only
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # Filter small noise
    if area < 50:
        return 0.0, 0.0, 0.0

    moments = cv2.moments(c)

    if moments["m00"] == 0:
        return 0.0, 0.0, 0.0

    cx = int(moments["m10"] / moments["m00"])

    height, width = mask.shape

    x_norm = (cx - width / 2) / (width / 2)
    area_norm = min(area / (width * height), 1.0)

    return 1.0, x_norm, area_norm