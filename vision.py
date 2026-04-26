import cv2
import numpy as np


def extract_features(frame):
    """
    Input: BGR image (OpenCV)
    Output: feature vector (list of floats)
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -------- GREEN (prey) --------
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    prey_visible, prey_x, prey_area = extract_blob(mask_green)

    # -------- RED (predators) --------
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = mask_red1 + mask_red2

    red_visible, red_x, red_area = extract_blob(mask_red)

    return [
        prey_visible,
        prey_x,
        prey_area,
        red_visible,
        red_x,
        red_area
    ]


def extract_blob(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, 0.0, 0.0

    # largest blob
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # filter noise
    if area < 50:
        return 0.0, 0.0, 0.0

    M = cv2.moments(c)

    if M["m00"] == 0:
        return 0.0, 0.0, 0.0

    cx = int(M["m10"] / M["m00"])

    width = mask.shape[1]

    # normalize to [-1, 1]
    x_norm = (cx - width / 2) / (width / 2)

    # normalize area (important for NN stability)
    area_norm = min(area / (width * mask.shape[0]), 1.0)

    return 1.0, x_norm, area_norm