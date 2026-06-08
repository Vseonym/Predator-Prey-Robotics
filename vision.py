import cv2
import numpy as np


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def extract_blob(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 50:
        return 0.0, 0.0, 0.0

    moments = cv2.moments(c)
    if moments["m00"] == 0:
        return 0.0, 0.0, 0.0

    cx = int(moments["m10"] / moments["m00"])
    height, width = mask.shape
    x_norm = (cx - width / 2) / (width / 2)
    area_norm = min(area / (width * height), 1.0)
    return 1.0, float(x_norm), float(area_norm)


def extract_features(frame):
    """
    Returns [prey_visible, prey_x, prey_area, red_visible, red_x, red_area].
    Green is prey. Red is predator teammate.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = clean_mask(cv2.inRange(hsv, lower_green, upper_green))
    prey_visible, prey_x, prey_area = extract_blob(mask_green)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = clean_mask(mask_red)
    red_visible, red_x, red_area = extract_blob(mask_red)

    return [prey_visible, prey_x, prey_area, red_visible, red_x, red_area]