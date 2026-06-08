import math
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from vision import extract_features


class Camera360Observation:
    """
    Four camera streams are merged into one panorama-like image, then the same
    vision extractor is applied once. The final controller input is compressed
    to the same conceptual 3 inputs as the privileged paper controller:

    [r_like, delta_theta_like, d_like]

    The panorama keeps the front camera in the middle so x ~= 0 corresponds to
    straight ahead. Rear camera is split across the left/right edges to reduce
    the discontinuity at the back of the robot.
    """

    def __init__(self, node, robot_name, predator_count):
        self.node = node
        self.robot_name = robot_name
        self.predator_count = predator_count
        self.bridge = CvBridge()

        self.camera_names = ["camera", "camera_left", "camera_right", "camera_rear"]
        self.frames = {name: None for name in self.camera_names}

        for cam in self.camera_names:
            node.create_subscription(
                Image,
                f"/{robot_name}/{cam}/image_raw",
                lambda msg, cam_name=cam: self.image_callback(msg, cam_name),
                10,
            )

    def image_callback(self, msg, cam_name):
        self.frames[cam_name] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    @staticmethod
    def _sqrt_area(area):
        return math.sqrt(max(0.0, float(area)))

    def build_panorama(self):
        front = self.frames["camera"]
        left = self.frames["camera_left"]
        right = self.frames["camera_right"]
        rear = self.frames["camera_rear"]

        if front is None:
            return None

        # Allow the first few frames to arrive gradually. Missing side cameras are black.
        h, w = front.shape[:2]

        def ensure(frame):
            if frame is None:
                return np.zeros_like(front)
            if frame.shape[:2] != (h, w):
                return cv2.resize(frame, (w, h))
            return frame

        left = ensure(left)
        right = ensure(right)
        rear = ensure(rear)

        # Split rear camera so the back direction sits at both panorama edges.
        rear_left_half = rear[:, : w // 2]
        rear_right_half = rear[:, w // 2 :]

        # Approximate angular order, left-to-right:
        # rear seam | right | front | left | rear seam
        # Front camera remains centered in the panorama.
        return np.hstack([rear_left_half, right, front, left, rear_right_half])

    def get_features(self):
        panorama = self.build_panorama()
        if panorama is None:
            return None

        prey_visible, prey_x, prey_area, red_visible, red_x, red_area = extract_features(panorama)

        if red_visible > 0.0:
            r_like = -float(red_x) * self._sqrt_area(red_area)
        else:
            r_like = 0.0

        if prey_visible > 0.0:
            delta_theta_like = -float(prey_x)
            d_like = self._sqrt_area(prey_area)
        else:
            delta_theta_like = 0.0
            d_like = 0.0

        return np.array([r_like, delta_theta_like, d_like], dtype=np.float32)