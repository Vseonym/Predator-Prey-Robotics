import math
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from vision import extract_features


class FPVObservation:
    """
    Front-camera observation compressed into the same conceptual 3 inputs
    as the privileged paper controller:

    [r_like, delta_theta_like, d_like]

    r_like:
        signed visual closeness of the nearest/strongest red teammate blob.
        Positive means teammate is visually on the left, negative on the right.
    delta_theta_like:
        estimated prey bearing from the front camera, normalized to [-1, 1].
        0 when prey is not visible.
    d_like:
        visual prey closeness proxy, sqrt(green_blob_area). 0 when not visible.
    """

    def __init__(self, node, robot_name, predator_count):
        self.node = node
        self.robot_name = robot_name
        self.predator_count = predator_count
        self.bridge = CvBridge()
        self.camera_features = None

        node.create_subscription(
            Image,
            f"/{robot_name}/camera_front_sensor/image_raw",
            self.image_callback,
            10,
        )

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.camera_features = extract_features(frame)

    @staticmethod
    def _sqrt_area(area):
        return math.sqrt(max(0.0, float(area)))

    def get_features(self):
        if self.camera_features is None:
            return None

        prey_visible, prey_x, prey_area, red_visible, red_x, red_area = self.camera_features

        if red_visible > 0.0:
            # extract_features: x < 0 means blob is left in image.
            # Privileged signed r uses positive for left, so invert image x.
            r_like = -float(red_x) * self._sqrt_area(red_area)
        else:
            r_like = 0.0

        if prey_visible > 0.0:
            # x < 0 means prey is left in image; privileged angle is positive-left.
            delta_theta_like = -float(prey_x)
            d_like = self._sqrt_area(prey_area)
        else:
            delta_theta_like = 0.0
            d_like = 0.0

        return np.array([r_like, delta_theta_like, d_like], dtype=np.float32)