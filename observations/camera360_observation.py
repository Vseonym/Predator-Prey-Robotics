import math
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from vision import extract_features


class Camera360Observation:
    def __init__(self, node, robot_name, predator_count):
        self.node = node
        self.robot_name = robot_name
        self.predator_count = predator_count
        self.bridge = CvBridge()

        self.camera_names = ["camera", "camera_left", "camera_right", "camera_rear"]
        self.camera_features = {name: [0.0] * 6 for name in self.camera_names}
        self.camera_seen = {name: False for name in self.camera_names}

        self.prox_names = ["center", "center_left", "center_right", "left", "right"]
        self.prox_values = {name: 0.0 for name in self.prox_names}

        for cam in self.camera_names:
            node.create_subscription(
                Image,
                f"/{robot_name}/{cam}/image_raw",
                lambda msg, cam_name=cam: self.image_callback(msg, cam_name),
                10,
            )

        for prox_name in self.prox_names:
            node.create_subscription(
                LaserScan,
                f"/{robot_name}/proximity/{prox_name}",
                lambda msg, prox=prox_name: self.proximity_callback(msg, prox),
                10,
            )

    def image_callback(self, msg, cam_name):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.camera_features[cam_name] = extract_features(frame)
        self.camera_seen[cam_name] = True

    def proximity_callback(self, msg, prox_name):
        if msg.range_max <= msg.range_min:
            self.prox_values[prox_name] = 0.0
            return
        valid = [r for r in msg.ranges if not math.isnan(r) and not math.isinf(r) and msg.range_min <= r <= msg.range_max]
        if not valid:
            self.prox_values[prox_name] = 0.0
            return
        closest = min(valid)
        closeness = 1.0 - ((closest - msg.range_min) / (msg.range_max - msg.range_min))
        self.prox_values[prox_name] = max(0.0, min(1.0, closeness))

    def role_value(self):
        if self.predator_count <= 1:
            return 0.0
        try:
            idx = int(self.robot_name.split("_")[-1])
        except ValueError:
            idx = 0
        idx = max(0, min(self.predator_count - 1, idx))
        return -1.0 + (2.0 * idx / (self.predator_count - 1))

    def get_features(self):
        # Wait until at least front camera has arrived; other cameras can be zeros early.
        if not self.camera_seen["camera"]:
            return None
        features = []
        for cam in self.camera_names:
            features.extend(self.camera_features[cam])
        features.extend([
            self.prox_values["center"],
            self.prox_values["center_left"],
            self.prox_values["center_right"],
            self.prox_values["left"],
            self.prox_values["right"],
            self.role_value(),
        ])
        return np.array(features, dtype=np.float32)