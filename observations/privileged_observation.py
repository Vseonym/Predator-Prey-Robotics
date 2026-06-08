import math
import numpy as np
from gazebo_msgs.msg import ModelStates
from model_state_utils import model_pose_dict, pose_xy_yaw, angle_wrap


class PrivilegedObservation:
    def __init__(self, node, robot_name, predator_count, model_states_topic="/model_states"):
        self.node = node
        self.robot_name = robot_name
        self.predator_count = predator_count
        self.model_states = None
        node.create_subscription(ModelStates, model_states_topic, self.model_states_callback, 10)

    def model_states_callback(self, msg):
        self.model_states = msg

    def get_features(self):
        if self.model_states is None:
            return None

        poses = model_pose_dict(self.model_states)
        if self.robot_name not in poses or "prey_0" not in poses:
            return None

        x, y, yaw = pose_xy_yaw(poses[self.robot_name])
        prey_x, prey_y, _ = pose_xy_yaw(poses["prey_0"])

        dx = prey_x - x
        dy = prey_y - y
        d = math.hypot(dx, dy)
        prey_angle = math.atan2(dy, dx)
        delta_theta = angle_wrap(prey_angle - yaw) / math.pi

        nearest_signed = 0.0
        nearest_dist = float("inf")

        for i in range(self.predator_count):
            other = f"predator_{i}"
            if other == self.robot_name or other not in poses:
                continue
            ox, oy, _ = pose_xy_yaw(poses[other])
            odx = ox - x
            ody = oy - y
            dist = math.hypot(odx, ody)
            if dist < nearest_dist:
                nearest_dist = dist
                # sign by relative bearing: left positive, right negative.
                bearing = angle_wrap(math.atan2(ody, odx) - yaw)
                sign = 1.0 if bearing >= 0.0 else -1.0
                nearest_signed = sign * dist

        if not math.isfinite(nearest_dist):
            nearest_signed = 0.0

        # Normalize distance roughly by arena diagonal of 2x2 square.
        return np.array([nearest_signed / 2.828, delta_theta, d / 2.828], dtype=np.float32)