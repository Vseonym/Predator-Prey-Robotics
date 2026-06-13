import math
import numpy as np

from model_state_utils import OdomStore, angle_wrap


class PrivilegedObservation:
    def __init__(self, node, robot_name, predator_count, model_states_topic=None):
        self.node = node
        self.robot_name = robot_name
        self.predator_count = predator_count
        self.predator_names = [f"predator_{i}" for i in range(predator_count)]
        self.all_robot_names = self.predator_names + ["prey_0"]
        self.odom_store = OdomStore(node, self.all_robot_names)

    def get_features(self):
        if not self.odom_store.has_all(self.all_robot_names):
            return None

        self_pose = self.odom_store.xy_yaw(self.robot_name)
        prey_pose = self.odom_store.xy_yaw("prey_0")
        if self_pose is None or prey_pose is None:
            return None

        x, y, yaw = self_pose
        prey_x, prey_y, _ = prey_pose

        dx = prey_x - x
        dy = prey_y - y
        d = math.hypot(dx, dy)
        prey_angle = math.atan2(dy, dx)
        delta_theta = angle_wrap(prey_angle - yaw) / math.pi

        nearest_signed = 0.0
        nearest_dist = float("inf")

        for other in self.predator_names:
            if other == self.robot_name:
                continue
            other_pose = self.odom_store.xy_yaw(other)
            if other_pose is None:
                continue
            ox, oy, _ = other_pose
            odx = ox - x
            ody = oy - y
            dist = math.hypot(odx, ody)
            if dist < nearest_dist:
                nearest_dist = dist
                bearing = angle_wrap(math.atan2(ody, odx) - yaw)
                sign = 1.0 if bearing >= 0.0 else -1.0
                nearest_signed = sign * dist

        if not math.isfinite(nearest_dist):
            nearest_signed = 0.0

        # Normalize distances by approximate 2m x 2m arena diagonal.
        return np.array([nearest_signed / 2.828, delta_theta, d / 2.828], dtype=np.float32)