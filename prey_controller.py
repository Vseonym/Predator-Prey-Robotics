import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates

from model_state_utils import model_pose_dict, pose_xy_yaw, angle_wrap


class GaussianPreyController(Node):
    def __init__(self):
        super().__init__("gaussian_prey_controller")

        self.robot_name = self.declare_parameter("robot_name", "prey_0").value
        self.predator_count = int(self.declare_parameter("predator_count", 3).value)
        self.model_states_topic = self.declare_parameter("model_states_topic", "/model_states").value

        self.start_delay = float(self.declare_parameter("start_delay", 2.0).value)
        self.arena_size = float(self.declare_parameter("arena_size", 2.0).value)
        self.sigma_w = float(self.declare_parameter("sigma_w", 0.2).value)
        self.sigma_p = float(self.declare_parameter("sigma_p", 0.25).value)
        self.alpha = float(self.declare_parameter("alpha", 0.1).value)
        self.max_forward_speed = float(self.declare_parameter("max_forward_speed", 0.12).value)
        self.max_angular_speed = float(self.declare_parameter("max_angular_speed", 1.5).value)
        self.angle_tolerance = float(self.declare_parameter("angle_tolerance", 0.35).value)

        self.model_states = None
        self.start_time = self.now_seconds()

        self.create_subscription(ModelStates, self.model_states_topic, self.model_states_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, f"/{self.robot_name}/cmd_vel", 10)
        self.create_timer(0.1, self.control_loop)

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def model_states_callback(self, msg):
        self.model_states = msg

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def wall_gradient(self, p, wall_position):
        sigma = self.sigma_w
        coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        e = math.exp(-((p - wall_position) ** 2) / (2.0 * sigma * sigma))
        return coeff * e * (-(p - wall_position) / (sigma * sigma))

    def predator_gradient_component(self, prey_x, prey_y, pred_x, pred_y, axis):
        sigma = self.sigma_p
        coeff = 1.0 / (2.0 * math.pi * sigma * sigma)
        e = math.exp(-(((prey_x - pred_x) ** 2) + ((prey_y - pred_y) ** 2)) / (2.0 * sigma * sigma))
        if axis == "x":
            return coeff * e * (-(prey_x - pred_x) / (sigma * sigma))
        return coeff * e * (-(prey_y - pred_y) / (sigma * sigma))

    def danger_gradient(self, prey_x, prey_y, predator_positions):
        half = self.arena_size / 2.0
        walls_x = [-half, half]
        walls_y = [-half, half]

        gx = sum(self.wall_gradient(prey_x, wx) for wx in walls_x)
        gy = sum(self.wall_gradient(prey_y, wy) for wy in walls_y)

        for pred_x, pred_y in predator_positions:
            gx += self.alpha * self.predator_gradient_component(prey_x, prey_y, pred_x, pred_y, "x")
            gy += self.alpha * self.predator_gradient_component(prey_x, prey_y, pred_x, pred_y, "y")

        return gx, gy

    def publish_cmd(self, linear, angular):
        cmd = Twist()
        cmd.linear.x = self.clamp(linear, 0.0, self.max_forward_speed)
        cmd.angular.z = self.clamp(angular, -self.max_angular_speed, self.max_angular_speed)
        self.cmd_pub.publish(cmd)

    def control_loop(self):
        if self.now_seconds() - self.start_time < self.start_delay:
            self.publish_cmd(0.0, 0.0)
            return

        if self.model_states is None:
            self.publish_cmd(0.0, 0.0)
            return

        poses = model_pose_dict(self.model_states)
        if self.robot_name not in poses:
            self.publish_cmd(0.0, 0.0)
            return

        prey_x, prey_y, prey_yaw = pose_xy_yaw(poses[self.robot_name])
        predator_positions = []

        for i in range(self.predator_count):
            name = f"predator_{i}"
            if name in poses:
                x, y, _ = pose_xy_yaw(poses[name])
                predator_positions.append((x, y))

        if not predator_positions:
            self.publish_cmd(0.0, 0.0)
            return

        gx, gy = self.danger_gradient(prey_x, prey_y, predator_positions)

        # Descending gradient points toward lower danger.
        escape_x = -gx
        escape_y = -gy
        norm = math.hypot(escape_x, escape_y)

        if norm < 1e-9:
            self.publish_cmd(0.04, 0.0)
            return

        target_angle = math.atan2(escape_y, escape_x)
        error = angle_wrap(target_angle - prey_yaw)

        angular = self.clamp(2.0 * error, -self.max_angular_speed, self.max_angular_speed)
        linear = self.max_forward_speed if abs(error) < self.angle_tolerance else 0.02
        self.publish_cmd(linear, angular)

    def publish_stop(self):
        self.publish_cmd(0.0, 0.0)


def main():
    rclpy.init()
    node = GaussianPreyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                for _ in range(5):
                    node.publish_stop()
                    rclpy.spin_once(node, timeout_sec=0.02)
                    time.sleep(0.02)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()