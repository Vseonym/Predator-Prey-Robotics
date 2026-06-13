import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from model_state_utils import OdomStore, angle_wrap


class GaussianPreyController(Node):
    def __init__(self):
        super().__init__("gaussian_prey_controller")

        self.robot_name = self.declare_parameter("robot_name", "prey_0").value
        self.predator_count = int(self.declare_parameter("predator_count", 3).value)
        self.declare_parameter("model_states_topic", "/model_states")  # compatibility only

        self.start_delay = float(self.declare_parameter("start_delay", 2.0).value)
        self.arena_size = float(self.declare_parameter("arena_size", 2.0).value)

        self.sigma_w = float(self.declare_parameter("sigma_w", 0.2).value)
        self.sigma_p = float(self.declare_parameter("sigma_p", 0.25).value)
        self.alpha = float(self.declare_parameter("alpha", 0.1).value)

        self.max_forward_speed = float(self.declare_parameter("max_forward_speed", 0.09).value)
        self.max_angular_speed = float(self.declare_parameter("max_angular_speed", 0.8).value)

        self.angular_gain = float(self.declare_parameter("angular_gain", 0.8).value)
        self.cruise_speed = float(self.declare_parameter("cruise_speed", 0.07).value)
        self.turn_speed = float(self.declare_parameter("turn_speed", 0.01).value)

        self.predator_names = [f"predator_{i}" for i in range(self.predator_count)]
        self.all_robot_names = self.predator_names + [self.robot_name]
        self.odom_store = OdomStore(self, self.all_robot_names)

        self.start_time = self.now_seconds()
        self.cmd_pub = self.create_publisher(Twist, f"/{self.robot_name}/cmd_vel", 10)
        self.create_timer(0.1, self.control_loop)

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

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
        e = math.exp(
            -(((prey_x - pred_x) ** 2) + ((prey_y - pred_y) ** 2))
            / (2.0 * sigma * sigma)
        )

        if axis == "x":
            return coeff * e * (-(prey_x - pred_x) / (sigma * sigma))

        return coeff * e * (-(prey_y - pred_y) / (sigma * sigma))

    def danger_gradient(self, prey_x, prey_y, predator_positions):
        half = self.arena_size / 2.0

        gx = (
            self.wall_gradient(prey_x, -half)
            + self.wall_gradient(prey_x, half)
        )

        gy = (
            self.wall_gradient(prey_y, -half)
            + self.wall_gradient(prey_y, half)
        )

        for pred_x, pred_y in predator_positions:
            gx += self.alpha * self.predator_gradient_component(
                prey_x, prey_y, pred_x, pred_y, "x"
            )
            gy += self.alpha * self.predator_gradient_component(
                prey_x, prey_y, pred_x, pred_y, "y"
            )

        return gx, gy

    def publish_cmd(self, linear, angular):
        cmd = Twist()
        cmd.linear.x = self.clamp(linear, 0.0, self.max_forward_speed)
        cmd.angular.z = self.clamp(
            angular,
            -self.max_angular_speed,
            self.max_angular_speed,
        )
        self.cmd_pub.publish(cmd)

    def control_loop(self):
        if self.now_seconds() - self.start_time < self.start_delay:
            self.publish_cmd(0.0, 0.0)
            return

        if not self.odom_store.has_all(self.all_robot_names):
            self.publish_cmd(0.0, 0.0)
            return

        prey_pose = self.odom_store.xy_yaw(self.robot_name)
        if prey_pose is None:
            self.publish_cmd(0.0, 0.0)
            return

        prey_x, prey_y, prey_yaw = prey_pose

        predator_positions = []
        for name in self.predator_names:
            pose = self.odom_store.xy_yaw(name)
            if pose is None:
                continue
            x, y, _ = pose
            predator_positions.append((x, y))

        if not predator_positions:
            self.publish_cmd(0.0, 0.0)
            return

        gx, gy = self.danger_gradient(prey_x, prey_y, predator_positions)

        escape_x = -gx
        escape_y = -gy

        if math.hypot(escape_x, escape_y) < 1e-9:
            self.publish_cmd(self.turn_speed, 0.0)
            return

        target_angle = math.atan2(escape_y, escape_x)
        error = angle_wrap(target_angle - prey_yaw)

        angular = self.clamp(
            self.angular_gain * error,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        alignment = max(0.0, 1.0 - abs(error) / math.pi)
        linear = self.turn_speed + alignment * (self.cruise_speed - self.turn_speed)
        linear = min(linear, self.max_forward_speed)

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