import importlib
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from observation_factory import build_observation


class NNController(Node):
    def __init__(self):
        super().__init__("nn_controller")

        self.robot_name = self.declare_parameter("robot_name", "predator_0").value
        self.policy_path = self.declare_parameter("policy_path", "current_policy.npy").value
        self.predator_count = int(self.declare_parameter("predator_count", 3).value)
        self.observation_type = self.declare_parameter("observation_type", "fpv").value
        self.policy_module_name = self.declare_parameter("policy_module", "policies.policy_fpv").value
        self.model_states_topic = self.declare_parameter("model_states_topic", "/model_states").value

        # Only used to let all controller processes, subscriptions and publishers start.
        # No scripted spreading/movement is performed anymore.
        self.startup_delay = float(self.declare_parameter("startup_delay", 2.0).value)

        self.wheel_radius = 0.022
        self.wheel_separation = 0.0935
        self.max_wheel_omega = 8.0
        self.forward_bias = 0.5
        self.max_linear = 0.08
        self.max_angular = 2.0

        self.policy = importlib.import_module(self.policy_module_name)
        self.genome = self.load_policy()

        self.observation = build_observation(
            node=self,
            observation_type=self.observation_type,
            robot_name=self.robot_name,
            predator_count=self.predator_count,
            model_states_topic=self.model_states_topic,
        )

        self.cmd_pub = self.create_publisher(Twist, f"/{self.robot_name}/cmd_vel", 10)
        self.start_time = self.now_seconds()
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f"Started {self.robot_name}: observation={self.observation_type}, "
            f"policy={self.policy_module_name}, weights={self.policy.N_WEIGHTS}, "
            f"startup_delay={self.startup_delay}"
        )

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def load_policy(self):
        if os.path.exists(self.policy_path):
            genome = np.load(self.policy_path).astype(np.float32)
            if len(genome) != self.policy.N_WEIGHTS:
                self.get_logger().error(
                    f"Policy size mismatch: {len(genome)} != {self.policy.N_WEIGHTS}; using zeros."
                )
                return np.zeros(self.policy.N_WEIGHTS, dtype=np.float32)
            return genome

        self.get_logger().error(f"Policy file not found: {self.policy_path}; using zeros.")
        return np.zeros(self.policy.N_WEIGHTS, dtype=np.float32)

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def publish_wheel_command(self, omega_left, omega_right):
        omega_left = self.clamp(omega_left, -self.max_wheel_omega, self.max_wheel_omega)
        omega_right = self.clamp(omega_right, -self.max_wheel_omega, self.max_wheel_omega)

        omega_left += self.forward_bias
        omega_right += self.forward_bias

        omega_left = self.clamp(omega_left, -self.max_wheel_omega, self.max_wheel_omega)
        omega_right = self.clamp(omega_right, -self.max_wheel_omega, self.max_wheel_omega)

        v_left = omega_left * self.wheel_radius
        v_right = omega_right * self.wheel_radius

        cmd = Twist()
        cmd.linear.x = self.clamp((v_left + v_right) / 2.0, 0.0, self.max_linear)
        cmd.angular.z = self.clamp((v_right - v_left) / self.wheel_separation, -self.max_angular, self.max_angular)
        self.cmd_pub.publish(cmd)

    def control_loop(self):
        if self.now_seconds() - self.start_time < self.startup_delay:
            self.publish_stop()
            return

        features = self.observation.get_features()
        if features is None:
            self.publish_stop()
            return

        omega_left, omega_right = self.policy.nn_forward(features, self.genome, self.max_wheel_omega)
        self.publish_wheel_command(omega_left, omega_right)


def main():
    rclpy.init()
    node = NNController()
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