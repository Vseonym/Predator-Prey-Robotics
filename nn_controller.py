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

        # Allow accelerated Gazebo runs to drive controller timing from /clock.
        try:
            self.declare_parameter("use_sim_time", True)
        except Exception:
            pass

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
        self.max_linear = 0.08
        self.max_angular = 1.0

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
        self.start_time = None

        # Debug only one robot to avoid flooding logs.
        self.debug_robot = self.declare_parameter("debug_robot", "predator_0").value
        self.debug_every_n = int(self.declare_parameter("debug_every_n", 10).value)
        self.debug_counter = 0

        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f"Started {self.robot_name}: observation={self.observation_type}, "
            f"policy={self.policy_module_name}, weights={self.policy.N_WEIGHTS}, "
            f"startup_delay={self.startup_delay}, use_sim_time=true"
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
        raw_left = float(omega_left)
        raw_right = float(omega_right)

        omega_left = self.clamp(raw_left, -self.max_wheel_omega, self.max_wheel_omega)
        omega_right = self.clamp(raw_right, -self.max_wheel_omega, self.max_wheel_omega)

        omega_left = self.clamp(omega_left, -self.max_wheel_omega, self.max_wheel_omega)
        omega_right = self.clamp(omega_right, -self.max_wheel_omega, self.max_wheel_omega)

        v_left = omega_left * self.wheel_radius
        v_right = omega_right * self.wheel_radius

        cmd = Twist()
        cmd.linear.x = self.clamp((v_left + v_right) / 2.0, 0.0, self.max_linear)
        cmd.angular.z = self.clamp(
            (v_right - v_left) / self.wheel_separation,
            -self.max_angular,
            self.max_angular,
        )
        self.cmd_pub.publish(cmd)

        return raw_left, raw_right, omega_left, omega_right, cmd.linear.x, cmd.angular.z

    def control_loop(self):
        now = self.now_seconds()
        if now <= 0.0:
            self.publish_stop()
            return

        if self.start_time is None:
            self.start_time = now

        if now - self.start_time < self.startup_delay:
            self.publish_stop()
            return

        features = self.observation.get_features()
        if features is None:
            self.publish_stop()
            return

        omega_left, omega_right = self.policy.nn_forward(
            features,
            self.genome,
            self.max_wheel_omega,
        )

        raw_l, raw_r, final_l, final_r, linear_x, angular_z = self.publish_wheel_command(
            omega_left,
            omega_right,
        )

        if self.robot_name == self.debug_robot:
            self.debug_counter += 1
            if self.debug_every_n > 0 and self.debug_counter % self.debug_every_n == 0:
                self.get_logger().info(
                    "[NN DEBUG] "
                    f"obs={features.tolist()} "
                    f"raw_wheels=({raw_l:.3f},{raw_r:.3f}) "
                    f"final_wheels=({final_l:.3f},{final_r:.3f}) "
                    f"cmd_linear={linear_x:.3f} "
                    f"cmd_angular={angular_z:.3f}"
                )


def main():
    rclpy.init()
    node = NNController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        # This can happen during shutdown if rclpy is interrupted while taking a message.
        # Log it once instead of printing a scary traceback after a completed episode.
        node.get_logger().warn(f"Controller spin stopped with RuntimeError: {exc}")
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