import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from vision import extract_features
from policy import nn_forward, N_WEIGHTS


class NNController(Node):
    def __init__(self):
        super().__init__('nn_controller')

        self.robot_name = self.declare_parameter('robot_name', 'predator_0').value
        self.policy_path = self.declare_parameter('policy_path', 'current_policy.npy').value

        # Number of predators in the team.
        # This keeps role encoding and scripted spread reusable for 4 or 5 predators.
        self.predator_count = int(
            self.declare_parameter('predator_count', 4).value
        )

        # =========================
        # Scripted spread settings
        # =========================
        self.spread_start_delay = float(
            self.declare_parameter('spread_start_delay', 2.0).value
        )
        self.spread_duration = float(
            self.declare_parameter('spread_duration', 15.0).value
        )
        self.spread_turn_duration = float(
            self.declare_parameter('spread_turn_duration', 0.0).value
        )

        self.spread_edge_linear = float(
            self.declare_parameter('spread_edge_linear', 0.08).value
        )
        self.spread_mid_linear = float(
            self.declare_parameter('spread_mid_linear', 0.045).value
        )
        self.spread_center_linear = float(
            self.declare_parameter('spread_center_linear', 0.035).value
        )

        self.spread_turn_angular = float(
            self.declare_parameter('spread_turn_angular', 0.0).value
        )

        # Set to -1.0 if left/right is reversed in Gazebo.
        self.spread_angular_scale = float(
            self.declare_parameter('spread_angular_scale', 1.0).value
        )

        self.start_time = self.now_seconds()

        self.bridge = CvBridge()
        self.camera_features = None

        self.prox_names = [
            "center",
            "center_left",
            "center_right",
            "left",
            "right",
        ]

        self.prox_values = {name: 0.0 for name in self.prox_names}

        self.wheel_radius = 0.022
        self.wheel_separation = 0.0935

        self.max_wheel_omega = 8.0
        self.forward_bias = 0.5
        self.max_linear = 0.08
        self.max_angular = 2.0

        self.genome = self.load_policy()

        camera_topic = f'/{self.robot_name}/camera_sensor/image_raw'
        cmd_topic = f'/{self.robot_name}/cmd_vel'

        self.create_subscription(Image, camera_topic, self.image_callback, 10)

        for prox_name in self.prox_names:
            topic = f'/{self.robot_name}/proximity/{prox_name}'
            self.create_subscription(
                LaserScan,
                topic,
                lambda msg, prox=prox_name: self.proximity_callback(msg, prox),
                10
            )

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        self.create_timer(0.1, self.control_loop)

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def load_policy(self):
        if os.path.exists(self.policy_path):
            genome = np.load(self.policy_path)

            if len(genome) != N_WEIGHTS:
                self.get_logger().error(
                    f"Policy has wrong size: {len(genome)} != {N_WEIGHTS}. Using zero policy."
                )
                return np.zeros(N_WEIGHTS, dtype=np.float32)

            return genome.astype(np.float32)

        self.get_logger().error(
            f"Policy file not found: {self.policy_path}. Using zero policy."
        )
        return np.zeros(N_WEIGHTS, dtype=np.float32)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.camera_features = extract_features(frame)

    def proximity_callback(self, msg, prox_name):
        """
        Convert LaserScan into normalized closeness.

        0.0 = far / nothing detected
        1.0 = very close
        """

        import math

        if msg.range_max <= msg.range_min:
            self.prox_values[prox_name] = 0.0
            return

        valid_ranges = []

        for r in msg.ranges:
            if math.isnan(r):
                continue

            if math.isinf(r):
                continue

            if msg.range_min <= r <= msg.range_max:
                valid_ranges.append(r)

        if not valid_ranges:
            self.prox_values[prox_name] = 0.0
            return

        closest_range = min(valid_ranges)

        closeness = 1.0 - (
            (closest_range - msg.range_min)
            / (msg.range_max - msg.range_min)
        )

        self.prox_values[prox_name] = max(0.0, min(1.0, closeness))

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def get_robot_index(self):
        try:
            return int(self.robot_name.split("_")[-1])
        except ValueError:
            return 0

    def get_robot_role_value(self):
        """
        Normalized role input for the NN.

        For predator_count=4:
            predator_0 = -1.0
            predator_1 ~= -0.33
            predator_2 ~=  0.33
            predator_3 =  1.0

        For predator_count=5:
            predator_0 = -1.0
            predator_1 = -0.5
            predator_2 =  0.0
            predator_3 =  0.5
            predator_4 =  1.0

        This is not a hardcoded action. It only tells the shared policy
        which relative role it controls.
        """

        if self.predator_count <= 1:
            return 0.0

        robot_index = self.get_robot_index()
        robot_index = self.clamp(robot_index, 0, self.predator_count - 1)

        return -1.0 + (2.0 * robot_index / (self.predator_count - 1))

    def get_inputs(self):
        if self.camera_features is None:
            return None

        return np.array([
            self.camera_features[0],
            self.camera_features[1],
            self.camera_features[2],
            self.camera_features[3],
            self.camera_features[4],
            self.camera_features[5],
            self.prox_values["center"],
            self.prox_values["center_left"],
            self.prox_values["center_right"],
            self.prox_values["left"],
            self.prox_values["right"],
            self.get_robot_role_value(),
        ], dtype=np.float32)

    def publish_stop(self):
        stop = Twist()
        self.cmd_pub.publish(stop)

    def get_spread_turn_role(self):
        """
        Turn roles for initial half-circle/fan formation.

        With role_value:
            left edge  = -1.0 -> positive angular.z
            right edge =  1.0 -> negative angular.z

        If left/right is reversed in Gazebo, set spread_angular_scale to -1.0.
        """

        role_value = self.get_robot_role_value()
        return -0.50 * role_value

    def get_spread_linear_role(self):
        """
        Forward speed roles.

        Edge predators are fastest.
        Middle predators are slower.
        If there is a true center predator, it is slowest.
        """

        role_abs = abs(self.get_robot_role_value())

        if role_abs >= 0.90:
            return self.spread_edge_linear

        if role_abs <= 0.10:
            return self.spread_center_linear

        return self.spread_mid_linear

    def get_scripted_spread_command(self):
        """
        Initial spread command.

        Phase 0:
            stop/wait so cmd_vel connection is ready

        Phase 1:
            rotate in place once, if spread_turn_duration > 0

        Phase 2:
            drive straight with role-based speed
        """

        elapsed = self.now_seconds() - self.start_time

        # Phase 0: wait for Gazebo/ROS command connections.
        if elapsed < self.spread_start_delay:
            return 0.0, 0.0

        spread_elapsed = elapsed - self.spread_start_delay

        # Phase 1: rotate in place once.
        if spread_elapsed < self.spread_turn_duration:
            linear_x = 0.0
            angular_z = (
                self.get_spread_turn_role()
                * self.spread_turn_angular
                * self.spread_angular_scale
            )
            return linear_x, angular_z

        # Phase 2: stop rotating and drive straight.
        linear_x = self.get_spread_linear_role()
        angular_z = 0.0

        return linear_x, angular_z

    def control_loop(self):
        elapsed = self.now_seconds() - self.start_time

        # Scripted setup:
        # wait -> rotate in place -> drive straight -> switch to NN.
        scripted_total_duration = self.spread_start_delay + self.spread_duration

        if elapsed < scripted_total_duration:
            linear_x, angular_z = self.get_scripted_spread_command()

            cmd = Twist()
            cmd.linear.x = self.clamp(linear_x, 0.0, self.max_linear)
            cmd.angular.z = self.clamp(angular_z, -self.max_angular, self.max_angular)

            self.cmd_pub.publish(cmd)
            return

        features = self.get_inputs()

        if features is None:
            return

        omega_left, omega_right = nn_forward(features, self.genome)

        omega_left = self.clamp(omega_left, -self.max_wheel_omega, self.max_wheel_omega)
        omega_right = self.clamp(omega_right, -self.max_wheel_omega, self.max_wheel_omega)

        omega_left += self.forward_bias
        omega_right += self.forward_bias

        omega_left = self.clamp(omega_left, -self.max_wheel_omega, self.max_wheel_omega)
        omega_right = self.clamp(omega_right, -self.max_wheel_omega, self.max_wheel_omega)

        v_left = omega_left * self.wheel_radius
        v_right = omega_right * self.wheel_radius

        linear_x = (v_left + v_right) / 2.0
        angular_z = (v_right - v_left) / self.wheel_separation

        cmd = Twist()

        # Important: no reverse driving
        cmd.linear.x = self.clamp(linear_x, 0.0, self.max_linear)
        cmd.angular.z = self.clamp(angular_z, -self.max_angular, self.max_angular)

        self.cmd_pub.publish(cmd)


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

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()