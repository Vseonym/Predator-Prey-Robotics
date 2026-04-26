import os

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from vision import extract_features
from policy import nn_forward, N_WEIGHTS


class NNController(Node):
    def __init__(self):
        super().__init__('nn_controller')

        self.robot_name = self.declare_parameter('robot_name', 'predator_0').value
        self.policy_path = self.declare_parameter('policy_path', 'current_policy.npy').value

        self.bridge = CvBridge()
        self.features = None

        self.wheel_radius = 0.022
        self.wheel_separation = 0.0935

        self.max_wheel_omega = 8.0      # rad/s, stable
        self.forward_bias = 2.0         # rad/s
        self.max_linear = 0.12          # m/s
        self.max_angular = 2.0          # rad/s

        self.genome = self.load_policy()

        camera_topic = f'/{self.robot_name}/camera_sensor/image_raw'
        cmd_topic = f'/{self.robot_name}/cmd_vel'

        self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f"Stable NN controller for {self.robot_name}")

    def load_policy(self):
        if os.path.exists(self.policy_path):
            return np.load(self.policy_path)

        genome = np.random.randn(N_WEIGHTS) * 0.1
        np.save(self.policy_path, genome)
        return genome

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.features = extract_features(frame)

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def control_loop(self):
        if self.features is None:
            return

        if os.path.exists(self.policy_path):
            self.genome = np.load(self.policy_path)

        omega_left, omega_right = nn_forward(self.features, self.genome)

        # Clamp raw NN wheel outputs
        omega_left = self.clamp(omega_left, -self.max_wheel_omega, self.max_wheel_omega)
        omega_right = self.clamp(omega_right, -self.max_wheel_omega, self.max_wheel_omega)

        # Mild forward bias to reduce pure spinning
        omega_left += self.forward_bias
        omega_right += self.forward_bias

        # Clamp again after bias
        omega_left = self.clamp(omega_left, -self.max_wheel_omega, self.max_wheel_omega)
        omega_right = self.clamp(omega_right, -self.max_wheel_omega, self.max_wheel_omega)

        v_left = omega_left * self.wheel_radius
        v_right = omega_right * self.wheel_radius

        linear_x = (v_left + v_right) / 2.0
        angular_z = (v_right - v_left) / self.wheel_separation

        cmd = Twist()
        cmd.linear.x = self.clamp(linear_x, -self.max_linear, self.max_linear)
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
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()