import time
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class PreyRandomController(Node):
    def __init__(self):
        super().__init__("prey_random_controller")

        self.robot_name = self.declare_parameter("robot_name", "prey_0").value

        # Predators are capped at about 0.12 m/s.
        # Keep prey clearly slower so predators can catch it.
        self.max_forward_speed = float(
            self.declare_parameter("max_forward_speed", 0.075).value
        )

        self.min_forward_speed = float(
            self.declare_parameter("min_forward_speed", 0.025).value
        )

        # Moderate turning rate. Too high makes prey unrealistically evasive.
        self.max_angular_speed = float(
            self.declare_parameter("max_angular_speed", 0.8).value
        )

        # How often prey chooses a new random motion command.
        self.min_change_time = float(
            self.declare_parameter("min_change_time", 1.0).value
        )

        self.max_change_time = float(
            self.declare_parameter("max_change_time", 3.0).value
        )

        # Probability that prey briefly stops.
        # This helps predators catch it sometimes.
        self.pause_probability = float(
            self.declare_parameter("pause_probability", 0.15).value
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            f"/{self.robot_name}/cmd_vel",
            10,
        )

        self.current_linear = 0.0
        self.current_angular = 0.0
        self.next_change_time = self.get_clock().now().nanoseconds / 1e9

        self.create_timer(0.1, self.control_loop)

    def choose_new_motion(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if random.random() < self.pause_probability:
            # Brief pause.
            self.current_linear = 0.0
            self.current_angular = random.uniform(
                -0.3 * self.max_angular_speed,
                0.3 * self.max_angular_speed,
            )
        else:
            # Random forward motion.
            self.current_linear = random.uniform(
                self.min_forward_speed,
                self.max_forward_speed,
            )

            # Mostly gentle turns, sometimes sharper turns.
            if random.random() < 0.7:
                self.current_angular = random.uniform(
                    -0.4 * self.max_angular_speed,
                    0.4 * self.max_angular_speed,
                )
            else:
                self.current_angular = random.uniform(
                    -self.max_angular_speed,
                    self.max_angular_speed,
                )

        self.next_change_time = now + random.uniform(
            self.min_change_time,
            self.max_change_time,
        )

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if now >= self.next_change_time:
            self.choose_new_motion()

        cmd = Twist()
        cmd.linear.x = self.current_linear
        cmd.angular.z = self.current_angular
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = PreyRandomController()

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


if __name__ == "__main__":
    main()