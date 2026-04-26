import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')

        self.current_distance = None

        self.robot_name = self.declare_parameter('robot_name', 'thymio2').value
        self.get_logger().info(f"Controlling robot: {self.robot_name}")

        self.create_subscription(
            LaserScan,
            f'/{self.robot_name}/proximity/center',
            self.sensor_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            f'/{self.robot_name}/cmd_vel',
            10
        )

        self.create_timer(0.1, self.control_loop)

    def sensor_callback(self, msg: LaserScan):
        if not msg.ranges:
            return

        value = msg.ranges[0]

        if math.isinf(value) or math.isnan(value):
            value = msg.range_max

        self.current_distance = value
        self.get_logger().info(f"{self.robot_name} distance={value:.4f}")

    def control_loop(self):
        if self.current_distance is None:
            return

        cmd = Twist()
        threshold = 0.1

        if self.current_distance < threshold:
            cmd.linear.x = -0.1
            self.get_logger().info(f"{self.robot_name}: BACKWARD")
        else:
            cmd.linear.x = 0.1
            self.get_logger().info(f"{self.robot_name}: FORWARD")

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = SimpleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()