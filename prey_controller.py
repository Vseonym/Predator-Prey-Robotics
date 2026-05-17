import math
import random
import time

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan

from vision import extract_features


# =========================
# Flee memory settings
# =========================

# How long prey keeps fleeing after danger was last detected.
# 1.0 = continue fleeing for 1 second
# 2.0 = continue fleeing for 2 seconds
FLEE_DURATION = 3


class PreyRuleBasedController(Node):
    def __init__(self):
        super().__init__("prey_rule_based_controller")

        self.robot_name = self.declare_parameter("robot_name", "prey_0").value

        # Speed setup:
        # If real Thymio max is ~0.12 m/s, keep prey max here
        # and restrict predators to 0.08 m/s in nn_controller.py.
        self.max_forward_speed = float(
            self.declare_parameter("max_forward_speed", 0.12).value
        )
        self.cruise_speed = float(
            self.declare_parameter("cruise_speed", 0.04).value
        )
        self.slow_speed = float(
            self.declare_parameter("slow_speed", 0.02).value
        )

        self.max_angular_speed = float(
            self.declare_parameter("max_angular_speed", 1.5).value
        )

        # Camera threshold only.
        self.predator_area_th = float(
            self.declare_parameter("predator_area_th", 0.01).value
        )

        # Proximity threshold.
        self.prox_active_eps = float(
            self.declare_parameter("prox_active_eps", 0.0001).value
        )

        self.bridge = CvBridge()
        self.camera_features = None

        self.prox_names = [
            "center",
            "center_left",
            "center_right",
            "left",
            "right",
            "rear_left",
            "rear_right",
        ]

        self.prox_values = {name: 0.0 for name in self.prox_names}

        camera_topic = f"/{self.robot_name}/camera_sensor/image_raw"
        self.create_subscription(Image, camera_topic, self.image_callback, 10)

        for prox_name in self.prox_names:
            topic = f"/{self.robot_name}/proximity/{prox_name}"
            self.create_subscription(
                LaserScan,
                topic,
                lambda msg, prox=prox_name: self.proximity_callback(msg, prox),
                10,
            )

        self.cmd_pub = self.create_publisher(
            Twist,
            f"/{self.robot_name}/cmd_vel",
            10,
        )

        # Wandering state
        self.wander_linear = self.cruise_speed
        self.wander_angular = 0.0
        self.next_wander_change = 0.0

        # Flee memory state
        self.flee_until_time = 0.0
        self.flee_linear = 0.0
        self.flee_angular = 0.0

        self.create_timer(0.1, self.control_loop)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.camera_features = extract_features(frame)
        except Exception as exc:
            self.get_logger().warn(f"Failed to process camera image: {exc}")

    def proximity_callback(self, msg, prox_name):
        if msg.range_max <= msg.range_min:
            self.prox_values[prox_name] = 0.0
            return

        valid_ranges = []

        for r in msg.ranges:
            if math.isnan(r) or math.isinf(r):
                continue

            if msg.range_min <= r <= msg.range_max:
                valid_ranges.append(r)

        if not valid_ranges:
            self.prox_values[prox_name] = 0.0
            return

        closest_range = min(valid_ranges)

        closeness = 1.0 - (
            (closest_range - msg.range_min) / (msg.range_max - msg.range_min)
        )

        self.prox_values[prox_name] = max(0.0, min(1.0, closeness))

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def is_active(self, value):
        return value > self.prox_active_eps

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def start_flee(self, linear, angular):
        """
        Start or refresh flee memory.
        The prey will keep publishing this command for FLEE_DURATION seconds.
        """
        self.flee_until_time = self.now_seconds() + FLEE_DURATION
        self.flee_linear = linear
        self.flee_angular = angular

    def is_fleeing(self):
        return self.now_seconds() < self.flee_until_time

    def get_proximity_groups(self):
        front = max(
            self.prox_values.get("center", 0.0),
            self.prox_values.get("center_left", 0.0),
            self.prox_values.get("center_right", 0.0),
        )

        left = max(
            self.prox_values.get("left", 0.0),
            self.prox_values.get("center_left", 0.0),
        )

        right = max(
            self.prox_values.get("right", 0.0),
            self.prox_values.get("center_right", 0.0),
        )

        back = max(
            self.prox_values.get("rear_left", 0.0),
            self.prox_values.get("rear_right", 0.0),
        )

        return front, left, right, back

    def update_wander(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if now < self.next_wander_change:
            return

        self.wander_linear = random.uniform(self.slow_speed, self.cruise_speed)
        self.wander_angular = random.uniform(
            -0.35 * self.max_angular_speed,
            0.35 * self.max_angular_speed,
        )

        self.next_wander_change = now + random.uniform(1.0, 3.0)

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
        front, left, right, back = self.get_proximity_groups()

        front_active = self.is_active(front)
        left_active = self.is_active(left)
        right_active = self.is_active(right)
        back_active = self.is_active(back)

        red_visible = 0.0
        red_x = 0.0
        red_area = 0.0

        if self.camera_features is not None:
            # extract_features returns:
            # [prey_visible, prey_x, prey_area, red_visible, red_x, red_area]
            red_visible = self.camera_features[3]
            red_x = self.camera_features[4]
            red_area = self.camera_features[5]

        predator_visible = red_visible > 0.0 and red_area > self.predator_area_th

        # 1. Any proximity sensor reads something.
        # This block has priority over camera because nearby danger is urgent.
        if front_active or left_active or right_active or back_active:
            print(
                f"PROX FIRED | "
                f"front={front:.3f} active={front_active} | "
                f"left={left:.3f} active={left_active} | "
                f"right={right:.3f} active={right_active} | "
                f"back={back:.3f} active={back_active} | "
                f"raw={self.prox_values}",
                flush=True,
            )

            # Something is directly in front.
            # Move while turning instead of spinning almost in place.
            if front_active:

                linear = max(self.cruise_speed, 0.05)

                if left > right:
                    angular = -0.45 * self.max_angular_speed
                elif right > left:
                    angular = 0.45 * self.max_angular_speed
                else:
                    angular = random.choice([-1.0, 1.0]) * 0.45 * self.max_angular_speed

                self.start_flee(linear, angular)
                self.publish_cmd(linear, angular)
                return

            # Something is behind or on the sides.
            # Run at maximum speed.
            linear = self.max_forward_speed

            rear_left = self.prox_values.get("rear_left", 0.0)
            rear_right = self.prox_values.get("rear_right", 0.0)

            # If only rear/back sensors are active, run mostly straight.
            if back_active and not left_active and not right_active:
                rear_balance = rear_right - rear_left

                if abs(rear_balance) < 0.08:
                    angular = 0.0
                else:
                    angular = 0.45 * rear_balance * self.max_angular_speed

            else:
                # If left side reads something, turn right.
                # If right side reads something, turn left.
                angular = (right - left) * self.max_angular_speed

            self.start_flee(linear, angular)
            self.publish_cmd(linear, angular)
            return

        # 2. Predator visible in camera.
        if predator_visible:

            urgency = min(1.0, red_area / 0.10)

            # red_x < 0 means predator is on left side of image.
            # For ROS angular.z:
            #   negative -> turn right
            #   positive -> turn left
            # So angular = red_x turns away from predator.
            angular = red_x * self.max_angular_speed

            # If predator is centered, pick a moderate escape direction.
            if abs(red_x) < 0.15:
                angular = random.choice([-1.0, 1.0]) * 0.45 * self.max_angular_speed

            # Move while turning instead of rotating in place.
            if urgency > 0.5 or abs(red_x) < 0.25:
                linear = max(self.cruise_speed, 0.05)
            else:
                linear = self.cruise_speed

            self.start_flee(linear, angular)
            self.publish_cmd(linear, angular)
            return

        # 3. Continue fleeing briefly after danger disappeared.
        if self.is_fleeing():
            print(
                f"FLEE MEMORY | "
                f"linear={self.flee_linear:.3f} "
                f"angular={self.flee_angular:.3f} "
                f"remaining={self.flee_until_time - self.now_seconds():.2f}s",
                flush=True,
            )

            self.publish_cmd(self.flee_linear, self.flee_angular)
            return

        # 4. Nothing dangerous: wander.
        self.update_wander()
        self.publish_cmd(self.wander_linear, self.wander_angular)

    def publish_stop(self):
        self.publish_cmd(0.0, 0.0)


def main():
    rclpy.init()
    node = PreyRuleBasedController()

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