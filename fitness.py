import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Range
from cv_bridge import CvBridge

from vision import extract_features


class CameraFitnessEvaluator(Node):
    def __init__(self, robot_names):
        super().__init__('camera_fitness_evaluator')

        self.bridge = CvBridge()
        self.robot_names = robot_names

        self.latest_robot_reward = {name: 0.0 for name in robot_names}
        self.prev_prey_area = {name: 0.0 for name in robot_names}

        self.prox_names = [
            "center",
            "center_left",
            "center_right",
            "left",
            "right",
        ]

        self.prox_values = {
            robot: {prox: 0.0 for prox in self.prox_names}
            for robot in robot_names
        }

        for name in robot_names:
            camera_topic = f'/{name}/camera_sensor/image_raw'

            self.create_subscription(
                Image,
                camera_topic,
                lambda msg, robot=name: self.image_callback(msg, robot),
                10
            )

            for prox_name in self.prox_names:
                topic = f'/{name}/proximity/{prox_name}'

                self.create_subscription(
                    Range,
                    topic,
                    lambda msg, robot=name, prox=prox_name: self.proximity_callback(msg, robot, prox),
                    10
                )

    def proximity_callback(self, msg, robot_name, prox_name):
        if msg.max_range <= msg.min_range:
            self.prox_values[robot_name][prox_name] = 0.0
            return

        clipped_range = max(msg.min_range, min(msg.range, msg.max_range))

        closeness = 1.0 - (
            (clipped_range - msg.min_range) /
            (msg.max_range - msg.min_range)
        )

        self.prox_values[robot_name][prox_name] = max(0.0, min(1.0, closeness))

    def image_callback(self, msg, robot_name):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        features = extract_features(frame)

        prey_visible = features[0]
        prey_x = features[1]
        prey_area = features[2]

        prev_area = self.prev_prey_area[robot_name]

        max_obstacle_close = max(self.prox_values[robot_name].values())

        if prey_visible > 0.0:
            area_reward = prey_area ** 0.5
            center_reward = max(0.0, 1.0 - abs(prey_x))

            # Positive when getting closer, negative when moving away
            progress_reward = prey_area - prev_area

            reward = (
                1.50 * progress_reward +
                0.35 * area_reward +
                0.10 * center_reward -
                0.10 * max_obstacle_close
            )

            # Capture / very close bonus
            if prey_area > 0.20:
                reward += 1.0

        else:
            reward = -0.02 - 0.05 * max_obstacle_close

        self.prev_prey_area[robot_name] = prey_area
        self.latest_robot_reward[robot_name] = reward

    def evaluate(self, duration=20.0, sample_dt=0.2):
        total = 0.0
        samples = 0

        start = time.time()

        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

            step_total = 0.0

            for name in self.robot_names:
                step_total += self.latest_robot_reward[name]

            step_average = step_total / len(self.robot_names)

            total += step_average
            samples += 1

            time.sleep(sample_dt)

        if samples == 0:
            return 0.0

        return total / samples