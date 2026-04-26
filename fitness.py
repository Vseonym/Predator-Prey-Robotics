import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision import extract_features


class CameraFitnessEvaluator(Node):
    def __init__(self, robot_names):
        super().__init__('camera_fitness_evaluator')

        self.bridge = CvBridge()
        self.robot_names = robot_names

        self.latest_green_area = {name: 0.0 for name in robot_names}

        for name in robot_names:
            topic = f'/{name}/camera_sensor/image_raw'
            self.create_subscription(
                Image,
                topic,
                lambda msg, robot=name: self.image_callback(msg, robot),
                10
            )

    def image_callback(self, msg, robot_name):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        features = extract_features(frame)

        # features = [prey_visible, prey_x, prey_area, red_visible, red_x, red_area]
        prey_area = features[2]

        self.latest_green_area[robot_name] = prey_area

    def evaluate(self, duration=30.0, sample_dt=0.3):
        total = 0.0
        samples = 0

        start = time.time()

        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

            step_total = 0.0

            for name in self.robot_names:
                step_total += self.latest_green_area[name]

            step_average = step_total / len(self.robot_names)

            total += step_average
            samples += 1

            time.sleep(sample_dt)

        if samples == 0:
            return 0.0

        return total / samples