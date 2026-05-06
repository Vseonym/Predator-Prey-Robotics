import time
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

from vision import extract_features


class CameraFitnessEvaluator(Node):
    def __init__(self, robot_names):
        super().__init__("camera_fitness_evaluator")

        self.bridge = CvBridge()
        self.robot_names = robot_names

        self.latest_robot_reward = {name: 0.0 for name in robot_names}

        # Smoothed prey area for less noisy progress reward
        self.smooth_prey_area = {name: 0.0 for name in robot_names}
        self.prev_smooth_prey_area = {name: 0.0 for name in robot_names}

        # Previous visibility for lost-sight penalty
        self.prev_prey_visible = {name: 0.0 for name in robot_names}

        # Debug counters
        self.capture_count = {name: 0 for name in robot_names}
        self.near_capture_count = {name: 0 for name in robot_names}
        self.visible_count = {name: 0 for name in robot_names}
        self.frame_count = {name: 0 for name in robot_names}

        # Debug maxima/minima to diagnose missed captures
        self.max_prey_area = {name: 0.0 for name in robot_names}
        self.max_front_close = {name: 0.0 for name in robot_names}
        self.max_center_close = {name: 0.0 for name in robot_names}
        self.min_abs_prey_x = {name: 999.0 for name in robot_names}

        # Count which part of capture condition failed during near-capture
        self.fail_area_count = {name: 0 for name in robot_names}
        self.fail_x_count = {name: 0 for name in robot_names}
        self.fail_front_count = {name: 0 for name in robot_names}

        # Store best almost-capture frame per robot
        self.best_capture_score = {name: -999.0 for name in robot_names}
        self.best_capture_snapshot = {
            name: {
                "prey_visible": 0.0,
                "prey_area": 0.0,
                "prey_x": 0.0,
                "front_close": 0.0,
                "center": 0.0,
                "center_left": 0.0,
                "center_right": 0.0,
                "left": 0.0,
                "right": 0.0,
            }
            for name in robot_names
        }

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
            camera_topic = f"/{name}/camera_sensor/image_raw"

            self.create_subscription(
                Image,
                camera_topic,
                lambda msg, robot=name: self.image_callback(msg, robot),
                10,
            )

            for prox_name in self.prox_names:
                topic = f"/{name}/proximity/{prox_name}"

                self.create_subscription(
                    LaserScan,
                    topic,
                    lambda msg, robot=name, prox=prox_name: self.proximity_callback(
                        msg,
                        robot,
                        prox,
                    ),
                    10,
                )

    def laser_scan_to_closeness(self, msg):
        """
        Convert LaserScan into normalized closeness.

        Returns:
            0.0 = far / nothing detected
            1.0 = very close / touching
        """

        if msg.range_max <= msg.range_min:
            return 0.0

        valid_ranges = []

        for r in msg.ranges:
            if math.isnan(r):
                continue

            if math.isinf(r):
                continue

            if msg.range_min <= r <= msg.range_max:
                valid_ranges.append(r)

        if not valid_ranges:
            return 0.0

        closest_range = min(valid_ranges)

        closeness = 1.0 - (
            (closest_range - msg.range_min)
            / (msg.range_max - msg.range_min)
        )

        return max(0.0, min(1.0, closeness))

    def proximity_callback(self, msg, robot_name, prox_name):
        self.prox_values[robot_name][prox_name] = self.laser_scan_to_closeness(msg)

    def get_front_close(self, robot_name):
        """
        Uses all forward-facing Thymio proximity sensors.
        These are center, center_left, center_right, left, right.
        """
        prox = self.prox_values[robot_name]

        return max(
            prox["center"],
            prox["center_left"],
            prox["center_right"],
            prox["left"],
            prox["right"],
        )

    def get_center_front_close(self, robot_name):
        """
        Strict front group only.
        Useful for comparing strict vs relaxed capture.
        """
        prox = self.prox_values[robot_name]

        return max(
            prox["center"],
            prox["center_left"],
            prox["center_right"],
        )

    def compute_obstacle_penalty(self, robot_name):
        """
        Front sensors matter more than side/front-diagonal sensors.

        Note:
        This also activates when touching prey, but capture reward is much larger,
        so successful capture is still rewarded.
        """

        prox = self.prox_values[robot_name]

        front = prox["center"]
        front_left = prox["center_left"]
        front_right = prox["center_right"]
        left = prox["left"]
        right = prox["right"]

        weighted_obstacle = (
            1.00 * front
            + 0.70 * max(front_left, front_right)
            + 0.30 * max(left, right)
        ) / 2.0

        return max(0.0, weighted_obstacle - 0.20)

    def update_capture_diagnostics(
        self,
        robot_name,
        prey_visible,
        prey_area,
        prey_x,
        front_close,
    ):
        """
        Track why captures are or are not detected.
        """

        abs_x = abs(prey_x)

        self.max_prey_area[robot_name] = max(
            self.max_prey_area[robot_name],
            prey_area,
        )

        self.max_front_close[robot_name] = max(
            self.max_front_close[robot_name],
            front_close,
        )

        self.max_center_close[robot_name] = max(
            self.max_center_close[robot_name],
            self.get_center_front_close(robot_name),
        )

        if prey_visible > 0.0:
            self.min_abs_prey_x[robot_name] = min(
                self.min_abs_prey_x[robot_name],
                abs_x,
            )

        # Very loose near-capture: useful to see if robot is almost there.
        near_capture_detected = (
            prey_visible > 0.0
            and prey_area > 0.03
            and abs_x < 0.70
            and front_close > 0.40
        )

        if near_capture_detected:
            self.near_capture_count[robot_name] += 1

        # Training capture condition
        area_ok = prey_area > 0.20
        x_ok = abs_x < 0.50
        front_ok = front_close > 0.75
        visible_ok = prey_visible > 0.0

        # Count failures only when prey is visible and at least one of the
        # closeness/camera signals is somewhat promising.
        promising_frame = (
            visible_ok
            and (
                prey_area > 0.03
                or front_close > 0.40
                or abs_x < 0.50
            )
        )

        if promising_frame:
            if not area_ok:
                self.fail_area_count[robot_name] += 1
            if not x_ok:
                self.fail_x_count[robot_name] += 1
            if not front_ok:
                self.fail_front_count[robot_name] += 1

        # Score how close this frame was to capture.
        # Higher means more capture-like.
        capture_score = 0.0

        if visible_ok:
            capture_score += 1.0

        capture_score += min(1.0, prey_area / 0.05)
        capture_score += max(0.0, 1.0 - abs_x / 0.50)
        capture_score += min(1.0, front_close / 0.75)

        if capture_score > self.best_capture_score[robot_name]:
            prox = self.prox_values[robot_name]

            self.best_capture_score[robot_name] = capture_score
            self.best_capture_snapshot[robot_name] = {
                "prey_visible": prey_visible,
                "prey_area": prey_area,
                "prey_x": prey_x,
                "front_close": front_close,
                "center": prox["center"],
                "center_left": prox["center_left"],
                "center_right": prox["center_right"],
                "left": prox["left"],
                "right": prox["right"],
            }

    def image_callback(self, msg, robot_name):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        features = extract_features(frame)

        prey_visible = features[0]
        prey_x = features[1]
        prey_area = features[2]

        red_visible = features[3]
        red_x = features[4]
        red_area = features[5]

        self.frame_count[robot_name] += 1

        if prey_visible > 0.0:
            self.visible_count[robot_name] += 1

        # Smooth prey area to reduce frame-to-frame camera flicker
        alpha = 0.75

        self.prev_smooth_prey_area[robot_name] = self.smooth_prey_area[robot_name]
        self.smooth_prey_area[robot_name] = (
            alpha * self.smooth_prey_area[robot_name]
            + (1.0 - alpha) * prey_area
        )

        smooth_area = self.smooth_prey_area[robot_name]
        prev_smooth_area = self.prev_smooth_prey_area[robot_name]

        obstacle_penalty = self.compute_obstacle_penalty(robot_name)

        # Penalize seeing another predator, especially if it is centered
        if red_visible > 0.0:
            red_centering = max(0.0, 1.0 - abs(red_x))
            red_penalty = (red_area ** 0.5) * red_centering
        else:
            red_penalty = 0.0

        if prey_visible > 0.0:
            visibility_reward = 1.0

            # Bigger prey blob roughly means closer prey
            area_reward = prey_area ** 0.5

            # Reward facing the prey, but only weakly
            center_reward = max(0.0, 1.0 - abs(prey_x))

            # Reward only positive approach
            progress_reward = max(0.0, smooth_area - prev_smooth_area)

            front_close = self.get_front_close(robot_name)

            self.update_capture_diagnostics(
                robot_name=robot_name,
                prey_visible=prey_visible,
                prey_area=prey_area,
                prey_x=prey_x,
                front_close=front_close,
            )

            # Smooth closeness reward based on visible prey area.
            # 0.15 is calibration, not "15% of whole image must be green".
            close_reward = min(1.0, prey_area / 0.15)

            # Penalize staring: prey is centered, but still far/small,
            # and the prey blob is not getting bigger.
            staring_penalty = 0.0

            if (
                center_reward > 0.80
                and prey_area < 0.10
                and progress_reward < 0.001
            ):
                staring_penalty = 1.0

            # Training capture condition.
            # Uses:
            # - prey visible
            # - enough green area to avoid "far prey + nearby wall/predator"
            # - prey roughly in front
            # - front proximity very close
            capture_detected = (
                prey_visible > 0.0
                and prey_area > 0.20
                and abs(prey_x) < 0.50
                and front_close > 0.75
            )

            capture_reward = 1.0 if capture_detected else 0.0

            if capture_detected:
                self.capture_count[robot_name] += 1
                # print(
                #     "CAPTURE DETECTED",
                #     robot_name,
                #     "area=", round(prey_area, 3),
                #     "x=", round(prey_x, 3),
                #     "front_close=", round(front_close, 3),
                # )

            lost_sight_penalty = 0.0

            reward = (
                0.02 * visibility_reward
                + 0.80 * area_reward
                + 0.05 * center_reward
                + 3.00 * progress_reward
                + 1.50 * close_reward
                + 6.00 * capture_reward
                - 0.50 * obstacle_penalty
                - 0.15 * red_penalty
                - 0.20 * staring_penalty
                - 0.10 * lost_sight_penalty
            )

        else:
            # Even when prey is not visible, still update best proximity diagnostics
            # with prey_visible = 0. This helps identify false positives from
            # hitting walls or other predators.
            front_close = self.get_front_close(robot_name)

            self.max_front_close[robot_name] = max(
                self.max_front_close[robot_name],
                front_close,
            )

            self.max_center_close[robot_name] = max(
                self.max_center_close[robot_name],
                self.get_center_front_close(robot_name),
            )

            lost_sight_penalty = (
                1.0 if self.prev_prey_visible[robot_name] > 0.0 else 0.0
            )

            reward = (
                -0.03
                - 0.50 * obstacle_penalty
                - 0.10 * red_penalty
                - 0.10 * lost_sight_penalty
            )

        self.prev_prey_visible[robot_name] = prey_visible
        self.latest_robot_reward[robot_name] = reward

    def reset_episode_state(self):
        for name in self.robot_names:
            self.latest_robot_reward[name] = 0.0
            self.smooth_prey_area[name] = 0.0
            self.prev_smooth_prey_area[name] = 0.0
            self.prev_prey_visible[name] = 0.0

            self.capture_count[name] = 0
            self.near_capture_count[name] = 0
            self.visible_count[name] = 0
            self.frame_count[name] = 0

            self.max_prey_area[name] = 0.0
            self.max_front_close[name] = 0.0
            self.max_center_close[name] = 0.0
            self.min_abs_prey_x[name] = 999.0

            self.fail_area_count[name] = 0
            self.fail_x_count[name] = 0
            self.fail_front_count[name] = 0

            self.best_capture_score[name] = -999.0
            self.best_capture_snapshot[name] = {
                "prey_visible": 0.0,
                "prey_area": 0.0,
                "prey_x": 0.0,
                "front_close": 0.0,
                "center": 0.0,
                "center_left": 0.0,
                "center_right": 0.0,
                "left": 0.0,
                "right": 0.0,
            }

            for prox_name in self.prox_names:
                self.prox_values[name][prox_name] = 0.0

    def evaluate(self, duration=20.0, sample_dt=0.2):
        self.reset_episode_state()

        total = 0.0
        samples = 0

        start = time.time()
        next_sample_time = start + sample_dt

        while time.time() - start < duration:
            # Process ROS callbacks as fast as possible.
            # This lets camera and LaserScan callbacks update frequently.
            rclpy.spin_once(self, timeout_sec=0.01)

            now = time.time()

            # Only sample the reward every sample_dt seconds.
            if now >= next_sample_time:
                step_total = 0.0

                for name in self.robot_names:
                    step_total += self.latest_robot_reward[name]

                step_average = step_total / len(self.robot_names)

                total += step_average
                samples += 1

                next_sample_time += sample_dt

        if samples == 0:
            return 0.0

        fitness = total / samples

        # self.print_debug_summary(fitness)

        return fitness

    def print_debug_summary(self, fitness):
        print("\n" + "=" * 100)
        print("FITNESS DEBUG SUMMARY")
        print("=" * 100)
        print(f"fitness: {fitness:.4f}")

        total_captures = 0
        total_near_captures = 0

        for name in self.robot_names:
            frames = max(1, self.frame_count[name])
            visible_ratio = self.visible_count[name] / frames
            captures = self.capture_count[name]
            near_captures = self.near_capture_count[name]

            total_captures += captures
            total_near_captures += near_captures

            if self.min_abs_prey_x[name] == 999.0:
                min_abs_x_str = "none"
            else:
                min_abs_x_str = f"{self.min_abs_prey_x[name]:.3f}"

            snap = self.best_capture_snapshot[name]

            print(
                f"{name}: "
                f"visible_ratio={visible_ratio:.3f}, "
                f"captures={captures}, "
                f"near_captures={near_captures}, "
                f"last_reward={self.latest_robot_reward[name]:.3f}, "
                f"max_area={self.max_prey_area[name]:.3f}, "
                f"max_front={self.max_front_close[name]:.3f}, "
                f"max_center_front={self.max_center_close[name]:.3f}, "
                f"min_abs_x={min_abs_x_str}, "
                f"fail_area={self.fail_area_count[name]}, "
                f"fail_x={self.fail_x_count[name]}, "
                f"fail_front={self.fail_front_count[name]}"
            )

            print(
                f"  best_capture_like_frame: "
                f"visible={snap['prey_visible']:.1f}, "
                f"area={snap['prey_area']:.3f}, "
                f"x={snap['prey_x']:.3f}, "
                f"front={snap['front_close']:.3f}, "
                f"center={snap['center']:.3f}, "
                f"center_left={snap['center_left']:.3f}, "
                f"center_right={snap['center_right']:.3f}, "
                f"left={snap['left']:.3f}, "
                f"right={snap['right']:.3f}"
            )

        print(f"total_capture_frames: {total_captures}")
        print(f"total_near_capture_frames: {total_near_captures}")

        print("\nCapture rule currently used:")
        print("  prey_visible > 0")
        print("  prey_area > 0.05")
        print("  abs(prey_x) < 0.50")
        print("  max(center, center_left, center_right, left, right) > 0.75")