import time
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

from vision import extract_features


# =========================
# Real-world-compatible team reward settings
# =========================
#
# This fitness uses only data that real robots can also provide:
# - camera prey_visible
# - camera prey_x
# - camera prey_area
# - front/side proximity closeness
# - red teammate camera features
#
# It does NOT use Gazebo model positions.
#
TEAM_SENSOR_REWARD_WEIGHT = 1.0


class CameraFitnessEvaluator(Node):
    def __init__(self, robot_names):
        super().__init__("camera_fitness_evaluator")

        self.bridge = CvBridge()
        self.robot_names = robot_names

        self.latest_robot_reward = {name: 0.0 for name in robot_names}

        # Sensor-only state used for real-world-compatible team reward.
        self.latest_sensor_state = {
            name: {
                "prey_visible": 0.0,
                "prey_x": 0.0,
                "prey_area": 0.0,
                "front_close": 0.0,
                "center_front_close": 0.0,
                "red_visible": 0.0,
                "red_x": 0.0,
                "red_area": 0.0,
            }
            for name in robot_names
        }

        self.latest_team_sensor_reward = 0.0

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

    def compute_real_world_team_sensor_reward(self):
        """
        Real-world-compatible team reward.

        Uses only sensor data:
          - prey camera features
          - proximity closeness
          - red teammate camera features

        Goal:
          Encourage real team capture:
            - multiple predators pressuring prey at the same time
            - predators approaching from different visual directions
            - predators not bunching/colliding with each other
        """

        states = list(self.latest_sensor_state.values())

        if not states:
            return 0.0

        n_robots = len(states)

        visible = [
            s for s in states
            if s["prey_visible"] > 0.0
        ]

        close_camera = [
            s for s in states
            if (
                s["prey_visible"] > 0.0
                and s["prey_area"] > 0.08
            )
        ]

        very_close_camera = [
            s for s in states
            if (
                s["prey_visible"] > 0.0
                and s["prey_area"] > 0.16
            )
        ]

        contact_pressure = [
            s for s in states
            if (
                s["prey_visible"] > 0.0
                and abs(s["prey_x"]) < 0.60
                and s["front_close"] > 0.45
            )
        ]

        # Softer team-capture pressure.
        # This is intentionally easier than the individual capture condition.
        # It rewards several predators being close to the prey at the same time,
        # not only one robot making perfect contact.
        team_capture_pressure = [
            s for s in states
            if (
                s["prey_visible"] > 0.0
                and s["prey_area"] > 0.10
                and abs(s["prey_x"]) < 0.70
                and s["front_close"] > 0.40
            )
        ]

        team_capture_reward = 0.0

        if len(team_capture_pressure) >= 2:
            team_capture_reward = 1.0

        if len(team_capture_pressure) >= 3:
            team_capture_reward = 2.0

        team_visibility_reward = min(1.0, len(visible) / max(1, n_robots))
        team_close_reward = min(1.0, len(close_camera) / 3.0)
        team_very_close_reward = min(1.0, len(very_close_camera) / 2.0)
        pressure_reward = min(1.0, len(contact_pressure) / 2.0)

        # Approximate visual spread using prey_x bins.
        # 1 bin = no spread reward
        # 2 bins = medium spread reward
        # 3 bins = full spread reward
        left_view = [
            s for s in visible
            if s["prey_x"] < -0.20
        ]

        center_view = [
            s for s in visible
            if abs(s["prey_x"]) <= 0.20
        ]

        right_view = [
            s for s in visible
            if s["prey_x"] > 0.20
        ]

        view_bins = 0

        if left_view:
            view_bins += 1
        if center_view:
            view_bins += 1
        if right_view:
            view_bins += 1

        visual_spread_reward = max(0.0, (view_bins - 1) / 2.0)

        # Penalize if 2+ predators see prey but all from the same image region.
        # This discourages all robots chasing from the same direction.
        same_view_penalty = 0.0
        if len(visible) >= 2 and view_bins <= 1:
            same_view_penalty = 1.0

        # General teammate crowding using red robot features.
        # This is mild: seeing a teammate is not always bad during a surround.
        red_crowding_values = []

        # Stronger collision-like penalty:
        # another predator is large and centered in the camera.
        teammate_collision_like_count = 0

        for s in states:
            if s["red_visible"] > 0.0:
                red_centering = max(0.0, 1.0 - abs(s["red_x"]))
                red_crowding = (s["red_area"] ** 0.5) * red_centering
                red_crowding_values.append(red_crowding)

                if (
                    s["red_area"] > 0.12
                    and abs(s["red_x"]) < 0.65
                ):
                    teammate_collision_like_count += 1

        if red_crowding_values:
            teammate_crowding_penalty = min(
                1.0,
                sum(red_crowding_values) / max(1, len(red_crowding_values)),
            )
        else:
            teammate_crowding_penalty = 0.0

        teammate_collision_penalty = min(
            1.0,
            teammate_collision_like_count / 2.0,
        )

        # Do not give team reward if only one predator is involved.
        multi_robot_bonus = 1.0 if len(visible) >= 2 else 0.0

        team_reward = (
            0.10 * team_visibility_reward
            + 0.25 * team_close_reward
            + 0.25 * team_very_close_reward
            + 0.40 * pressure_reward
            + 0.65 * visual_spread_reward
            + 1.50 * team_capture_reward
            - 0.35 * same_view_penalty
            - 0.45 * teammate_crowding_penalty
            - 0.80 * teammate_collision_penalty
        )

        team_reward *= multi_robot_bonus

        return TEAM_SENSOR_REWARD_WEIGHT * team_reward

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

        near_capture_detected = (
            prey_visible > 0.0
            and prey_area > 0.03
            and abs_x < 0.70
            and front_close > 0.40
        )

        if near_capture_detected:
            self.near_capture_count[robot_name] += 1

        area_ok = prey_area > 0.20
        x_ok = abs_x < 0.50
        front_ok = front_close > 0.75
        visible_ok = prey_visible > 0.0

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

    def update_latest_sensor_state(
        self,
        robot_name,
        prey_visible,
        prey_x,
        prey_area,
        front_close,
        red_visible,
        red_x,
        red_area,
    ):
        self.latest_sensor_state[robot_name] = {
            "prey_visible": prey_visible,
            "prey_x": prey_x,
            "prey_area": prey_area,
            "front_close": front_close,
            "center_front_close": self.get_center_front_close(robot_name),
            "red_visible": red_visible,
            "red_x": red_x,
            "red_area": red_area,
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

        alpha = 0.75

        self.prev_smooth_prey_area[robot_name] = self.smooth_prey_area[robot_name]
        self.smooth_prey_area[robot_name] = (
            alpha * self.smooth_prey_area[robot_name]
            + (1.0 - alpha) * prey_area
        )

        smooth_area = self.smooth_prey_area[robot_name]
        prev_smooth_area = self.prev_smooth_prey_area[robot_name]

        obstacle_penalty = self.compute_obstacle_penalty(robot_name)

        # Individual red penalty.
        # This discourages staring at another predator in the camera.
        if red_visible > 0.0:
            red_centering = max(0.0, 1.0 - abs(red_x))
            red_penalty = (red_area ** 0.5) * red_centering
        else:
            red_penalty = 0.0

        front_close = self.get_front_close(robot_name)

        self.update_latest_sensor_state(
            robot_name=robot_name,
            prey_visible=prey_visible,
            prey_x=prey_x,
            prey_area=prey_area,
            front_close=front_close,
            red_visible=red_visible,
            red_x=red_x,
            red_area=red_area,
        )

        if prey_visible > 0.0:
            visibility_reward = 1.0
            area_reward = prey_area ** 0.5
            center_reward = max(0.0, 1.0 - abs(prey_x))
            progress_reward = max(0.0, smooth_area - prev_smooth_area)

            self.update_capture_diagnostics(
                robot_name=robot_name,
                prey_visible=prey_visible,
                prey_area=prey_area,
                prey_x=prey_x,
                front_close=front_close,
            )

            close_reward = min(1.0, prey_area / 0.15)

            staring_penalty = 0.0

            if (
                center_reward > 0.80
                and prey_area < 0.10
                and progress_reward < 0.001
            ):
                staring_penalty = 1.0

            capture_detected = (
                prey_visible > 0.0
                and prey_area > 0.20
                and abs(prey_x) < 0.50
                and front_close > 0.75
            )

            capture_reward = 1.0 if capture_detected else 0.0

            if capture_detected:
                self.capture_count[robot_name] += 1

            lost_sight_penalty = 0.0

            reward = (
                0.02 * visibility_reward
                + 0.80 * area_reward
                + 0.05 * center_reward
                + 3.00 * progress_reward
                + 1.50 * close_reward
                + 5.00 * capture_reward
                - 0.50 * obstacle_penalty
                - 0.20 * red_penalty
                - 0.20 * staring_penalty
                - 0.10 * lost_sight_penalty
            )

        else:
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
                - 0.20 * red_penalty
                - 0.10 * lost_sight_penalty
            )

        self.prev_prey_visible[robot_name] = prey_visible
        self.latest_robot_reward[robot_name] = reward

    def reset_episode_state(self):
        self.latest_team_sensor_reward = 0.0

        for name in self.robot_names:
            self.latest_robot_reward[name] = 0.0

            self.latest_sensor_state[name] = {
                "prey_visible": 0.0,
                "prey_x": 0.0,
                "prey_area": 0.0,
                "front_close": 0.0,
                "center_front_close": 0.0,
                "red_visible": 0.0,
                "red_x": 0.0,
                "red_area": 0.0,
            }

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

    def evaluate(self, duration=20.0, sample_dt=0.2, warmup_duration=0.0):
        self.reset_episode_state()

        total = 0.0
        samples = 0

        start = time.time()
        next_sample_time = start + sample_dt

        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.01)

            now = time.time()
            elapsed = now - start

            # During warmup, controllers may be running scripted behavior.
            # We still spin callbacks, but do not score this phase.
            if elapsed < warmup_duration:
                next_sample_time = now + sample_dt
                continue

            if now >= next_sample_time:
                step_total = 0.0

                for name in self.robot_names:
                    step_total += self.latest_robot_reward[name]

                step_average = step_total / len(self.robot_names)

                team_sensor_reward = self.compute_real_world_team_sensor_reward()
                self.latest_team_sensor_reward = team_sensor_reward

                total += step_average + team_sensor_reward
                samples += 1

                next_sample_time += sample_dt

        if samples == 0:
            return 0.0

        fitness = total / samples

        # Uncomment while tuning reward weights.
        # self.print_debug_summary(fitness)

        return fitness

    def print_debug_summary(self, fitness):
        print("\n" + "=" * 100)
        print("FITNESS DEBUG SUMMARY")
        print("=" * 100)
        print(f"fitness: {fitness:.4f}")
        print(f"latest_team_sensor_reward: {self.latest_team_sensor_reward:.4f}")

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
            state = self.latest_sensor_state[name]

            print(
                f"{name}: "
                f"visible_ratio={visible_ratio:.3f}, "
                f"captures={captures}, "
                f"near_captures={near_captures}, "
                f"last_reward={self.latest_robot_reward[name]:.3f}, "
                f"max_area={self.max_prey_area[name]:.3f}, "
                f"max_front={self.max_front_close[name]:.3f}, "
                f"max_center_front={self.max_center_close[name]:.3f}, "
                f"red_visible={state['red_visible']:.1f}, "
                f"red_area={state['red_area']:.3f}, "
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
        print("  prey_area > 0.20")
        print("  abs(prey_x) < 0.50")
        print("  max(center, center_left, center_right, left, right) > 0.75")
        print("\nTeam capture pressure also rewarded when 2+ predators satisfy:")
        print("  prey_visible > 0")
        print("  prey_area > 0.10")
        print("  abs(prey_x) < 0.70")
        print("  front_close > 0.40")