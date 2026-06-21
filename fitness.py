import math
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from model_state_utils import OdomStore


class PaperFitnessEvaluator(Node):
    def __init__(self, robot_names, model_states_topic=None, use_sim_time=True):
        super().__init__(
            "paper_fitness_evaluator",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, bool(use_sim_time))
            ],
        )

        self.robot_names = list(robot_names)
        self.all_robot_names = self.robot_names + ["prey_0"]
        self.odom_store = OdomStore(self, self.all_robot_names)

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def wait_for_clock(self, timeout_wall=10.0):
        wall_start = time.time()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.now_seconds() > 0.0:
                return True

            if time.time() - wall_start > timeout_wall:
                return False

        return False

    def wait_sim_time(self, duration, timeout_wall=None):
        if duration <= 0.0:
            return True

        if not self.wait_for_clock(timeout_wall=10.0):
            return False

        start = self.now_seconds()
        wall_start = time.time()

        while rclpy.ok() and (self.now_seconds() - start) < duration:
            rclpy.spin_once(self, timeout_sec=0.001)

            if timeout_wall is not None and (time.time() - wall_start) > timeout_wall:
                return False

        return True

    def debug_print_positions(self, label):
        print(f"\n[POSITION DEBUG] {label}", flush=True)

        for name in self.all_robot_names:
            pose = self.odom_store.xy_yaw(name)
            if pose is None:
                print(f"  {name}: None", flush=True)
                continue

            x, y, yaw = pose
            print(f"  {name}: x={x:.4f}, y={y:.4f}, yaw={yaw:.4f}", flush=True)

        prey_pose = self.odom_store.xy_yaw("prey_0")
        if prey_pose is None:
            return

        prey_x, prey_y, _ = prey_pose

        for name in self.robot_names:
            pose = self.odom_store.xy_yaw(name)
            if pose is None:
                continue

            x, y, _ = pose
            d = math.hypot(prey_x - x, prey_y - y)
            inv = 1.0 / max(0.03, d)
            print(
                f"  distance {name} -> prey_0: d={d:.6f}, inv={inv:.4f}",
                flush=True,
            )

    def compute_step_components(self):
        """
        Returns:
          fitness_step,
          mean_inverse_distance,
          mean_nearest_predator_distance,
          min_predator_prey_distance,
          mean_predator_prey_distance

        Same fitness formula as before:
          fitness = mean(1 / D_i) + mean(R_i)
        """
        if not self.odom_store.has_all(self.all_robot_names):
            return None

        prey_pose = self.odom_store.xy_yaw("prey_0")
        if prey_pose is None:
            return None

        prey_x, prey_y, _ = prey_pose
        predator_positions = []

        for name in self.robot_names:
            pose = self.odom_store.xy_yaw(name)
            if pose is None:
                return None
            x, y, _ = pose
            predator_positions.append((name, x, y))

        if not predator_positions:
            return None

        eps = 0.03
        inv_dist_sum = 0.0
        nearest_predator_sum = 0.0
        prey_distances = []

        for idx, (_, x, y) in enumerate(predator_positions):
            d_prey = math.hypot(prey_x - x, prey_y - y)

            if d_prey < 0.08:
                print(
                    f"[SUSPICIOUS DISTANCE] predator={self.robot_names[idx]} "
                    f"d_prey={d_prey:.6f} "
                    f"pred=({x:.4f},{y:.4f}) "
                    f"prey=({prey_x:.4f},{prey_y:.4f})",
                    flush=True,
                )

            prey_distances.append(d_prey)
            inv_dist_sum += 1.0 / max(eps, d_prey)

            nearest = 0.0
            if len(predator_positions) > 1:
                nearest = min(
                    math.hypot(ox - x, oy - y)
                    for j, (_, ox, oy) in enumerate(predator_positions)
                    if j != idx
                )

            nearest_predator_sum += nearest

        n = len(predator_positions)

        mean_inv_distance = inv_dist_sum / n
        mean_spacing = nearest_predator_sum / n
        fitness_step = mean_inv_distance + mean_spacing

        min_prey_distance = min(prey_distances)
        mean_prey_distance = sum(prey_distances) / n

        return (
            fitness_step,
            mean_inv_distance,
            mean_spacing,
            min_prey_distance,
            mean_prey_distance,
        )

    def compute_step_fitness(self):
        components = self.compute_step_components()
        if components is None:
            return 0.0
        return components[0]

    def evaluate(self, duration=35.0, sample_dt=0.2, warmup_duration=0.0):
        if not self.wait_for_clock(timeout_wall=10.0):
            self.get_logger().warn("No /clock received; returning 0 fitness.")
            return 0.0

        total_fitness = 0.0
        total_inv_distance = 0.0
        total_spacing = 0.0
        total_mean_prey_distance = 0.0

        samples = 0
        invalid_samples = 0

        episode_min_prey_distance = float("inf")
        episode_max_step_fitness = -float("inf")
        episode_max_inv_distance = -float("inf")

        real_start = time.time()
        sim_start = self.now_seconds()
        next_sample_time = sim_start + sample_dt

        print(
            f"[FITNESS DEBUG] start: "
            f"duration_sim={duration:.3f}s, sample_dt={sample_dt:.3f}s, "
            f"real_start={real_start:.3f}, sim_start={sim_start:.3f}",
            flush=True,
        )

        self.debug_print_positions("EVALUATION START")

        last_debug_real = real_start
        printed_after_5s = False

        while rclpy.ok() and (self.now_seconds() - sim_start) < duration:
            rclpy.spin_once(self, timeout_sec=0.001)

            real_now = time.time()
            sim_now = self.now_seconds()
            sim_elapsed = sim_now - sim_start
            real_elapsed = real_now - real_start

            if not printed_after_5s and sim_elapsed >= 5.0:
                self.debug_print_positions("AFTER 5 SIM SECONDS")
                printed_after_5s = True

            if real_now - last_debug_real >= 5.0:
                factor = sim_elapsed / max(real_elapsed, 1e-9)
                print(
                    f"[FITNESS DEBUG] running: "
                    f"real_elapsed={real_elapsed:.2f}s, "
                    f"sim_elapsed={sim_elapsed:.2f}s, "
                    f"effective_factor={factor:.2f}, "
                    f"samples={samples}, "
                    f"invalid_samples={invalid_samples}",
                    flush=True,
                )
                last_debug_real = real_now

            if sim_elapsed < warmup_duration:
                next_sample_time = sim_now + sample_dt
                continue

            if sim_now >= next_sample_time:
                components = self.compute_step_components()

                if components is None:
                    invalid_samples += 1
                else:
                    (
                        fitness_step,
                        mean_inv_distance,
                        mean_spacing,
                        min_prey_distance,
                        mean_prey_distance,
                    ) = components

                    total_fitness += fitness_step
                    total_inv_distance += mean_inv_distance
                    total_spacing += mean_spacing
                    total_mean_prey_distance += mean_prey_distance

                    episode_min_prey_distance = min(
                        episode_min_prey_distance,
                        min_prey_distance,
                    )
                    episode_max_step_fitness = max(
                        episode_max_step_fitness,
                        fitness_step,
                    )
                    episode_max_inv_distance = max(
                        episode_max_inv_distance,
                        mean_inv_distance,
                    )

                    samples += 1

                while next_sample_time <= sim_now:
                    next_sample_time += sample_dt

        real_end = time.time()
        sim_end = self.now_seconds()
        real_delta = real_end - real_start
        sim_delta = sim_end - sim_start
        factor = sim_delta / max(real_delta, 1e-9)

        if samples == 0:
            result = 0.0
            mean_inv = 0.0
            mean_spacing = 0.0
            mean_prey_dist = 0.0
            min_prey_dist = 0.0
            max_step_fitness = 0.0
            max_inv = 0.0
        else:
            result = total_fitness / samples
            mean_inv = total_inv_distance / samples
            mean_spacing = total_spacing / samples
            mean_prey_dist = total_mean_prey_distance / samples
            min_prey_dist = episode_min_prey_distance
            max_step_fitness = episode_max_step_fitness
            max_inv = episode_max_inv_distance

        print(
            f"[FITNESS DEBUG] end: "
            f"real_delta={real_delta:.2f}s, "
            f"sim_delta={sim_delta:.2f}s, "
            f"effective_factor={factor:.2f}, "
            f"samples={samples}, "
            f"invalid_samples={invalid_samples}, "
            f"fitness={result:.4f}",
            flush=True,
        )

        print(
            f"[FITNESS COMPONENTS] "
            f"mean_inv_distance={mean_inv:.4f}, "
            f"mean_spacing={mean_spacing:.4f}, "
            f"mean_prey_distance={mean_prey_dist:.4f}, "
            f"min_prey_distance={min_prey_dist:.4f}, "
            f"max_step_fitness={max_step_fitness:.4f}, "
            f"max_mean_inv_distance={max_inv:.4f}",
            flush=True,
        )

        return result