import math
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates

from model_state_utils import model_pose_dict, pose_xy_yaw


class PaperFitnessEvaluator(Node):
    """
    Paper-style ground-truth fitness used for all three models.

    fitness(t) = mean_i(1 / D_i(t)) + mean_i(R_i(t))

    D_i = distance from predator i to prey
    R_i = distance from predator i to nearest other predator

    The controller may be FPV/360/privileged, but the evaluator is allowed
    to use Gazebo ground truth because this is a simulation training metric.
    """

    def __init__(self, robot_names, model_states_topic="/model_states"):
        super().__init__("paper_fitness_evaluator")
        self.robot_names = robot_names
        self.model_states = None
        self.create_subscription(ModelStates, model_states_topic, self.model_states_callback, 10)

    def model_states_callback(self, msg):
        self.model_states = msg

    def compute_step_fitness(self):
        if self.model_states is None:
            return 0.0

        poses = model_pose_dict(self.model_states)
        if "prey_0" not in poses:
            return 0.0

        prey_x, prey_y, _ = pose_xy_yaw(poses["prey_0"])
        predator_positions = []

        for name in self.robot_names:
            if name not in poses:
                return 0.0
            x, y, _ = pose_xy_yaw(poses[name])
            predator_positions.append((name, x, y))

        if not predator_positions:
            return 0.0

        inv_dist_sum = 0.0
        nearest_predator_sum = 0.0
        eps = 0.03

        for idx, (_, x, y) in enumerate(predator_positions):
            d_prey = math.hypot(prey_x - x, prey_y - y)
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
        return (inv_dist_sum / n) + (nearest_predator_sum / n)

    def evaluate(self, duration=35.0, sample_dt=0.2, warmup_duration=0.0):
        total = 0.0
        samples = 0
        start = time.time()
        next_sample_time = start + sample_dt

        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.01)
            now = time.time()
            elapsed = now - start

            if elapsed < warmup_duration:
                next_sample_time = now + sample_dt
                continue

            if now >= next_sample_time:
                total += self.compute_step_fitness()
                samples += 1
                next_sample_time += sample_dt

        return 0.0 if samples == 0 else total / samples