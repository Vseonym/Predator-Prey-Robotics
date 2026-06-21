import argparse
import os
import signal
import subprocess
import time

import numpy as np
import rclpy

from config_utils import load_config, cfg_get
from fitness import PaperFitnessEvaluator
from spawn_robots import reset_world, spawn_default_world, clear_simulation


processes = []


def predator_names(predator_count):
    return [f"predator_{i}" for i in range(predator_count)]


def start_controller(name, cfg, policy_path):
    args = [
        "python3", "nn_controller.py",
        "--ros-args",
        "-p", f"robot_name:={name}",
        "-p", f"policy_path:={policy_path}",
        "-p", f"predator_count:={cfg_get(cfg, 'predators.count', 3)}",
        "-p", f"observation_type:={cfg_get(cfg, 'observation.type', 'fpv')}",
        "-p", f"policy_module:={cfg_get(cfg, 'policy.module', 'policies.policy_fpv')}",
        "-p", f"startup_delay:={cfg_get(cfg, 'startup.controller_delay', 2.0)}",
    ]

    p = subprocess.Popen(args)
    processes.append(p)


def start_prey_controller(cfg):
    args = [
        "python3", "prey_controller.py",
        "--ros-args",
        "-p", "robot_name:=prey_0",
        "-p", f"predator_count:={cfg_get(cfg, 'predators.count', 3)}",
        "-p", f"start_delay:={cfg_get(cfg, 'startup.controller_delay', 2.0)}",
        "-p", f"arena_size:={cfg_get(cfg, 'arena.size', 2.0)}",
        "-p", f"sigma_w:={cfg_get(cfg, 'prey.sigma_w', 0.2)}",
        "-p", f"sigma_p:={cfg_get(cfg, 'prey.sigma_p', 0.25)}",
        "-p", f"alpha:={cfg_get(cfg, 'prey.alpha', 0.1)}",
        "-p", f"max_forward_speed:={cfg_get(cfg, 'prey.max_forward_speed', 0.12)}",
        "-p", f"max_angular_speed:={cfg_get(cfg, 'prey.max_angular_speed', 1.5)}",
    ]

    p = subprocess.Popen(args)
    processes.append(p)


def stop_all():
    global processes

    for p in processes:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)

    for p in processes:
        try:
            p.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

    subprocess.run(["pkill", "-f", "nn_controller.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "prey_controller.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    processes = []


def stop_robots(predator_count):
    for robot_name in predator_names(predator_count) + ["prey_0"]:
        subprocess.run(
            [
                "ros2", "topic", "pub", "--once",
                f"/{robot_name}/cmd_vel",
                "geometry_msgs/msg/Twist",
                "{}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def evaluate_once(episode_id, cfg, policy_path):
    predator_count = int(cfg_get(cfg, "predators.count", 3))
    robot_names = predator_names(predator_count)
    startup_delay = float(cfg_get(cfg, "startup.controller_delay", 2.0))

    print(f"\n=== Evaluation episode {episode_id} ===")

    stop_all()
    stop_robots(predator_count)
    time.sleep(0.5)

    reset_world()
    time.sleep(1.5)

    fitness_node = PaperFitnessEvaluator(robot_names)

    try:
        for name in robot_names:
            start_controller(name, cfg, policy_path)

        start_prey_controller(cfg)

        time.sleep(startup_delay)

        fitness = fitness_node.evaluate(
            duration=float(cfg_get(cfg, "training.episode_duration", 30.0)),
            sample_dt=float(cfg_get(cfg, "training.sample_dt", 0.2)),
            warmup_duration=0.0,
        )

    finally:
        fitness_node.destroy_node()
        stop_all()
        stop_robots(predator_count)

    print(f"Evaluation episode {episode_id} fitness = {fitness:.4f}")
    return fitness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = cfg_get(cfg, "experiment.mode", "experiment")
    predator_count = int(cfg_get(cfg, "predators.count", 3))
    arena_size = float(cfg_get(cfg, "arena.size", 2.0))

    policy_path = args.policy or f"best_policy_{mode}.npy"

    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Could not find policy file: {policy_path}")

    rclpy.init()

    try:
        print(f"Using config: {args.config}")
        print(f"Using policy: {policy_path}")
        print(f"Mode: {mode}")
        print(f"Predators: {predator_count}")
        print(f"Arena: {arena_size}m x {arena_size}m")
        print(f"Observation: {cfg_get(cfg, 'observation.type')}")
        print(f"Policy module: {cfg_get(cfg, 'policy.module')}")
        print(f"Episodes: {args.episodes}")

        clear_simulation()
        time.sleep(1.0)

        spawn_default_world(
            predator_count=predator_count,
            arena_size=arena_size,
        )
        time.sleep(2.0)

        fitnesses = []

        for episode_id in range(args.episodes):
            fitnesses.append(
                evaluate_once(
                    episode_id=episode_id,
                    cfg=cfg,
                    policy_path=policy_path,
                )
            )

        print("\n" + "=" * 80)
        print("BEST POLICY EVALUATION SUMMARY")
        print("=" * 80)
        print(f"policy: {policy_path}")
        print(f"episodes: {args.episodes}")
        print(f"fitnesses: {[round(f, 4) for f in fitnesses]}")
        print(f"mean fitness: {np.mean(fitnesses):.4f}")
        print(f"std fitness: {np.std(fitnesses):.4f}")
        print(f"best fitness: {np.max(fitnesses):.4f}")
        print(f"worst fitness: {np.min(fitnesses):.4f}")

    finally:
        stop_all()
        stop_robots(predator_count)
        clear_simulation()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()