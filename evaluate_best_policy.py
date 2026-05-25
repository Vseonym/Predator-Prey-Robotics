import os
import time
import signal
import subprocess

import rclpy
import numpy as np

from fitness import CameraFitnessEvaluator
from spawn_robots import reset_world, spawn_default_world, clear_simulation


processes = []


# =========================
# Evaluation settings
# =========================

PREDATOR_COUNT = 4
POLICY_PATH = "best_policy.npy"
NUM_EPISODES = 5

# Keep aligned with run_simulation.py.
EPISODE_DURATION = 50.0


# =========================
# Scripted spread settings
# =========================

SPREAD_START_DELAY = 2.0
SPREAD_DURATION = 15.0
SPREAD_TURN_DURATION = 0.0

SPREAD_EDGE_LINEAR = 0.08
SPREAD_MID_LINEAR = 0.045
SPREAD_CENTER_LINEAR = 0.035

SPREAD_TURN_ANGULAR = 0.0
SPREAD_ANGULAR_SCALE = 1.0

PREY_START_DELAY = SPREAD_START_DELAY
CONTROLLER_STARTUP_DELAY = 1.0

FITNESS_WARMUP = max(
    0.0,
    SPREAD_START_DELAY + SPREAD_DURATION - CONTROLLER_STARTUP_DELAY,
)


def predator_names():
    return [f"predator_{i}" for i in range(PREDATOR_COUNT)]


def start_controller(name, policy_path=POLICY_PATH):
    p = subprocess.Popen(
        [
            "python3",
            "nn_controller.py",
            "--ros-args",
            "-p",
            f"robot_name:={name}",
            "-p",
            f"policy_path:={policy_path}",
            "-p",
            f"predator_count:={PREDATOR_COUNT}",
            "-p",
            f"spread_start_delay:={SPREAD_START_DELAY}",
            "-p",
            f"spread_duration:={SPREAD_DURATION}",
            "-p",
            f"spread_turn_duration:={SPREAD_TURN_DURATION}",
            "-p",
            f"spread_edge_linear:={SPREAD_EDGE_LINEAR}",
            "-p",
            f"spread_mid_linear:={SPREAD_MID_LINEAR}",
            "-p",
            f"spread_center_linear:={SPREAD_CENTER_LINEAR}",
            "-p",
            f"spread_turn_angular:={SPREAD_TURN_ANGULAR}",
            "-p",
            f"spread_angular_scale:={SPREAD_ANGULAR_SCALE}",
        ]
    )
    processes.append(p)


def start_all_predator_controllers(policy_path=POLICY_PATH):
    for name in predator_names():
        start_controller(name, policy_path)


def start_prey_controller():
    p = subprocess.Popen(
        [
            "python3",
            "prey_controller.py",
            "--ros-args",
            "-p",
            "robot_name:=prey_0",
            "-p",
            f"start_delay:={PREY_START_DELAY}",
            "-p",
            "max_forward_speed:=0.12",
            "-p",
            "cruise_speed:=0.04",
            "-p",
            "slow_speed:=0.02",
            "-p",
            "max_angular_speed:=1.5",
            "-p",
            "predator_area_th:=0.01",
            "-p",
            "prox_active_eps:=0.0001",
        ]
    )
    processes.append(p)


def stop_all():
    global processes

    if processes:
        print("Stopping controllers...")

    for p in processes:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)

    for p in processes:
        try:
            p.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

    subprocess.run(
        ["pkill", "-f", "nn_controller.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    subprocess.run(
        ["pkill", "-f", "prey_controller.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    processes = []


def stop_robots():
    """
    Send zero cmd_vel to all predators and prey.
    Useful because Gazebo may keep the last velocity command briefly.
    """
    robot_names = predator_names() + ["prey_0"]

    for robot_name in robot_names:
        topic = f"/{robot_name}/cmd_vel"

        subprocess.run(
            [
                "ros2",
                "topic",
                "pub",
                "--once",
                topic,
                "geometry_msgs/msg/Twist",
                "{}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def evaluate_once(episode_id, policy_path=POLICY_PATH):
    print(f"\n=== Evaluation episode {episode_id} ===")

    robot_names = predator_names()

    stop_all()
    stop_robots()
    time.sleep(0.5)

    reset_world()
    time.sleep(1.5)

    fitness_node = CameraFitnessEvaluator(robot_names)

    try:
        start_all_predator_controllers(policy_path=policy_path)
        start_prey_controller()

        time.sleep(CONTROLLER_STARTUP_DELAY)

        fitness = fitness_node.evaluate(
            duration=EPISODE_DURATION,
            sample_dt=0.2,
            warmup_duration=FITNESS_WARMUP,
        )

    finally:
        fitness_node.destroy_node()
        stop_all()
        stop_robots()

    print(f"Evaluation episode {episode_id} fitness = {fitness:.4f}")
    return fitness


def main():
    rclpy.init()

    try:
        if not os.path.exists(POLICY_PATH):
            raise FileNotFoundError(f"Could not find {POLICY_PATH}")

        print(f"Using policy: {POLICY_PATH}")

        print("\nEvaluation settings:")
        print(f"  PREDATOR_COUNT={PREDATOR_COUNT}")
        print(f"  NUM_EPISODES={NUM_EPISODES}")
        print(f"  EPISODE_DURATION={EPISODE_DURATION}")
        print(f"  SPREAD_START_DELAY={SPREAD_START_DELAY}")
        print(f"  SPREAD_DURATION={SPREAD_DURATION}")
        print(f"  SPREAD_TURN_DURATION={SPREAD_TURN_DURATION}")
        print(f"  SPREAD_EDGE_LINEAR={SPREAD_EDGE_LINEAR}")
        print(f"  SPREAD_MID_LINEAR={SPREAD_MID_LINEAR}")
        print(f"  SPREAD_CENTER_LINEAR={SPREAD_CENTER_LINEAR}")
        print(f"  SPREAD_TURN_ANGULAR={SPREAD_TURN_ANGULAR}")
        print(f"  SPREAD_ANGULAR_SCALE={SPREAD_ANGULAR_SCALE}")
        print(f"  PREY_START_DELAY={PREY_START_DELAY}")
        print(f"  CONTROLLER_STARTUP_DELAY={CONTROLLER_STARTUP_DELAY}")
        print(f"  FITNESS_WARMUP={FITNESS_WARMUP}")

        print("\nClearing existing simulation...")
        clear_simulation()
        time.sleep(1.0)

        print("Spawning predators and prey...")
        spawn_default_world(predator_count=PREDATOR_COUNT)
        time.sleep(2.0)

        fitnesses = []

        for episode_id in range(NUM_EPISODES):
            fitness = evaluate_once(
                episode_id=episode_id,
                policy_path=POLICY_PATH,
            )
            fitnesses.append(fitness)

        print("\n" + "=" * 80)
        print("BEST POLICY EVALUATION SUMMARY")
        print("=" * 80)
        print(f"episodes: {NUM_EPISODES}")
        print(f"duration per episode: {EPISODE_DURATION}")
        print(f"fitnesses: {[round(f, 4) for f in fitnesses]}")
        print(f"mean fitness: {np.mean(fitnesses):.4f}")
        print(f"std fitness: {np.std(fitnesses):.4f}")
        print(f"best fitness: {np.max(fitnesses):.4f}")
        print(f"worst fitness: {np.min(fitnesses):.4f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        stop_all()
        stop_robots()

        print("Cleaning simulation...")
        clear_simulation()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()