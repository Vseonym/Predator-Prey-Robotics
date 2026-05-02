import subprocess
import signal
import time
import os

import rclpy
import numpy as np
import cma

from fitness import CameraFitnessEvaluator
from spawn_robots import clear_simulation, reset_world, spawn_default_world
from policy import N_WEIGHTS


processes = []


def start_controller(name):
    p = subprocess.Popen(
        ["python3", "nn_controller.py", "--ros-args", "-p", f"robot_name:={name}"]
    )
    processes.append(p)


def start_all_predator_controllers():
    for i in range(5):
        start_controller(f"predator_{i}")


def stop_all():
    global processes

    if len(processes) > 0:
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

    subprocess.run(["pkill", "-f", "nn_controller.py"])
    processes = []


def save_policy(genome):
    tmp = "current_policy_tmp.npy"
    final = "current_policy.npy"

    np.save(tmp, np.array(genome, dtype=np.float32))
    os.replace(tmp, final)


def run_episode(genome, episode_id):
    print(f"\n=== Episode {episode_id} ===")

    robot_names = [f"predator_{i}" for i in range(5)]

    # Stop old controllers so no robot keeps old policy/state
    stop_all()
    time.sleep(0.5)

    # Reset simulation while robots are not controlled
    reset_world()
    time.sleep(1.5)

    # Save candidate policy before starting controllers
    save_policy(genome)
    time.sleep(0.3)

    # Start controllers fresh with this candidate policy
    start_all_predator_controllers()
    time.sleep(1.0)

    fitness_node = CameraFitnessEvaluator(robot_names)

    try:
        fitness = fitness_node.evaluate(duration=20.0, sample_dt=0.2)
    finally:
        fitness_node.destroy_node()
        stop_all()

    print(f"Episode {episode_id} fitness = {fitness}")
    return fitness


def main():
    rclpy.init()

    best_fitness = -999999.0
    episode_id = 0

    popsize = 8
    generations = 30

    es = cma.CMAEvolutionStrategy(
        np.zeros(N_WEIGHTS),
        0.5,
        {
            "popsize": popsize,
        }
    )

    try:
        print("Clearing and spawning simulation...")
        clear_simulation()
        time.sleep(1.0)

        spawn_default_world()
        time.sleep(2.0)

        for generation in range(generations):
            print(f"\n========== GENERATION {generation} ==========")

            genomes = es.ask()
            losses = []

            for genome in genomes:
                fitness = run_episode(genome, episode_id)
                episode_id += 1

                losses.append(-fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    np.save("best_policy.npy", np.array(genome, dtype=np.float32))
                    print(f"NEW BEST FITNESS: {best_fitness}")

            es.tell(genomes, losses)
            es.disp()

        print("\nTraining finished.")
        print(f"Best fitness: {best_fitness}")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        stop_all()
        clear_simulation()
        rclpy.shutdown()


if __name__ == "__main__":
    main()