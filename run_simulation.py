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

    np.save(tmp, np.array(genome))
    os.replace(tmp, final)


def refresh_robots():
    stop_all()
    clear_simulation()
    time.sleep(2.0)

    spawn_default_world()
    start_all_predator_controllers()
    time.sleep(2.0)


def run_episode(genome, episode_id):
    print(f"\n=== Episode {episode_id} ===")

    # stop motion before reset
    save_policy(np.zeros(N_WEIGHTS))
    time.sleep(0.5)

    reset_world()
    time.sleep(1.5)

    # apply candidate policy
    save_policy(genome)
    time.sleep(0.5)

    robot_names = [f"predator_{i}" for i in range(5)]
    fitness_node = CameraFitnessEvaluator(robot_names)

    fitness = fitness_node.evaluate(duration=10.0, sample_dt=0.3)

    fitness_node.destroy_node()

    print(f"Episode {episode_id} fitness = {fitness}")
    return fitness


def main():
    rclpy.init()

    best_fitness = -999999.0
    episode_id = 0

    popsize = 5
    generations = 12
    restart_every = 3

    es = cma.CMAEvolutionStrategy(
        np.zeros(N_WEIGHTS),
        0.5,
        {"popsize": popsize}
    )

    try:
        clear_simulation()
        spawn_default_world()
        start_all_predator_controllers()
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
                    np.save("best_policy.npy", np.array(genome))
                    print(f"NEW BEST FITNESS: {best_fitness}")

                if episode_id % restart_every == 0:
                    print("Refreshing robots/controllers...")
                    refresh_robots()

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