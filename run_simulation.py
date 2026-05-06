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

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path("training_logs")
CSV_PATH = LOG_DIR / "fitness_history.csv"
PLOT_PATH = LOG_DIR / "fitness_curve.png"


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

    fitness_node = CameraFitnessEvaluator(robot_names)

    try:
        # Start controllers fresh with this candidate policy
        start_all_predator_controllers()
        time.sleep(1.0)

        fitness = fitness_node.evaluate(duration=20.0, sample_dt=0.2)
    finally:
        fitness_node.destroy_node()
        stop_all()

    print(f"Episode {episode_id} fitness = {fitness}")
    return fitness

def init_training_log():
    LOG_DIR.mkdir(exist_ok=True)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "episode",
            "candidate",
            "fitness",
            "loss",
            "generation_best_fitness",
            "generation_mean_fitness",
            "best_so_far_fitness",
        ])


def append_training_log(
    generation,
    episode,
    candidate,
    fitness,
    loss,
    generation_best_fitness,
    generation_mean_fitness,
    best_so_far_fitness,
):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            generation,
            episode,
            candidate,
            fitness,
            loss,
            generation_best_fitness,
            generation_mean_fitness,
            best_so_far_fitness,
        ])


def plot_training_curve():
    generations = []
    gen_best = []
    gen_mean = []
    best_so_far = []

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        last_seen_generation = None

        for row in reader:
            generation = int(row["generation"])

            # only keep one row per generation for generation-level curves
            if generation == last_seen_generation:
                continue

            last_seen_generation = generation
            generations.append(generation)
            gen_best.append(float(row["generation_best_fitness"]))
            gen_mean.append(float(row["generation_mean_fitness"]))
            best_so_far.append(float(row["best_so_far_fitness"]))

    if not generations:
        return

    plt.figure()
    plt.plot(generations, gen_best, label="Generation best")
    plt.plot(generations, gen_mean, label="Generation mean")
    plt.plot(generations, best_so_far, label="Best so far")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("CMA-ES Fitness Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

def main():
    rclpy.init()

    best_fitness = -999999.0
    episode_id = 0

    init_training_log()

    popsize = 6
    generations = 20

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
            generation_fitnesses = []
            generation_records = []

            for candidate_id, genome in enumerate(genomes):
                fitness = run_episode(genome, episode_id)
                loss = -fitness

                losses.append(loss)
                generation_fitnesses.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    np.save("best_policy.npy", np.array(genome, dtype=np.float32))
                    print(f"NEW BEST FITNESS: {best_fitness}")

                generation_records.append({
                    "generation": generation,
                    "episode": episode_id,
                    "candidate": candidate_id,
                    "fitness": fitness,
                    "loss": loss,
                })

                episode_id += 1

            generation_best = max(generation_fitnesses)
            generation_mean = float(np.mean(generation_fitnesses))

            for record in generation_records:
                append_training_log(
                    generation=record["generation"],
                    episode=record["episode"],
                    candidate=record["candidate"],
                    fitness=record["fitness"],
                    loss=record["loss"],
                    generation_best_fitness=generation_best,
                    generation_mean_fitness=generation_mean,
                    best_so_far_fitness=best_fitness,
                )

            plot_training_curve()

            print(
                f"Generation {generation}: "
                f"best={generation_best:.4f}, "
                f"mean={generation_mean:.4f}, "
                f"best_so_far={best_fitness:.4f}"
            )

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