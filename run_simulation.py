import subprocess
import signal
import time
import os
import csv
from pathlib import Path

import rclpy
import numpy as np

from fitness import CameraFitnessEvaluator
from spawn_robots import clear_simulation, reset_world, spawn_default_world
from policy import N_WEIGHTS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_DIR = Path("training_logs")
CSV_PATH = LOG_DIR / "fitness_history.csv"
PLOT_PATH = LOG_DIR / "fitness_curve.png"

processes = []


# =========================
# Predator setup
# =========================

PREDATOR_COUNT = 4


# =========================
# Genetic Algorithm settings
# =========================

POP_SIZE = 12
ELITES = 4
GENERATIONS = 35

INIT_SIGMA = 0.20
MUTATION_SIGMA = 0.12

# 1 is faster. Use 2 for a more reliable final run.
EVALS_PER_CANDIDATE = 2

# Episode duration includes the scripted setup/warmup phase.
# With 15s spread and 2s start delay, fitness starts after ~16s,
# so 35s gives the NN around 19 seconds of scored behaviour.
EPISODE_DURATION = 35.0


# =========================
# Scripted spread settings
# =========================
#
# Passed to nn_controller.py.
#
# Timeline from predator controller start:
#
#   0 -> SPREAD_START_DELAY:
#       stop/wait so cmd_vel connections are ready
#
#   SPREAD_START_DELAY -> SPREAD_START_DELAY + SPREAD_TURN_DURATION:
#       rotate in place once
#
#   after turn -> SPREAD_START_DELAY + SPREAD_DURATION:
#       drive straight with role-based speed
#
#   after SPREAD_START_DELAY + SPREAD_DURATION:
#       switch to NN
#
SPREAD_START_DELAY = 2.0
SPREAD_DURATION = 15.0
SPREAD_TURN_DURATION = 0.0

# Edges move faster than center to get over / around the prey.
SPREAD_EDGE_LINEAR = 0.08
SPREAD_MID_LINEAR = 0.045
SPREAD_CENTER_LINEAR = 0.035

SPREAD_TURN_ANGULAR = 0.0

# If left/right is reversed in Gazebo, change this to -1.0.
SPREAD_ANGULAR_SCALE = 1.0

# Prey waits briefly so it does not escape while predator cmd_vel topics connect.
PREY_START_DELAY = SPREAD_START_DELAY

# Time between starting controllers and starting fitness evaluation.
CONTROLLER_STARTUP_DELAY = 1.0

# Ignore scripted predator setup in the fitness.
FITNESS_WARMUP = max(
    0.0,
    SPREAD_START_DELAY + SPREAD_DURATION - CONTROLLER_STARTUP_DELAY,
)


def predator_names():
    return [f"predator_{i}" for i in range(PREDATOR_COUNT)]


def start_controller(name):
    p = subprocess.Popen(
        [
            "python3",
            "nn_controller.py",
            "--ros-args",
            "-p",
            f"robot_name:={name}",
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


def start_prey_controller():
    p = subprocess.Popen(
        [
            "python3",
            "prey_controller.py",
            "--ros-args",
            "-p",
            f"robot_name:=prey_0",
            "-p",
            f"start_delay:={PREY_START_DELAY}",
            "-p",
            f"max_forward_speed:=0.12",
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


def start_all_predator_controllers():
    for name in predator_names():
        start_controller(name)


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
    subprocess.run(["pkill", "-f", "prey_controller.py"])
    processes = []


def save_policy(genome):
    tmp = "current_policy_tmp.npy"
    final = "current_policy.npy"

    np.save(tmp, np.array(genome, dtype=np.float32))
    os.replace(tmp, final)


def run_episode(genome, episode_id):
    print(f"\n=== Episode {episode_id} ===")

    robot_names = predator_names()

    stop_all()
    time.sleep(0.5)

    reset_world()
    time.sleep(1.5)

    save_policy(genome)
    time.sleep(0.3)

    fitness_node = CameraFitnessEvaluator(robot_names)

    try:
        start_all_predator_controllers()
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

    print(f"Episode {episode_id} fitness = {fitness}")
    return fitness


def evaluate_genome(genome, episode_id):
    scores = []

    for _ in range(EVALS_PER_CANDIDATE):
        fitness = run_episode(genome, episode_id)
        scores.append(fitness)
        episode_id += 1

    mean_fitness = float(np.mean(scores))
    return mean_fitness, episode_id


def make_child(parent_a, parent_b):
    alpha = np.random.rand(N_WEIGHTS)

    child = alpha * parent_a + (1.0 - alpha) * parent_b
    child += np.random.normal(0.0, MUTATION_SIGMA, size=N_WEIGHTS)

    return child


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
            "generation_std_fitness",
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
    generation_std_fitness,
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
            generation_std_fitness,
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
    plt.title("Genetic Algorithm Fitness Over Time")
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

    population = np.random.normal(
        0.0,
        INIT_SIGMA,
        size=(POP_SIZE, N_WEIGHTS),
    )

    try:
        print("Clearing and spawning simulation...")
        clear_simulation()
        time.sleep(1.0)

        spawn_default_world(predator_count=PREDATOR_COUNT)
        time.sleep(2.0)

        print("\n4-predator role-input + team-capture fitness settings:")
        print(f"  PREDATOR_COUNT={PREDATOR_COUNT}")
        print(f"  N_WEIGHTS={N_WEIGHTS}")
        print(f"  POP_SIZE={POP_SIZE}")
        print(f"  ELITES={ELITES}")
        print(f"  GENERATIONS={GENERATIONS}")
        print(f"  MUTATION_SIGMA={MUTATION_SIGMA}")
        print(f"  EVALS_PER_CANDIDATE={EVALS_PER_CANDIDATE}")
        print(f"  SPREAD_START_DELAY={SPREAD_START_DELAY}")
        print(f"  SPREAD_DURATION={SPREAD_DURATION}")
        print(f"  SPREAD_TURN_DURATION={SPREAD_TURN_DURATION}")
        print(f"  SPREAD_EDGE_LINEAR={SPREAD_EDGE_LINEAR}")
        print(f"  SPREAD_MID_LINEAR={SPREAD_MID_LINEAR}")
        print(f"  SPREAD_CENTER_LINEAR={SPREAD_CENTER_LINEAR}")
        print(f"  SPREAD_TURN_ANGULAR={SPREAD_TURN_ANGULAR}")
        print(f"  SPREAD_ANGULAR_SCALE={SPREAD_ANGULAR_SCALE}")
        print(f"  PREY_START_DELAY={PREY_START_DELAY}")
        print(f"  FITNESS_WARMUP={FITNESS_WARMUP}")
        print(f"  EPISODE_DURATION={EPISODE_DURATION}")

        for generation in range(GENERATIONS):
            print(f"\n========== GENERATION {generation} ==========")

            generation_fitnesses = []
            generation_records = []

            for candidate_id, genome in enumerate(population):
                fitness, episode_id = evaluate_genome(genome, episode_id)
                loss = -fitness

                generation_fitnesses.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    np.save("best_policy.npy", np.array(genome, dtype=np.float32))
                    print(f"NEW BEST FITNESS: {best_fitness:.4f}")

                generation_records.append({
                    "generation": generation,
                    "episode": episode_id,
                    "candidate": candidate_id,
                    "fitness": fitness,
                    "loss": loss,
                })

            generation_fitnesses = np.array(generation_fitnesses)

            generation_best = float(np.max(generation_fitnesses))
            generation_mean = float(np.mean(generation_fitnesses))
            generation_std = float(np.std(generation_fitnesses))

            for record in generation_records:
                append_training_log(
                    generation=record["generation"],
                    episode=record["episode"],
                    candidate=record["candidate"],
                    fitness=record["fitness"],
                    loss=record["loss"],
                    generation_best_fitness=generation_best,
                    generation_mean_fitness=generation_mean,
                    generation_std_fitness=generation_std,
                    best_so_far_fitness=best_fitness,
                )

            plot_training_curve()

            print(
                f"Generation {generation}: "
                f"best={generation_best:.4f}, "
                f"mean={generation_mean:.4f}, "
                f"std={generation_std:.4f}, "
                f"best_so_far={best_fitness:.4f}"
            )

            sorted_idx = np.argsort(generation_fitnesses)[::-1]

            elites = population[sorted_idx[:ELITES]].copy()

            new_population = []

            for elite in elites:
                new_population.append(elite.copy())

            while len(new_population) < POP_SIZE:
                parent_ids = np.random.choice(ELITES, size=2, replace=True)

                parent_a = elites[parent_ids[0]]
                parent_b = elites[parent_ids[1]]

                child = make_child(parent_a, parent_b)
                new_population.append(child)

            population = np.array(new_population)

        print("\nTraining finished.")
        print(f"Best fitness: {best_fitness:.4f}")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        stop_all()
        clear_simulation()
        rclpy.shutdown()


if __name__ == "__main__":
    main()