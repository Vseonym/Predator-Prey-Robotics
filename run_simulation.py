import argparse
import csv
import importlib
import os
import signal
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy

from config_utils import load_config, cfg_get
from fitness import PaperFitnessEvaluator
from spawn_robots import (
    clear_simulation,
    spawn_default_world,
    reset_robot_poses,
)


processes = []


def predator_names(predator_count):
    return [f"predator_{i}" for i in range(predator_count)]


def start_controller(name, cfg, policy_path):
    predator_count = int(cfg_get(cfg, "predators.count", 3))
    observation_type = cfg_get(cfg, "observation.type", "fpv")
    policy_module = cfg_get(cfg, "policy.module", "policies.policy_fpv")
    model_states_topic = cfg_get(cfg, "fitness.model_states_topic", "/model_states")
    startup_delay = cfg_get(cfg, "startup.controller_delay", 2.0)

    args = [
        "python3", "nn_controller.py",
        "--ros-args",
        "-p", f"robot_name:={name}",
        "-p", f"policy_path:={policy_path}",
        "-p", f"predator_count:={predator_count}",
        "-p", f"observation_type:={observation_type}",
        "-p", f"policy_module:={policy_module}",
        "-p", f"model_states_topic:={model_states_topic}",
        "-p", f"startup_delay:={startup_delay}",
        "-p", f"use_sim_time:={str(cfg_get(cfg, 'simulation.use_sim_time', True)).lower()}",
    ]
    p = subprocess.Popen(args)
    processes.append(p)


def start_prey_controller(cfg):
    predator_count = int(cfg_get(cfg, "predators.count", 3))
    model_states_topic = cfg_get(cfg, "fitness.model_states_topic", "/model_states")
    start_delay = cfg_get(cfg, "startup.controller_delay", 2.0)

    args = [
        "python3", "prey_controller.py",
        "--ros-args",
        "-p", "robot_name:=prey_0",
        "-p", f"predator_count:={predator_count}",
        "-p", f"model_states_topic:={model_states_topic}",
        "-p", f"start_delay:={start_delay}",
        "-p", f"arena_size:={cfg_get(cfg, 'arena.size', 2.0)}",
        "-p", f"sigma_w:={cfg_get(cfg, 'prey.sigma_w', 0.2)}",
        "-p", f"sigma_p:={cfg_get(cfg, 'prey.sigma_p', 0.25)}",
        "-p", f"alpha:={cfg_get(cfg, 'prey.alpha', 0.1)}",
        "-p", f"max_forward_speed:={cfg_get(cfg, 'prey.max_forward_speed', 0.12)}",
        "-p", f"max_angular_speed:={cfg_get(cfg, 'prey.max_angular_speed', 1.5)}",
        "-p", f"angular_gain:={cfg_get(cfg, 'prey.angular_gain', 0.8)}",
        "-p", f"cruise_speed:={cfg_get(cfg, 'prey.cruise_speed', 0.10)}",
        "-p", f"turn_speed:={cfg_get(cfg, 'prey.turn_speed', 0.02)}",
        "-p", f"use_sim_time:={str(cfg_get(cfg, 'simulation.use_sim_time', True)).lower()}",
    ]
    p = subprocess.Popen(args)
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
            p.wait(timeout=2.0)
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


def save_policy(genome, policy_path):
    tmp = f"{policy_path}.tmp.npy"
    np.save(tmp, np.array(genome, dtype=np.float32))
    os.replace(tmp, policy_path)


def update_top_policies(top_policies, fitness, genome, limit=5):
    """Keep the best `limit` policies seen so far."""
    top_policies.append((float(fitness), np.array(genome, dtype=np.float32).copy()))
    top_policies.sort(key=lambda item: item[0], reverse=True)
    del top_policies[limit:]


def save_top_policies(top_policies, mode):
    """Save top policies to training_logs/<mode>/top_policies/."""
    output_dir = Path("training_logs") / mode / "top_policies"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for rank, (fitness, genome) in enumerate(top_policies, start=1):
        path = output_dir / f"top_{rank}_policy_{mode}_fitness_{fitness:.4f}.npy"
        np.save(path, genome)
        saved_paths.append(path)

    return saved_paths


def run_episode(genome, episode_id, cfg, policy_path):
    print(f"\n=== Episode {episode_id} ===")
    predator_count = int(cfg_get(cfg, "predators.count", 3))
    names = predator_names(predator_count)
    model_states_topic = cfg_get(cfg, "fitness.model_states_topic", "/model_states")
    controller_startup_delay = float(cfg_get(cfg, "startup.controller_delay", 2.0))

    # These are real-time sleeps for OS/process/service stability only.
    # Episode duration and controller startup are handled with simulation time.
    stop_sleep = float(cfg_get(cfg, "timing.stop_sleep", 0.05))
    reset_sleep = float(cfg_get(cfg, "timing.reset_sleep", 0.10))
    policy_sleep = float(cfg_get(cfg, "timing.policy_sleep", 0.02))
    process_start_wall_sleep = float(cfg_get(cfg, "timing.process_start_wall_sleep", 0.20))

    stop_all()
    time.sleep(stop_sleep)

    stop_robots(predator_count)
    time.sleep(0.05)

    reset_robot_poses(
        predator_count=predator_count,
        arena_size=float(cfg_get(cfg, "arena.size", 2.0)),
    )

    time.sleep(reset_sleep)

    save_policy(genome, policy_path)
    time.sleep(policy_sleep)

    fitness_node = PaperFitnessEvaluator(
        names,
        model_states_topic=model_states_topic,
        use_sim_time=bool(cfg_get(cfg, "simulation.use_sim_time", True)),
    )

    try:
        for name in names:
            start_controller(name, cfg, policy_path)
        start_prey_controller(cfg)

        # Give subprocesses a small amount of wall-clock time to start.
        # Then wait controller_startup_delay in simulation time.
        time.sleep(process_start_wall_sleep)
        fitness_node.wait_sim_time(
            controller_startup_delay,
            timeout_wall=float(cfg_get(cfg, "timing.startup_timeout_wall", 10.0)),
        )

        fitness = fitness_node.evaluate(
            duration=float(cfg_get(cfg, "training.episode_duration", 35.0)),
            sample_dt=float(cfg_get(cfg, "training.sample_dt", 0.2)),
            warmup_duration=0.0,
        )
    finally:
        fitness_node.destroy_node()
        stop_all()

    print(f"Episode {episode_id} fitness = {fitness}")
    return fitness


def evaluate_genome(genome, episode_id, cfg, policy_path):
    scores = []
    for _ in range(int(cfg_get(cfg, "training.evals_per_candidate", 2))):
        scores.append(run_episode(genome, episode_id, cfg, policy_path))
        episode_id += 1
    return float(np.mean(scores)), episode_id


def make_child(parent_a, parent_b, n_weights, mutation_sigma):
    alpha = np.random.rand(n_weights)
    child = alpha * parent_a + (1.0 - alpha) * parent_b
    child += np.random.normal(0.0, mutation_sigma, size=n_weights)
    return child


def init_training_log(log_dir, optimizer_type):
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "fitness_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "optimizer", "generation", "episode", "candidate", "fitness", "loss",
            "generation_best_fitness", "generation_mean_fitness", "generation_std_fitness",
            "best_so_far_fitness", "sigma",
        ])
    return csv_path


def append_training_log(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def plot_training_curve(csv_path, plot_path, optimizer_type):
    generations, gen_best, gen_mean, best_so_far = [], [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        last_seen = None
        for row in reader:
            generation = int(row["generation"])
            if generation == last_seen:
                continue
            last_seen = generation
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
    plt.ylabel("Paper fitness")
    plt.title(f"{optimizer_type.upper()} Fitness Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def run_ga(cfg, n_weights, policy_path, best_policy_path, csv_path, plot_path):
    pop_size = int(cfg_get(cfg, "training.pop_size", 12))
    elites_n = int(cfg_get(cfg, "training.elites", 4))
    generations = int(cfg_get(cfg, "training.generations", 35))
    init_sigma = float(cfg_get(cfg, "training.init_sigma", 0.20))
    mutation_sigma = float(cfg_get(cfg, "training.mutation_sigma", 0.12))

    best_fitness = -999999.0
    episode_id = 0
    top_policies = []

    population = np.random.normal(0.0, init_sigma, size=(pop_size, n_weights))

    for generation in range(generations):
        print(f"\n========== GENERATION {generation} (GA) ==========")
        fitnesses = []
        records = []

        for candidate_id, genome in enumerate(population):
            fitness, episode_id = evaluate_genome(genome, episode_id, cfg, policy_path)
            fitnesses.append(fitness)
            loss = -fitness

            if fitness > best_fitness:
                best_fitness = fitness
                np.save(best_policy_path, np.array(genome, dtype=np.float32))
                print(f"NEW BEST FITNESS: {best_fitness:.4f}")

            update_top_policies(top_policies, fitness, genome, limit=5)
            records.append((generation, episode_id, candidate_id, fitness, loss))

        fitnesses = np.array(fitnesses)
        gen_best = float(np.max(fitnesses))
        gen_mean = float(np.mean(fitnesses))
        gen_std = float(np.std(fitnesses))

        for generation_, episode_, candidate_, fitness_, loss_ in records:
            append_training_log(csv_path, [
                "ga", generation_, episode_, candidate_, fitness_, loss_,
                gen_best, gen_mean, gen_std, best_fitness, "",
            ])

        plot_training_curve(csv_path, plot_path, "ga")
        print(
            f"Generation {generation}: best={gen_best:.4f}, mean={gen_mean:.4f}, "
            f"std={gen_std:.4f}, best_so_far={best_fitness:.4f}"
        )

        sorted_idx = np.argsort(fitnesses)[::-1]
        elites = population[sorted_idx[:elites_n]].copy()
        new_population = [elite.copy() for elite in elites]

        while len(new_population) < pop_size:
            parent_ids = np.random.choice(elites_n, size=2, replace=True)
            child = make_child(
                elites[parent_ids[0]],
                elites[parent_ids[1]],
                n_weights,
                mutation_sigma,
            )
            new_population.append(child)

        population = np.array(new_population)

    saved_paths = save_top_policies(top_policies, cfg_get(cfg, "experiment.mode", "experiment"))
    print("Top policies saved:")
    for path in saved_paths:
        print(f"  {path}")

    return best_fitness


def run_cmaes(cfg, n_weights, policy_path, best_policy_path, csv_path, plot_path):
    try:
        import cma
    except ImportError as exc:
        raise ImportError(
            "CMA-ES selected, but package 'cma' is not installed. "
            "Install it with: pip install cma"
        ) from exc

    pop_size = int(cfg_get(cfg, "training.pop_size", 13))
    generations = int(cfg_get(cfg, "training.generations", 35))
    sigma = float(cfg_get(cfg, "optimizer.sigma", 0.5))
    initial_mean = np.zeros(n_weights, dtype=np.float32)

    # Keep CMA-ES output readable. Our own logging prints the useful values.
    opts = {
        "popsize": pop_size,
        "verbose": -9,
    }

    es = cma.CMAEvolutionStrategy(initial_mean, sigma, opts)

    best_fitness = -999999.0
    episode_id = 0
    top_policies = []

    for generation in range(generations):
        print(f"\n========== GENERATION {generation} (CMA-ES) ==========")

        solutions = es.ask()
        losses = []
        fitnesses = []
        records = []

        for candidate_id, genome in enumerate(solutions):
            genome = np.asarray(genome, dtype=np.float32)
            fitness, episode_id = evaluate_genome(genome, episode_id, cfg, policy_path)
            loss = -fitness

            fitnesses.append(fitness)
            losses.append(loss)

            if fitness > best_fitness:
                best_fitness = fitness
                np.save(best_policy_path, genome)
                print(f"NEW BEST FITNESS: {best_fitness:.4f}")

            update_top_policies(top_policies, fitness, genome, limit=5)
            records.append((generation, episode_id, candidate_id, fitness, loss))

        es.tell(solutions, losses)

        fitnesses = np.array(fitnesses)
        gen_best = float(np.max(fitnesses))
        gen_mean = float(np.mean(fitnesses))
        gen_std = float(np.std(fitnesses))

        for generation_, episode_, candidate_, fitness_, loss_ in records:
            append_training_log(csv_path, [
                "cmaes", generation_, episode_, candidate_, fitness_, loss_,
                gen_best, gen_mean, gen_std, best_fitness, es.sigma,
            ])

        plot_training_curve(csv_path, plot_path, "cmaes")
        print(
            f"Generation {generation}: best={gen_best:.4f}, mean={gen_mean:.4f}, "
            f"std={gen_std:.4f}, best_so_far={best_fitness:.4f}, sigma={es.sigma:.4f}"
        )

    saved_paths = save_top_policies(top_policies, cfg_get(cfg, "experiment.mode", "experiment"))
    print("Top policies saved:")
    for path in saved_paths:
        print(f"  {path}")

    return best_fitness

def stop_robots(predator_count):
    robot_names = predator_names(predator_count) + ["prey_0"]

    for robot_name in robot_names:
        subprocess.run(
            [
                "ros2",
                "topic",
                "pub",
                "--once",
                f"/{robot_name}/cmd_vel",
                "geometry_msgs/msg/Twist",
                "{}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = cfg_get(cfg, "experiment.mode", "experiment")
    policy_module_name = cfg_get(cfg, "policy.module", "policies.policy_fpv")
    policy_module = importlib.import_module(policy_module_name)
    n_weights = policy_module.N_WEIGHTS

    optimizer_type = str(cfg_get(cfg, "optimizer.type", "ga")).lower()

    predator_count = int(cfg_get(cfg, "predators.count", 3))
    arena_size = float(cfg_get(cfg, "arena.size", 2.0))

    log_dir = Path("training_logs") / mode
    csv_path = init_training_log(log_dir, optimizer_type)
    plot_path = log_dir / "fitness_curve.png"
    policy_path = f"current_policy_{mode}.npy"
    best_policy_path = f"best_policy_{mode}.npy"

    rclpy.init()

    try:
        print("Clearing and spawning simulation...")
        clear_simulation()
        time.sleep(float(cfg_get(cfg, "timing.clear_sleep", 0.2)))

        spawn_default_world(predator_count=predator_count, arena_size=arena_size)
        time.sleep(float(cfg_get(cfg, "timing.spawn_sleep", 0.5)))

        print(f"\nExperiment mode: {mode}")
        print(f"  optimizer={optimizer_type}")
        print(f"  arena_size={arena_size}m x {arena_size}m")
        print(f"  predators={predator_count}")
        print(f"  observation={cfg_get(cfg, 'observation.type')}")
        print(f"  policy={policy_module_name}")
        print(f"  N_WEIGHTS={n_weights}")
        print(f"  startup_delay={cfg_get(cfg, 'startup.controller_delay', 2.0)} sim seconds")
        print(f"  use_sim_time={cfg_get(cfg, 'simulation.use_sim_time', True)}")
        print(f"  episode_duration={cfg_get(cfg, 'training.episode_duration', 35.0)}")
        print(f"  evals_per_candidate={cfg_get(cfg, 'training.evals_per_candidate', 2)}")
        print("  fitness=paper ground-truth fitness for all modes")
        print("  scripted_spread=disabled")

        if optimizer_type == "ga":
            best_fitness = run_ga(
                cfg=cfg,
                n_weights=n_weights,
                policy_path=policy_path,
                best_policy_path=best_policy_path,
                csv_path=csv_path,
                plot_path=plot_path,
            )
        elif optimizer_type in {"cmaes", "cma-es", "cma"}:
            best_fitness = run_cmaes(
                cfg=cfg,
                n_weights=n_weights,
                policy_path=policy_path,
                best_policy_path=best_policy_path,
                csv_path=csv_path,
                plot_path=plot_path,
            )
        else:
            raise ValueError(f"Unknown optimizer.type: {optimizer_type}")

        print("\nTraining finished.")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Best policy: {best_policy_path}")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        stop_all()
        clear_simulation()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()