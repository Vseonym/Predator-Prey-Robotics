#!/usr/bin/env python3
"""
Final evaluation of one already-selected policy.

This script runs a fresh set of evaluation episodes (10 by default), writes the
fitness of every episode, records all predator/prey trajectories, and produces
a final summary.

Place this file in the repository root beside:
    evaluate_best_policy.py
    fitness.py
    config_utils.py
    model_state_utils.py
    spawn_robots.py
    configs/

Examples:
    python3 evaluate_final_policy.py privileged \
        results/controller_selection/selected_policy_privileged.npy

    python3 evaluate_final_policy.py fpv \
        results/controller_selection/selected_policy_fpv.npy

    python3 evaluate_final_policy.py camera360 \
        results/controller_selection/selected_policy_camera360.npy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from config_utils import cfg_get, load_config
from evaluate_best_policy import (
    predator_names,
    start_controller,
    start_prey_controller,
    stop_all,
    stop_robots,
)
from fitness import PaperFitnessEvaluator
from model_state_utils import odom_xy_yaw
from spawn_robots import clear_simulation, reset_world, spawn_default_world


MODEL_CONFIGS = {
    "privileged": "configs/privileged.yaml",
    "fpv": "configs/fpv.yaml",
    "camera360": "configs/camera360.yaml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrajectoryRecorder(Node):
    """Record ground-truth odometry while one evaluation episode is active."""

    def __init__(self, robot_names: list[str]):
        super().__init__(
            "final_evaluation_trajectory_recorder",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ],
        )

        self.robot_names = list(robot_names)
        self._lock = threading.Lock()
        self._recording = False
        self._episode = None
        self._start_stamp = None
        self._rows: list[dict[str, Any]] = []

        for robot_name in self.robot_names:
            self.create_subscription(
                Odometry,
                f"/{robot_name}/ground_truth/odom",
                lambda msg, name=robot_name: self._odom_callback(msg, name),
                20,
            )

    @staticmethod
    def _message_time(msg: Odometry) -> float:
        stamp = msg.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) / 1e9

    def _odom_callback(self, msg: Odometry, robot_name: str) -> None:
        with self._lock:
            if not self._recording:
                return

            stamp = self._message_time(msg)
            if self._start_stamp is None:
                self._start_stamp = stamp

            x, y, yaw = odom_xy_yaw(msg)
            role = "prey" if robot_name == "prey_0" else "predator"

            self._rows.append(
                {
                    "episode": self._episode,
                    "time_s": max(0.0, stamp - self._start_stamp),
                    "robot": robot_name,
                    "role": role,
                    "x": float(x),
                    "y": float(y),
                    "yaw": float(yaw),
                }
            )

    def start_episode(self, episode: int) -> None:
        with self._lock:
            self._episode = episode
            self._start_stamp = None
            self._rows = []
            self._recording = True

    def stop_episode(self) -> list[dict[str, Any]]:
        with self._lock:
            self._recording = False
            rows = list(self._rows)
            self._rows = []
            self._episode = None
            self._start_stamp = None
            return rows


def write_trajectory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No trajectory samples were recorded.")

    fieldnames = [
        "episode",
        "time_s",
        "robot",
        "role",
        "x",
        "y",
        "yaw",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    **row,
                    "time_s": f"{float(row['time_s']):.6f}",
                    "x": f"{float(row['x']):.8f}",
                    "y": f"{float(row['y']):.8f}",
                    "yaw": f"{float(row['yaw']):.8f}",
                }
            )


def append_episode_result(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "model",
        "policy_file",
        "policy_sha256",
        "episode",
        "fitness",
        "trajectory_file",
        "completed_at_utc",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def read_episode_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_episode(
    episode: int,
    cfg: dict[str, Any],
    policy_path: Path,
    recorder: TrajectoryRecorder,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Run one episode using the same evaluation procedure as
    evaluate_best_policy.evaluate_once(), while recording trajectories.
    """
    predator_count = int(cfg_get(cfg, "predators.count", 3))
    robots = predator_names(predator_count)
    startup_delay = float(cfg_get(cfg, "startup.controller_delay", 2.0))

    print(f"\n=== Final evaluation episode {episode} ===")

    stop_all()
    stop_robots(predator_count)
    time.sleep(0.5)

    reset_world()
    time.sleep(1.5)

    fitness_node = PaperFitnessEvaluator(robots)
    trajectory_rows: list[dict[str, Any]] = []

    try:
        for robot_name in robots:
            start_controller(robot_name, cfg, str(policy_path))

        start_prey_controller(cfg)
        time.sleep(startup_delay)

        recorder.start_episode(episode)

        fitness = float(
            fitness_node.evaluate(
                duration=float(
                    cfg_get(cfg, "training.episode_duration", 30.0)
                ),
                sample_dt=float(cfg_get(cfg, "training.sample_dt", 0.2)),
                warmup_duration=0.0,
            )
        )

        trajectory_rows = recorder.stop_episode()

    finally:
        # Safe even when recording already stopped.
        if not trajectory_rows:
            trajectory_rows = recorder.stop_episode()

        fitness_node.destroy_node()
        stop_all()
        stop_robots(predator_count)

    print(f"Final evaluation episode {episode} fitness = {fitness:.6f}")
    return fitness, trajectory_rows


def write_summary(
    summary_csv: Path,
    summary_json: Path,
    model: str,
    policy_path: Path,
    policy_hash: str,
    scores: list[float],
) -> None:
    sample_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    summary = {
        "model": model,
        "policy_file": policy_path.name,
        "policy_path": str(policy_path),
        "policy_sha256": policy_hash,
        "episodes": len(scores),
        "mean_fitness": float(np.mean(scores)),
        "sample_std_fitness": sample_std,
        "median_fitness": float(np.median(scores)),
        "min_fitness": float(np.min(scores)),
        "max_fitness": float(np.max(scores)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    fieldnames = list(summary.keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)

    summary_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fresh final evaluation set for one selected policy and "
            "record trajectories for every episode."
        )
    )
    parser.add_argument(
        "model",
        choices=sorted(MODEL_CONFIGS),
        help="Observation model: privileged, fpv, or camera360.",
    )
    parser.add_argument(
        "policy",
        help="Path to the selected .npy policy.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of fresh final-evaluation episodes (default: 10).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path; defaults to configs/<model>.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_evaluation",
        help="Base output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing final results for this model and start again.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    repo_root = Path(__file__).resolve().parent
    model = args.model

    policy_path = Path(args.policy).expanduser()
    if not policy_path.is_absolute():
        policy_path = (repo_root / policy_path).resolve()

    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if policy_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy policy file: {policy_path}")

    if args.config:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = (repo_root / config_path).resolve()
    else:
        config_path = (repo_root / MODEL_CONFIGS[model]).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(str(config_path))
    config_mode = str(cfg_get(cfg, "experiment.mode", model))
    if config_mode != model:
        raise ValueError(
            f"Requested model '{model}', but config experiment.mode is "
            f"'{config_mode}'."
        )

    base_output = Path(args.output_dir).expanduser()
    if not base_output.is_absolute():
        base_output = (repo_root / base_output).resolve()

    model_output = base_output / model

    if args.overwrite and model_output.exists():
        shutil.rmtree(model_output)

    trajectories_dir = model_output / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    episodes_csv = model_output / "episodes.csv"
    summary_csv = model_output / "summary.csv"
    summary_json = model_output / "summary.json"

    policy_hash = sha256_file(policy_path)
    existing_rows = read_episode_results(episodes_csv)

    different_policies = {
        row["policy_sha256"]
        for row in existing_rows
        if row.get("policy_sha256") and row["policy_sha256"] != policy_hash
    }
    if different_policies:
        raise RuntimeError(
            "The existing output directory contains results from another "
            "policy. Use --overwrite or choose another --output-dir."
        )

    completed = {
        int(row["episode"])
        for row in existing_rows
        if row.get("policy_sha256") == policy_hash
    }

    predator_count = int(cfg_get(cfg, "predators.count", 3))
    arena_size = float(cfg_get(cfg, "arena.size", 2.0))
    all_robot_names = predator_names(predator_count) + ["prey_0"]

    print("=" * 80)
    print("FINAL SELECTED-POLICY EVALUATION")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Policy: {policy_path}")
    print(f"Config: {config_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {model_output}")
    print("=" * 80)

    rclpy.init()

    recorder = TrajectoryRecorder(all_robot_names)
    executor = SingleThreadedExecutor()
    executor.add_node(recorder)
    executor_thread = threading.Thread(
        target=executor.spin,
        name="trajectory-recorder-executor",
        daemon=True,
    )
    executor_thread.start()

    try:
        clear_simulation()
        time.sleep(1.0)

        spawn_default_world(
            predator_count=predator_count,
            arena_size=arena_size,
        )
        time.sleep(2.0)

        for episode in range(1, args.episodes + 1):
            if episode in completed:
                print(f"Skipping completed episode {episode}.")
                continue

            fitness, trajectory_rows = evaluate_episode(
                episode=episode,
                cfg=cfg,
                policy_path=policy_path,
                recorder=recorder,
            )

            trajectory_path = (
                trajectories_dir / f"episode_{episode:02d}.csv"
            )
            write_trajectory_csv(trajectory_path, trajectory_rows)

            append_episode_result(
                episodes_csv,
                {
                    "model": model,
                    "policy_file": policy_path.name,
                    "policy_sha256": policy_hash,
                    "episode": episode,
                    "fitness": f"{fitness:.10f}",
                    "trajectory_file": str(
                        trajectory_path.relative_to(model_output)
                    ),
                    "completed_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                },
            )
            completed.add(episode)

        final_rows = [
            row
            for row in read_episode_results(episodes_csv)
            if row.get("policy_sha256") == policy_hash
            and 1 <= int(row["episode"]) <= args.episodes
        ]

        if len(final_rows) != args.episodes:
            raise RuntimeError(
                f"Expected {args.episodes} completed final episodes, "
                f"found {len(final_rows)}."
            )

        final_rows.sort(key=lambda row: int(row["episode"]))
        scores = [float(row["fitness"]) for row in final_rows]

        write_summary(
            summary_csv=summary_csv,
            summary_json=summary_json,
            model=model,
            policy_path=policy_path,
            policy_hash=policy_hash,
            scores=scores,
        )

        print("\n" + "=" * 80)
        print("FINAL EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Fitnesses: {[round(score, 4) for score in scores]}")
        print(f"Mean: {np.mean(scores):.6f}")
        print(
            "Sample standard deviation: "
            f"{np.std(scores, ddof=1) if len(scores) > 1 else 0.0:.6f}"
        )
        print(f"Median: {np.median(scores):.6f}")
        print(f"Episode results: {episodes_csv}")
        print(f"Summary: {summary_csv}")
        print(f"Trajectories: {trajectories_dir}")

        return 0

    finally:
        stop_all()
        stop_robots(predator_count)
        clear_simulation()

        executor.shutdown()
        executor_thread.join(timeout=3.0)
        recorder.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed episodes and trajectories remain saved; "
            "rerun the same command to resume."
        )
        raise SystemExit(130)
