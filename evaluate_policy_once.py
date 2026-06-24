#!/usr/bin/env python3
"""
Run one evaluation episode for one saved predator policy.

Place this file in the repository root, next to:
    evaluate_best_policy.py
    config_utils.py
    spawn_robots.py
    configs/

Examples:
    python3 evaluate_policy_once.py privileged training_logs/privileged/top_policies/top_1_policy_privileged_fitness_3.2.npy

    python3 evaluate_policy_once.py fpv results/controller_selection/selected_policy_fpv.npy

    python3 evaluate_policy_once.py camera360 results/controller_selection/selected_policy_camera360.npy
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy

from config_utils import cfg_get, load_config
from evaluate_best_policy import evaluate_once, stop_all, stop_robots
from spawn_robots import clear_simulation, spawn_default_world


MODEL_CONFIGS = {
    "privileged": "configs/privileged.yaml",
    "fpv": "configs/fpv.yaml",
    "camera360": "configs/camera360.yaml",
}


def append_result(
    output_path: Path,
    model: str,
    policy_path: Path,
    fitness: float,
) -> None:
    """Optionally append the result to a CSV file."""
    fieldnames = [
        "model",
        "policy_file",
        "policy_path",
        "fitness",
        "completed_at_utc",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "model": model,
                "policy_file": policy_path.name,
                "policy_path": str(policy_path),
                "fitness": f"{fitness:.10f}",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one simulation episode for one saved policy."
    )
    parser.add_argument(
        "model",
        choices=sorted(MODEL_CONFIGS),
        help="Observation model: privileged, fpv, or camera360.",
    )
    parser.add_argument(
        "policy",
        help="Path to the saved .npy policy file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional YAML configuration path. By default, the script uses "
            "configs/<model>.yaml."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV file to which the result is appended.",
    )
    args = parser.parse_args()

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
            f"Model argument is '{model}', but the configuration contains "
            f"experiment.mode='{config_mode}'."
        )

    predator_count = int(cfg_get(cfg, "predators.count", 3))
    arena_size = float(cfg_get(cfg, "arena.size", 2.0))

    print("=" * 80)
    print("SINGLE POLICY EVALUATION")
    print("=" * 80)
    print(f"Model:  {model}")
    print(f"Policy: {policy_path}")
    print(f"Config: {config_path}")
    print("=" * 80)

    rclpy.init()

    try:
        # Start from a clean Gazebo state.
        clear_simulation()
        time.sleep(1.0)

        # Spawn the arena, predators, and prey once for this evaluation.
        spawn_default_world(
            predator_count=predator_count,
            arena_size=arena_size,
        )
        time.sleep(2.0)

        fitness = float(
            evaluate_once(
                episode_id="single-evaluation",
                cfg=cfg,
                policy_path=str(policy_path),
            )
        )

        print("\n" + "=" * 80)
        print(f"FINAL FITNESS: {fitness:.10f}")
        print("=" * 80)

        if args.output:
            output_path = Path(args.output).expanduser()
            if not output_path.is_absolute():
                output_path = (repo_root / output_path).resolve()

            append_result(
                output_path=output_path,
                model=model,
                policy_path=policy_path,
                fitness=fitness,
            )
            print(f"Result appended to: {output_path}")

        return 0

    finally:
        # Stop all controller processes and remove spawned Gazebo entities.
        stop_all()
        stop_robots(predator_count)
        clear_simulation()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
        raise SystemExit(130)
