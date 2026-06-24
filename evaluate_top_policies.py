#!/usr/bin/env python3
"""
Evaluate the five retained controllers for one observation model.

Place this file in the repository root, next to:
    evaluate_best_policy.py
    run_simulation.py
    configs/
    training_logs/

For each of the five saved policies, the script:
  1. runs N evaluation episodes (default: 10),
  2. writes every episode fitness to CSV immediately,
  3. calculates mean and sample standard deviation,
  4. selects the policy with the highest mean fitness,
  5. copies the selected policy to the results directory.

The script reuses evaluate_best_policy.evaluate_once(), so selection uses the
same world reset, robot controllers, prey controller, and fitness evaluator as
the normal evaluation script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy

from config_utils import cfg_get, load_config
from evaluate_best_policy import evaluate_once, stop_all, stop_robots
from spawn_robots import clear_simulation, spawn_default_world


MODEL_CONFIGS = {
    "privileged": "configs/privileged.yaml",
    "fpv": "configs/fpv.yaml",
    "camera360": "configs/camera360.yaml",
}

FITNESS_RE = re.compile(r"_fitness_(-?\d+(?:\.\d+)?)\.npy$")


def sha256_file(path: Path) -> str:
    """Return a stable identifier for a policy file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_training_fitness(path: Path) -> float | None:
    """Extract the training fitness stored in a top-policy filename."""
    match = FITNESS_RE.search(path.name)
    return float(match.group(1)) if match else None


def find_top_policies(repo_root: Path, mode: str) -> list[dict[str, Any]]:
    """
    Find one policy for each rank 1..5.

    If stale files from earlier runs exist for the same rank, the newest file is
    used and a warning is printed.
    """
    policy_dir = repo_root / "training_logs" / mode / "top_policies"
    if not policy_dir.is_dir():
        raise FileNotFoundError(
            f"Top-policy directory does not exist:\n  {policy_dir}\n"
            "Run training first or provide the correct repository."
        )

    policies: list[dict[str, Any]] = []

    for rank in range(1, 6):
        matches = list(
            policy_dir.glob(f"top_{rank}_policy_{mode}_fitness_*.npy")
        )

        if not matches:
            raise FileNotFoundError(
                f"No saved policy found for rank {rank} in:\n  {policy_dir}"
            )

        matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        chosen = matches[0]

        if len(matches) > 1:
            print(
                f"WARNING: found {len(matches)} files for rank {rank}; "
                f"using newest: {chosen.name}"
            )

        genome = np.load(chosen)
        if genome.ndim != 1:
            raise ValueError(
                f"Policy must be a one-dimensional vector: {chosen}"
            )

        policies.append(
            {
                "rank": rank,
                "path": chosen.resolve(),
                "sha256": sha256_file(chosen),
                "training_fitness": parse_training_fitness(chosen),
                "parameter_count": int(genome.size),
            }
        )

    return policies


def read_existing_episode_rows(path: Path) -> list[dict[str, str]]:
    """Read an existing results CSV for resumable evaluations."""
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_episode_row(path: Path, row: dict[str, Any]) -> None:
    """Append one completed episode and flush it safely to disk."""
    fieldnames = [
        "model",
        "policy_rank",
        "policy_file",
        "policy_sha256",
        "training_fitness",
        "episode",
        "fitness",
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


def write_summary(
    path: Path,
    mode: str,
    policies: list[dict[str, Any]],
    episode_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Calculate policy summaries and mark the selected policy."""
    summaries: list[dict[str, Any]] = []

    for policy in policies:
        scores = [
            float(row["fitness"])
            for row in episode_rows
            if row["policy_sha256"] == policy["sha256"]
        ]

        if not scores:
            raise RuntimeError(
                f"No completed evaluations found for {policy['path'].name}"
            )

        mean_fitness = float(np.mean(scores))
        sample_std = (
            float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        )

        summaries.append(
            {
                "model": mode,
                "policy_rank": policy["rank"],
                "policy_file": policy["path"].name,
                "policy_sha256": policy["sha256"],
                "training_fitness": policy["training_fitness"],
                "episodes": len(scores),
                "mean_fitness": mean_fitness,
                "sample_std_fitness": sample_std,
                "min_fitness": float(np.min(scores)),
                "max_fitness": float(np.max(scores)),
                "selected": False,
            }
        )

    # Primary criterion: highest mean fitness.
    # Tie-breakers: lower variability, then better saved training rank.
    selected = max(
        summaries,
        key=lambda item: (
            item["mean_fitness"],
            -item["sample_std_fitness"],
            -item["policy_rank"],
        ),
    )
    selected["selected"] = True

    fieldnames = [
        "model",
        "policy_rank",
        "policy_file",
        "policy_sha256",
        "training_fitness",
        "episodes",
        "mean_fitness",
        "sample_std_fitness",
        "min_fitness",
        "max_fitness",
        "selected",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the five retained policies for one observation model "
            "and select the policy with the highest mean fitness."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIGS),
        help="Observation model to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Evaluation episodes per retained policy (default: 10).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path; defaults to configs/<model>.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/controller_selection",
        help="Directory for CSV files and selected policy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing selection results for this model and start over.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    repo_root = Path(__file__).resolve().parent
    mode = args.model

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else (repo_root / MODEL_CONFIGS[mode]).resolve()
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_csv = output_dir / f"selection_episodes_{mode}.csv"
    summary_csv = output_dir / f"selection_summary_{mode}.csv"
    selected_policy_path = output_dir / f"selected_policy_{mode}.npy"
    selected_metadata_path = output_dir / f"selected_policy_{mode}.json"

    if args.overwrite:
        for path in (
            episodes_csv,
            summary_csv,
            selected_policy_path,
            selected_metadata_path,
        ):
            path.unlink(missing_ok=True)

    cfg = load_config(str(config_path))
    config_mode = str(cfg_get(cfg, "experiment.mode", mode))
    if config_mode != mode:
        raise ValueError(
            f"Requested model '{mode}', but config experiment.mode is "
            f"'{config_mode}'."
        )

    policies = find_top_policies(repo_root, mode)
    existing_rows = read_existing_episode_rows(episodes_csv)
    completed = {
        (row["policy_sha256"], int(row["episode"]))
        for row in existing_rows
    }

    predator_count = int(cfg_get(cfg, "predators.count", 3))
    arena_size = float(cfg_get(cfg, "arena.size", 2.0))

    print("=" * 80)
    print("TOP POLICY SELECTION")
    print("=" * 80)
    print(f"model: {mode}")
    print(f"config: {config_path}")
    print(f"episodes per policy: {args.episodes}")
    print(f"output directory: {output_dir}")
    print("policies:")
    for policy in policies:
        print(
            f"  rank {policy['rank']}: {policy['path'].name} "
            f"(parameters={policy['parameter_count']})"
        )

    rclpy.init()

    try:
        clear_simulation()
        time.sleep(1.0)

        spawn_default_world(
            predator_count=predator_count,
            arena_size=arena_size,
        )
        time.sleep(2.0)

        for policy in policies:
            print("\n" + "-" * 80)
            print(
                f"Evaluating rank {policy['rank']}: "
                f"{policy['path'].name}"
            )
            print("-" * 80)

            for episode in range(1, args.episodes + 1):
                key = (policy["sha256"], episode)

                if key in completed:
                    print(
                        f"Skipping already completed episode {episode}/"
                        f"{args.episodes}"
                    )
                    continue

                fitness = evaluate_once(
                    episode_id=f"rank-{policy['rank']}-episode-{episode}",
                    cfg=cfg,
                    policy_path=str(policy["path"]),
                )

                append_episode_row(
                    episodes_csv,
                    {
                        "model": mode,
                        "policy_rank": policy["rank"],
                        "policy_file": policy["path"].name,
                        "policy_sha256": policy["sha256"],
                        "training_fitness": (
                            ""
                            if policy["training_fitness"] is None
                            else policy["training_fitness"]
                        ),
                        "episode": episode,
                        "fitness": f"{float(fitness):.10f}",
                        "completed_at_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                )
                completed.add(key)

        all_rows = read_existing_episode_rows(episodes_csv)
        current_rows = [
            row
            for row in all_rows
            if row["policy_sha256"]
            in {policy["sha256"] for policy in policies}
            and 1 <= int(row["episode"]) <= args.episodes
        ]

        expected = len(policies) * args.episodes
        if len(current_rows) != expected:
            raise RuntimeError(
                f"Expected {expected} completed episode rows, "
                f"found {len(current_rows)}."
            )

        selected = write_summary(
            summary_csv,
            mode,
            policies,
            current_rows,
        )

        selected_source = next(
            policy["path"]
            for policy in policies
            if policy["sha256"] == selected["policy_sha256"]
        )
        shutil.copy2(selected_source, selected_policy_path)

        metadata = {
            **selected,
            "selected_policy_output": str(selected_policy_path),
            "config": str(config_path),
            "episodes_per_policy": args.episodes,
            "selection_rule": (
                "highest mean fitness; ties resolved by lower sample standard "
                "deviation, then better training rank"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        selected_metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 80)
        print("SELECTION COMPLETE")
        print("=" * 80)
        print(f"selected rank: {selected['policy_rank']}")
        print(f"selected source: {selected['policy_file']}")
        print(f"mean fitness: {selected['mean_fitness']:.6f}")
        print(
            "sample standard deviation: "
            f"{selected['sample_std_fitness']:.6f}"
        )
        print(f"episode results: {episodes_csv}")
        print(f"summary: {summary_csv}")
        print(f"selected policy copy: {selected_policy_path}")
        print(f"metadata: {selected_metadata_path}")

        return 0

    finally:
        stop_all()
        stop_robots(predator_count)
        clear_simulation()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Completed episodes remain saved; rerun to resume.")
        raise SystemExit(130)
