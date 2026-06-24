#!/usr/bin/env python3
"""
Create a representative robot-trajectory picture from final evaluation data.

By default, the script selects the final-evaluation episode whose fitness is
closest to the median fitness. Predators are red and the prey is green.

Place this file in the repository root.

Examples:
    python3 plot_representative_trajectory.py privileged
    python3 plot_representative_trajectory.py fpv
    python3 plot_representative_trajectory.py camera360

Choose a specific episode instead:
    python3 plot_representative_trajectory.py privileged --episode 4
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from config_utils import cfg_get, load_config


MODEL_CONFIGS = {
    "privileged": "configs/privileged.yaml",
    "fpv": "configs/fpv.yaml",
    "camera360": "configs/camera360.yaml",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def choose_episode(
    rows: list[dict[str, str]],
    requested_episode: int | None,
) -> dict[str, str]:
    if not rows:
        raise RuntimeError("No final-evaluation episode rows were found.")

    if requested_episode is not None:
        matches = [
            row
            for row in rows
            if int(row["episode"]) == requested_episode
        ]
        if not matches:
            raise ValueError(
                f"Episode {requested_episode} is not present in episodes.csv."
            )
        return matches[0]

    median_fitness = statistics.median(
        float(row["fitness"]) for row in rows
    )

    return min(
        rows,
        key=lambda row: (
            abs(float(row["fitness"]) - median_fitness),
            int(row["episode"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one representative final-evaluation trajectory with red "
            "predators and a green prey."
        )
    )
    parser.add_argument(
        "model",
        choices=sorted(MODEL_CONFIGS),
        help="Observation model: privileged, fpv, or camera360.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/final_evaluation",
        help="Base directory created by evaluate_final_policy.py.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help=(
            "Specific episode to plot. By default, the episode closest to the "
            "median final fitness is used."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in dots per inch (default: 300).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    model = args.model

    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = (repo_root / results_dir).resolve()

    model_dir = results_dir / model
    episodes_path = model_dir / "episodes.csv"

    episode_rows = read_csv(episodes_path)
    episode_rows = [
        row for row in episode_rows if row.get("model") == model
    ]
    selected = choose_episode(episode_rows, args.episode)

    trajectory_path = Path(selected["trajectory_file"])
    if not trajectory_path.is_absolute():
        trajectory_path = model_dir / trajectory_path

    trajectory_rows = read_csv(trajectory_path)
    if not trajectory_rows:
        raise RuntimeError(f"No trajectory data in {trajectory_path}")

    config_path = (repo_root / MODEL_CONFIGS[model]).resolve()
    arena_size = 2.0
    if config_path.is_file():
        cfg = load_config(str(config_path))
        arena_size = float(cfg_get(cfg, "arena.size", 2.0))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in trajectory_rows:
        grouped.setdefault(row["robot"], []).append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["time_s"]))

    predator_names = sorted(
        name for name in grouped if name.startswith("predator_")
    )
    if not predator_names:
        raise RuntimeError("No predator trajectories were found.")
    if "prey_0" not in grouped:
        raise RuntimeError("No prey trajectory was found.")

    figure, axis = plt.subplots(figsize=(7.0, 7.0))

    half = arena_size / 2.0
    axis.add_patch(
        Rectangle(
            (-half, -half),
            arena_size,
            arena_size,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            label="Arena boundary",
        )
    )

    predator_line_styles = ["-", "--", ":", "-."]

    for index, robot_name in enumerate(predator_names):
        rows = grouped[robot_name]
        x_values = [float(row["x"]) for row in rows]
        y_values = [float(row["y"]) for row in rows]

        axis.plot(
            x_values,
            y_values,
            color="red",
            linestyle=predator_line_styles[
                index % len(predator_line_styles)
            ],
            linewidth=1.8,
            alpha=0.85,
            label=f"Predator {index + 1}",
        )
        axis.scatter(
            x_values[0],
            y_values[0],
            color="red",
            marker="o",
            s=45,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        axis.scatter(
            x_values[-1],
            y_values[-1],
            color="red",
            marker="X",
            s=65,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

    prey_rows = grouped["prey_0"]
    prey_x = [float(row["x"]) for row in prey_rows]
    prey_y = [float(row["y"]) for row in prey_rows]

    axis.plot(
        prey_x,
        prey_y,
        color="green",
        linewidth=2.4,
        label="Prey",
    )
    axis.scatter(
        prey_x[0],
        prey_y[0],
        color="green",
        marker="o",
        s=55,
        edgecolors="black",
        linewidths=0.6,
        zorder=6,
    )
    axis.scatter(
        prey_x[-1],
        prey_y[-1],
        color="green",
        marker="X",
        s=75,
        edgecolors="black",
        linewidths=0.6,
        zorder=6,
    )

    episode = int(selected["episode"])
    fitness = float(selected["fitness"])

    display_name = {
        "privileged": "Privileged",
        "fpv": "Front-camera (FPV)",
        "camera360": "360-degree camera",
    }[model]

    axis.set_title(
        f"{display_name}: representative episode {episode}\n"
        f"fitness = {fitness:.3f}"
    )
    axis.set_xlabel("x position (m)")
    axis.set_ylabel("y position (m)")
    axis.set_aspect("equal", adjustable="box")

    margin = 0.08
    axis.set_xlim(-half - margin, half + margin)
    axis.set_ylim(-half - margin, half + margin)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", frameon=True)

    axis.text(
        0.02,
        0.02,
        "Circle = start, X = end",
        transform=axis.transAxes,
        fontsize=9,
        verticalalignment="bottom",
    )

    figure.tight_layout()

    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = (repo_root / output_path).resolve()
    else:
        output_path = model_dir / (
            f"representative_trajectory_{model}.png"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"Selected episode: {episode}")
    print(f"Episode fitness: {fitness:.6f}")
    print(f"Trajectory source: {trajectory_path}")
    print(f"Picture saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
