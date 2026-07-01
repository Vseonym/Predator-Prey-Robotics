# Observation Models in Evolutionary Predator–Prey Robotics

This repository accompanies the BSc Artificial Intelligence thesis:

> **Observation Models in Evolutionary Predator–Prey Robotics: An Exploratory Comparison of Privileged, 360° Camera, and First-Person View Observations**  
> Georgijs Konditerovs, Vrije Universiteit Amsterdam, 2026

The project compares three observation models for evolved predator controllers in a simulated multi-robot predator–prey task:

- **Privileged observation** — exact simulator-derived geometric information.
- **360° camera observation** — four camera streams stitched into a panoramic representation.
- **First-person view (FPV)** — only the front-facing camera.

Three homogeneous Thymio predators share the same 3–4–2 feedforward neural-network controller. Its 26 parameters are optimised with CMA-ES while the arena, prey controller, fitness function, starting positions, and training settings are kept constant across the three observation conditions.

## Repository structure

```text
.
├── configs/                         # Experiment configurations
│   ├── privileged.yaml
│   ├── camera360.yaml
│   └── fpv.yaml
├── observations/                    # Observation-model implementations
├── policies/                        # Neural-network forward passes
├── thymio_description/              # ROS 2 Thymio description package
├── worlds/fast_empty.world          # Gazebo Classic world
├── training_logs/                   # Saved training histories and top policies
├── results/                         # Controller-selection and final-evaluation data
├── run_simulation.py                # Main training entry point
├── evaluate_top_policies.py         # Select best of five retained policies
├── evaluate_final_policy.py         # Fresh final evaluation and trajectories
├── evaluate_policy_once.py          # Run one saved policy once
├── plot_representative_trajectory.py
├── nn_controller.py                 # Predator controller ROS 2 node
├── prey_controller.py               # Gaussian danger-field prey controller
├── fitness.py                       # Ground-truth fitness evaluation
└── spawn_robots.py                  # Arena and robot spawning/reset logic
```

## Tested environment

The thesis experiments were developed with:

- Ubuntu 22.04 LTS
- ROS 2 Humble
- Gazebo Classic 11
- Python 3.10

The repository is written specifically around ROS 2 Humble and Gazebo Classic. Other distributions or simulator versions may require changes.

## 1. Install ROS 2 Humble

Install ROS 2 Humble Desktop on Ubuntu 22.04 by following the official ROS 2 instructions:

https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

After ROS 2 is installed, install the packages used by this project:

```bash
sudo apt update
sudo apt install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-msgs \
    ros-humble-cv-bridge \
    ros-humble-xacro \
    ros-humble-urdf \
    ros-humble-nav-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-matplotlib \
    python3-yaml
```

Install the Python implementation of CMA-ES:

```bash
python3 -m pip install --user cma
```

The project also imports `numpy`, `matplotlib`, `cv2`, and `yaml`. The commands above install their Ubuntu packages.

### Virtual environments

A normal isolated Python virtual environment cannot see ROS 2 Python packages such as `rclpy` and `cv_bridge`. The simplest approach is to use the system Python environment shown above.

When a virtual environment is required, create it with access to system packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install cma
```

## 2. Clone and build the workspace

Create a ROS 2 workspace and clone the repository into its `src` directory:

```bash
mkdir -p ~/predator_prey_ws/src
cd ~/predator_prey_ws/src
git clone https://github.com/Vseonym/Predator-Prey-Robotics.git
cd ~/predator_prey_ws
```

Initialise `rosdep` if it has not previously been configured on the machine:

```bash
sudo rosdep init
rosdep update
```

If `rosdep` reports that it has already been initialised, continue with the next step.

Install declared ROS dependencies and build the bundled `thymio_description` package:

```bash
source /opt/ros/humble/setup.bash
cd ~/predator_prey_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Confirm that ROS can find the robot-description package:

```bash
ros2 pkg prefix thymio_description
```

The command should print a path inside `~/predator_prey_ws/install`.

For convenience, the ROS environment can be sourced automatically in new terminals:

```bash
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'source ~/predator_prey_ws/install/setup.bash' >> ~/.bashrc
```

## 3. Start Gazebo Classic

Open a new terminal and run:

```bash
source /opt/ros/humble/setup.bash
source ~/predator_prey_ws/install/setup.bash
cd ~/predator_prey_ws/src/Predator-Prey-Robotics

ros2 launch gazebo_ros gazebo.launch.py \
    world:="$(pwd)/worlds/fast_empty.world"
```

Wait until Gazebo has fully started. The simulation must be running before any training or evaluation script is launched.

In another terminal, verify that the required Gazebo services are available:

```bash
source /opt/ros/humble/setup.bash
source ~/predator_prey_ws/install/setup.bash

ros2 service list | grep -E \
'/(spawn_entity|delete_entity|set_entity_state|reset_world)'
```

The project uses these services to create the arena, spawn robots, reset their poses, and remove entities between runs.

## 4. Run a quick smoke test

A full thesis run uses 1,300 simulation episodes per observation model and can take a long time. Before starting a full run, create a small temporary configuration:

```bash
cd ~/predator_prey_ws/src/Predator-Prey-Robotics
cp configs/privileged.yaml /tmp/privileged_smoke.yaml
```

Edit `/tmp/privileged_smoke.yaml` and change the relevant values to:

```yaml
experiment:
  mode: privileged_smoke

training:
  pop_size: 2
  generations: 1
  evals_per_candidate: 1
  episode_duration: 5.0
  sample_dt: 0.2
```

Keep the remaining privileged-observation settings unchanged, then run:

```bash
source /opt/ros/humble/setup.bash
source ~/predator_prey_ws/install/setup.bash
cd ~/predator_prey_ws/src/Predator-Prey-Robotics

python3 run_simulation.py --config /tmp/privileged_smoke.yaml
```

A successful smoke test should spawn three red predators, one green prey, and four arena walls; execute two short candidate episodes; and create training output under `training_logs/privileged_smoke/`.

## 5. Train the three observation models

Run only one experiment at a time. The experiments use the same Gazebo entity names and must not be executed concurrently.

### Privileged observation

```bash
python3 run_simulation.py --config configs/privileged.yaml
```

### 360° camera observation

```bash
python3 run_simulation.py --config configs/camera360.yaml
```

### First-person view

```bash
python3 run_simulation.py --config configs/fpv.yaml
```

The default thesis configuration for each model uses:

- 3 predators and 1 prey
- 2 m × 2 m arena
- 13 CMA-ES candidates per generation
- 100 generations
- 1 episode per candidate
- 30 simulation seconds per episode
- 0.2-second fitness sampling interval

This gives **1,300 training episodes per observation model**.

### Training outputs

For a mode such as `privileged`, training creates:

```text
best_policy_privileged.npy
current_policy_privileged.npy
training_logs/privileged/fitness_history.csv
training_logs/privileged/fitness_curve.png
training_logs/privileged/top_policies/
```

The `top_policies` directory contains the five highest single-episode training policies retained across the complete run.

## 6. Evaluate a saved policy once

Use this command for a quick inspection of one `.npy` policy:

```bash
python3 evaluate_policy_once.py \
    privileged \
    best_policy_privileged.npy
```

Examples for selected policies:

```bash
python3 evaluate_policy_once.py \
    fpv \
    results/controller_selection/selected_policy_fpv.npy

python3 evaluate_policy_once.py \
    camera360 \
    results/controller_selection/selected_policy_camera360.npy
```

Append the result to a CSV file with `--output`:

```bash
python3 evaluate_policy_once.py \
    privileged \
    best_policy_privileged.npy \
    --output results/single_evaluations.csv
```

## 7. Select the best retained controller

The thesis selection procedure evaluates each of the five retained policies for ten episodes and selects the policy with the highest mean fitness.

```bash
python3 evaluate_top_policies.py \
    --model privileged \
    --episodes 10 \
    --overwrite

python3 evaluate_top_policies.py \
    --model camera360 \
    --episodes 10 \
    --overwrite

python3 evaluate_top_policies.py \
    --model fpv \
    --episodes 10 \
    --overwrite
```

Selection output is written to:

```text
results/controller_selection/
├── selection_episodes_<model>.csv
├── selection_summary_<model>.csv
├── selected_policy_<model>.npy
└── selected_policy_<model>.json
```

The scripts are resumable. Without `--overwrite`, already completed episodes are skipped when the same policy files and output data are present.

## 8. Run the final evaluation

The final procedure evaluates each selected controller in a fresh set of ten episodes. It also records ground-truth trajectories for every robot.

```bash
python3 evaluate_final_policy.py \
    privileged \
    results/controller_selection/selected_policy_privileged.npy \
    --episodes 10 \
    --overwrite

python3 evaluate_final_policy.py \
    camera360 \
    results/controller_selection/selected_policy_camera360.npy \
    --episodes 10 \
    --overwrite

python3 evaluate_final_policy.py \
    fpv \
    results/controller_selection/selected_policy_fpv.npy \
    --episodes 10 \
    --overwrite
```

Outputs are written to:

```text
results/final_evaluation/<model>/
├── episodes.csv
├── summary.csv
├── summary.json
└── trajectories/
    ├── episode_01.csv
    ├── episode_02.csv
    └── ...
```

## 9. Plot a representative trajectory

By default, the plotting script selects the final-evaluation episode whose fitness is closest to the median fitness for that model:

```bash
python3 plot_representative_trajectory.py privileged
python3 plot_representative_trajectory.py camera360
python3 plot_representative_trajectory.py fpv
```

Select a specific episode:

```bash
python3 plot_representative_trajectory.py privileged --episode 4
```

Set a custom output path:

```bash
python3 plot_representative_trajectory.py \
    privileged \
    --output results/privileged_trajectory.png
```

## Configuration

The experiment configurations are stored in `configs/*.yaml`.

Important fields include:

```yaml
experiment:
  mode: privileged

arena:
  size: 2.0

predators:
  count: 3

observation:
  type: privileged

optimizer:
  type: cmaes
  sigma: 0.5

training:
  pop_size: 13
  generations: 100
  evals_per_candidate: 1
  episode_duration: 30.0
  sample_dt: 0.2
```

The three standard modes are:

| Mode | Observation source | Policy module |
|---|---|---|
| `privileged` | Ground-truth odometry and geometry | `policies.policy_privileged` |
| `camera360` | Front, left, right, and rear cameras | `policies.policy_camera360` |
| `fpv` | Front camera only | `policies.policy_fpv` |

## Main ROS 2 topics

Predator command topics:

```text
/predator_0/cmd_vel
/predator_1/cmd_vel
/predator_2/cmd_vel
```

Ground-truth odometry:

```text
/predator_0/ground_truth/odom
/predator_1/ground_truth/odom
/predator_2/ground_truth/odom
/prey_0/ground_truth/odom
```

FPV camera topic:

```text
/predator_<i>/camera_front_sensor/image_raw
```

The 360° model additionally uses:

```text
/predator_<i>/camera_left_sensor/image_raw
/predator_<i>/camera_right_sensor/image_raw
/predator_<i>/camera_rear_sensor/image_raw
```

Inspect available topics with:

```bash
ros2 topic list
```

## Experimental details

### Predator controller

Each predator uses the same feedforward neural network:

```text
3 inputs → 4 hidden units → 2 wheel-speed outputs
```

Including biases, the network contains 26 trainable parameters.

### Observation inputs

All three models return three controller inputs, but their meanings differ:

- **Privileged:** signed nearest-teammate distance, prey bearing, and prey distance.
- **FPV:** signed visual teammate closeness, visual prey bearing, and visual prey-closeness proxy.
- **360°:** the same visual features as FPV, extracted from a stitched four-camera panorama.

### Prey controller

The prey uses a fixed Gaussian danger-field controller. The default parameters are:

```text
sigma_w = 0.20
sigma_p = 0.25
alpha   = 0.10
```

### Fitness

At every sample, fitness combines:

1. the mean reciprocal predator–prey distance; and
2. the mean distance from each predator to its nearest teammate.

The first term rewards proximity to the prey. The second discourages all predators from clustering in the same location.

## Thesis results included in the repository

The selected controllers obtained the following final-evaluation results over ten episodes:

| Observation model | Mean fitness | Sample standard deviation |
|---|---:|---:|
| Privileged | 3.07 | 0.52 |
| 360° camera | 2.89 | 0.45 |
| FPV | 2.65 | 0.42 |

A Kruskal–Wallis test did not detect a statistically significant difference between the three final fitness-score distributions (`H(2) = 2.511`, `p = 0.285`). Because only one evolutionary run was conducted per model, these results should be interpreted as an exploratory comparison rather than a general ranking of observation models.

## Reproducibility notes

- The standard configurations use fixed robot starting positions and orientations.
- Each training candidate is evaluated in only one episode.
- Gazebo physics, ROS 2 callback timing, message timing, and operating-system scheduling can introduce execution-level nondeterminism.
- Repeating a complete evolutionary run may therefore produce a different controller and a different performance ordering.
- The statistical analysis in the thesis concerns repeated executions of the three selected controllers, not variability across multiple evolutionary runs.

## Troubleshooting

### `Package 'thymio_description' not found`

Rebuild and source the workspace:

```bash
cd ~/predator_prey_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 pkg prefix thymio_description
```

### Gazebo services are missing

Confirm that Gazebo was started through `gazebo_ros` and wait for it to finish loading:

```bash
ros2 service list | grep gazebo
```

Training cannot start correctly without the entity-spawn and reset services.

### `No /clock received`

Make sure Gazebo is running and not paused. The controllers and evaluation timers use simulation time.

```bash
ros2 topic echo /clock --once
```

### Ground-truth odometry is missing

Check the topics:

```bash
ros2 topic list | grep ground_truth
```

The robot spawning code enables the ground-truth odometry plugin through the `publish_ground_truth:=true` Xacro argument.

### Camera topics are missing

```bash
ros2 topic list | grep image_raw
```

For FPV, every predator must publish a front-camera image. For the 360° condition, all four directional camera topics are expected.

### `ModuleNotFoundError: No module named 'cma'`

```bash
python3 -m pip install --user cma
```

### `cv_bridge` or OpenCV import problems

Prefer the Ubuntu/ROS packages installed earlier. Avoid a fully isolated virtual environment, or create it with `--system-site-packages`.

### A previous interrupted run left controller processes active

```bash
pkill -f nn_controller.py || true
pkill -f prey_controller.py || true
```

Restart Gazebo if entity services or robot states no longer respond correctly.

## Citation

```bibtex
@thesis{konditerovs2026observation,
  author = {Georgijs Konditerovs},
  title = {Observation Models in Evolutionary Predator--Prey Robotics: An Exploratory Comparison of Privileged, 360-Degree Camera, and First-Person View Observations},
  school = {Vrije Universiteit Amsterdam},
  type = {Bachelor's thesis},
  year = {2026}
}
```

## Acknowledgements

The Thymio model included in this repository is based on work from the ROS2swarm `thymio_description` project and its upstream robot-description sources.

The predator–prey setup and Gaussian prey-controller design were inspired by:

G. Lan, J. Chen, and A. E. Eiben, “Simulated and Real-World Evolution of Predator Robots,” *2019 IEEE Symposium Series on Computational Intelligence*, pp. 1974–1981, 2019.
