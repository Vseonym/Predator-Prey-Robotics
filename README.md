# Predator-prey observation-model refactor

This generated structure supports three thesis experiments with one runner:

```bash
python3 run_simulation.py --config configs/privileged.yaml
python3 run_simulation.py --config configs/fpv.yaml
python3 run_simulation.py --config configs/camera360.yaml
```

Fixed across all experiments:

- 2m x 2m arena
- same Gaussian prey controller
- same GA implementation
- same paper-style ground-truth fitness
- same robot model/spawning logic

Experimental variable:

- `privileged`: no camera input, uses Gazebo model states for `[r, delta_theta, d]`
- `fpv`: front camera only: `/<robot>/camera/image_raw`
- `camera360`: front + left + right + rear cameras:
  - `/<robot>/camera/image_raw`
  - `/<robot>/camera_left/image_raw`
  - `/<robot>/camera_right/image_raw`
  - `/<robot>/camera_rear/image_raw`

Important:

- This assumes Gazebo publishes model states on `/model_states`. If your topic is different, change `fitness.model_states_topic` in each config.
- Copy this folder's files into the root where you currently run `run_simulation.py`.
- Keep your `thymio_description` package installed/source-built as before.