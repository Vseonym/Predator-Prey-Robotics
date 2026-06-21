import numpy as np

# Paper-style controller:
# inputs: r, delta_theta, d
# hidden: 4
# outputs: left/right wheel angular velocity
# N_WEIGHTS = 3*4 + 4 + 4*2 + 2 = 26, matching the paper report.
INPUT_SIZE = 3
HIDDEN_SIZE = 4
OUTPUT_SIZE = 2

N_WEIGHTS = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE


def unpack_weights(genome):
    genome = np.asarray(genome, dtype=np.float32)
    idx = 0
    w1 = genome[idx:idx + INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += INPUT_SIZE * HIDDEN_SIZE
    b1 = genome[idx:idx + HIDDEN_SIZE]
    idx += HIDDEN_SIZE
    w2 = genome[idx:idx + HIDDEN_SIZE * OUTPUT_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    idx += HIDDEN_SIZE * OUTPUT_SIZE
    b2 = genome[idx:idx + OUTPUT_SIZE]
    return w1, b1, w2, b2


def nn_forward(features, genome, max_wheel_speed=8.0):
    w1, b1, w2, b2 = unpack_weights(genome)
    x = np.asarray(features, dtype=np.float32)
    h = np.tanh(x @ w1 + b1)
    out = np.tanh(0.25 * (h @ w2 + b2))
    return float(out[0] * max_wheel_speed), float(out[1] * max_wheel_speed)