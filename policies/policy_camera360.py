import numpy as np

# 4 cameras * 6 vision features + 5 proximity values + role = 30 inputs.
INPUT_SIZE = 30
HIDDEN_SIZE = 12
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
    out = np.tanh(h @ w2 + b2)
    return float(out[0] * max_wheel_speed), float(out[1] * max_wheel_speed)