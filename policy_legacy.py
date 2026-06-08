import numpy as np

# Inputs:
# 0  prey_visible
# 1  prey_x
# 2  prey_area
# 3  red_visible
# 4  red_x
# 5  red_area
# 6  proximity_center
# 7  proximity_center_left
# 8  proximity_center_right
# 9  proximity_left
# 10 proximity_right
# 11 robot_role_value
#
# robot_role_value:
#   predator_0 = -1.0
#   predator_1 = -0.5
#   predator_2 =  0.0
#   predator_3 =  0.5
#   predator_4 =  1.0
#
# This allows one shared NN policy to learn role-dependent behaviour.
INPUT_SIZE = 12
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2

N_WEIGHTS = (
    INPUT_SIZE * HIDDEN_SIZE +
    HIDDEN_SIZE +
    HIDDEN_SIZE * OUTPUT_SIZE +
    OUTPUT_SIZE
)


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


def nn_forward(features, genome):
    w1, b1, w2, b2 = unpack_weights(genome)

    x = np.asarray(features, dtype=np.float32)

    h = np.tanh(x @ w1 + b1)
    out = np.tanh(h @ w2 + b2)

    max_wheel_speed = 8.0

    omega_left = out[0] * max_wheel_speed
    omega_right = out[1] * max_wheel_speed

    return omega_left, omega_right