from observations.privileged_observation import PrivilegedObservation
from observations.fpv_observation import FPVObservation
from observations.camera360_observation import Camera360Observation


def build_observation(node, observation_type, robot_name, predator_count, model_states_topic="/model_states"):
    if observation_type == "privileged":
        return PrivilegedObservation(node, robot_name, predator_count, model_states_topic=model_states_topic)
    if observation_type == "fpv":
        return FPVObservation(node, robot_name, predator_count)
    if observation_type == "camera360":
        return Camera360Observation(node, robot_name, predator_count)
    raise ValueError(f"Unknown observation_type: {observation_type}")