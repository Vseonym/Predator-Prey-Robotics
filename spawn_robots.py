import os
import math
import subprocess

MAX_PREDATORS_TO_CLEAR = 8


def safe_call(cmd, timeout=5):
    try:
        result = subprocess.run(cmd, shell=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {cmd}")
        return False


def clear_simulation():
    print("Clearing existing robots...")
    names = [f"predator_{i}" for i in range(MAX_PREDATORS_TO_CLEAR)] + ["prey_0"]
    for name in names:
        safe_call(
            f"ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "
            f"\"{{name: '{name}'}}\"",
            timeout=3,
        )


def reset_world():
    safe_call("ros2 service call /reset_world std_srvs/srv/Empty", timeout=5)


def spawn_robot(name, x, y, color="Gazebo/White", yaw=math.pi / 2):
    urdf_path = f"/tmp/{name}.urdf"
    ok = safe_call(
        f"xacro $(ros2 pkg prefix thymio_description)/share/thymio_description/urdf/thymio.urdf.xacro "
        f"name:={name} body_color:={color} > {urdf_path}",
        timeout=5,
    )
    if not ok or not os.path.exists(urdf_path) or os.path.getsize(urdf_path) == 0:
        raise RuntimeError(f"URDF generation failed for {name}")

    ok = safe_call(
        f"ros2 run gazebo_ros spawn_entity.py "
        f"-entity {name} -file {urdf_path} "
        f"-x {x} -y {y} -z 0.1 -Y {yaw}",
        timeout=10,
    )
    if not ok:
        raise RuntimeError(f"Failed to spawn {name}")


def predator_start_positions(predator_count, arena_size=2.0):
    """
    Paper-style setup in a 2m x 2m arena:
    prey starts at center, predators start parallel near one wall.
    """
    half = arena_size / 2.0
    y = -half + 0.20

    if predator_count == 1:
        xs = [0.0]
    else:
        usable_width = arena_size * 0.55
        start_x = -usable_width / 2.0
        step = usable_width / (predator_count - 1)
        xs = [start_x + i * step for i in range(predator_count)]

    return [(x, y) for x in xs]


def spawn_default_world(predator_count=3, arena_size=2.0):
    for idx, (x, y) in enumerate(predator_start_positions(predator_count, arena_size)):
        spawn_robot(f"predator_{idx}", x, y, "Gazebo/Red", yaw=math.pi / 2)

    spawn_robot("prey_0", 0.0, 0.0, "Gazebo/Green", yaw=math.pi / 2)