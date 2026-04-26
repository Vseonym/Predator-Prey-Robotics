import os
import math
import subprocess


def safe_call(cmd, timeout=5):
    try:
        result = subprocess.run(cmd, shell=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {cmd}")
        return False


def clear_simulation():
    print("Clearing existing robots...")

    names = [f"predator_{i}" for i in range(5)] + ["prey_0"]

    for name in names:
        safe_call(
            f"ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "
            f"\"{{name: '{name}'}}\"",
            timeout=3
        )


def reset_world():
    safe_call(
        "ros2 service call /reset_world std_srvs/srv/Empty",
        timeout=5
    )


def spawn_robot(name, x, y, color="Gazebo/White", yaw=math.pi / 2):
    urdf_path = f"/tmp/{name}.urdf"

    ok = safe_call(
        f"xacro $(ros2 pkg prefix thymio_description)/share/thymio_description/urdf/thymio.urdf.xacro "
        f"name:={name} body_color:={color} > {urdf_path}",
        timeout=5
    )

    if not ok or not os.path.exists(urdf_path) or os.path.getsize(urdf_path) == 0:
        raise RuntimeError(f"URDF generation failed for {name}")

    safe_call(
        f"ros2 run gazebo_ros spawn_entity.py "
        f"-entity {name} "
        f"-file {urdf_path} "
        f"-x {x} -y {y} -z 0.1 "
        f"-Y {yaw}",
        timeout=10
    )


def spawn_default_world():
    for i in range(5):
        spawn_robot(f"predator_{i}", i + 1, 0, "Gazebo/Red")

    spawn_robot("prey_0", 3, 3, "Gazebo/Green")