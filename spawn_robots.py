import os
import math
import subprocess
from pathlib import Path

MAX_PREDATORS_TO_CLEAR = 8

WALL_NAMES = [
    "wall_north",
    "wall_south",
    "wall_east",
    "wall_west",
]


def safe_call(cmd, timeout=5):
    try:
        result = subprocess.run(cmd, shell=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {cmd}")
        return False


def clear_simulation():
    print("Clearing existing robots and walls...")

    names = (
        [f"predator_{i}" for i in range(MAX_PREDATORS_TO_CLEAR)]
        + ["prey_0"]
        + WALL_NAMES
    )

    for name in names:
        safe_call(
            f"ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "
            f"\"{{name: '{name}'}}\"",
            timeout=3,
        )


def reset_world():
    safe_call("ros2 service call /reset_world std_srvs/srv/Empty", timeout=5)


def spawn_sdf(name, sdf_path, x, y, z=0.0, yaw=0.0):
    ok = safe_call(
        f"ros2 run gazebo_ros spawn_entity.py "
        f"-entity {name} "
        f"-file {sdf_path} "
        f"-x {x} -y {y} -z {z} "
        f"-Y {yaw}",
        timeout=10,
    )

    if not ok:
        raise RuntimeError(f"Failed to spawn {name}")


def write_wall_sdf(name, length, thickness, height, color="Gazebo/Grey"):
    sdf_path = Path(f"/tmp/{name}.sdf")

    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>{length} {thickness} {height}</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{length} {thickness} {height}</size>
          </box>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>{color}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    sdf_path.write_text(sdf)
    return str(sdf_path)


def spawn_walls(arena_size=2.0):
    """
    Spawn a physical 2m x 2m square arena.

    The usable inside region is approximately:
      x in [-1.0, 1.0]
      y in [-1.0, 1.0]

    Walls are centered on the boundary lines.
    """
    half = arena_size / 2.0

    wall_thickness = 0.05
    wall_height = 0.20

    # Make walls slightly longer so corners close properly.
    long_side = arena_size + wall_thickness

    horizontal_sdf = write_wall_sdf(
        name="horizontal_wall_template",
        length=long_side,
        thickness=wall_thickness,
        height=wall_height,
    )

    vertical_sdf = write_wall_sdf(
        name="vertical_wall_template",
        length=long_side,
        thickness=wall_thickness,
        height=wall_height,
    )

    # North/south walls: long side along X.
    spawn_sdf(
        "wall_north",
        horizontal_sdf,
        x=0.0,
        y=half,
        z=wall_height / 2.0,
        yaw=0.0,
    )

    spawn_sdf(
        "wall_south",
        horizontal_sdf,
        x=0.0,
        y=-half,
        z=wall_height / 2.0,
        yaw=0.0,
    )

    # East/west walls: rotate same box 90 degrees so long side goes along Y.
    spawn_sdf(
        "wall_east",
        vertical_sdf,
        x=half,
        y=0.0,
        z=wall_height / 2.0,
        yaw=math.pi / 2.0,
    )

    spawn_sdf(
        "wall_west",
        vertical_sdf,
        x=-half,
        y=0.0,
        z=wall_height / 2.0,
        yaw=math.pi / 2.0,
    )


def spawn_robot(name, x, y, color="Gazebo/White", yaw=math.pi / 2):
    urdf_path = f"/tmp/{name}.urdf"

    ok = safe_call(
        f"xacro $(ros2 pkg prefix thymio_description)/share/thymio_description/urdf/thymio.urdf.xacro "
        f"name:={name} body_color:={color} publish_ground_truth:=true > {urdf_path}",
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

    # Slightly away from wall so robots do not spawn inside it.
    y = -half + 0.25

    if predator_count == 1:
        xs = [0.0]
    else:
        usable_width = arena_size * 0.55
        start_x = -usable_width / 2.0
        step = usable_width / (predator_count - 1)
        xs = [start_x + i * step for i in range(predator_count)]

    return [(x, y) for x in xs]


def spawn_default_world(predator_count=3, arena_size=2.0):
    spawn_walls(arena_size=arena_size)

    for idx, (x, y) in enumerate(predator_start_positions(predator_count, arena_size)):
        spawn_robot(f"predator_{idx}", x, y, "Gazebo/Red", yaw=math.pi / 2)

    spawn_robot("prey_0", 0.0, 0.0, "Gazebo/Green", yaw=math.pi / 2)