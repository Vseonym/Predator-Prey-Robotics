import os
import shutil
import subprocess
import signal
import time

from spawn_robots import clear_simulation, reset_world, spawn_default_world


processes = []


def start_controller(name):
    p = subprocess.Popen(
        ["python3", "nn_controller.py", "--ros-args", "-p", f"robot_name:={name}"]
    )
    processes.append(p)


def start_all_predator_controllers():
    for i in range(5):
        start_controller(f"predator_{i}")


def stop_all():
    global processes

    for i in range(5):
        subprocess.run(
            [
                "ros2", "topic", "pub", "--once",
                f"/predator_{i}/cmd_vel",
                "geometry_msgs/msg/Twist",
                "{linear: {x: 0.0}, angular: {z: 0.0}}",
            ],
            timeout=2.0,
        )

    time.sleep(0.2)

    for p in processes:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)

    for p in processes:
        try:
            p.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

    subprocess.run(["pkill", "-f", "nn_controller.py"])
    processes = []


def main():
    if not os.path.exists("best_policy.npy"):
        raise FileNotFoundError("best_policy.npy not found. Train first.")

    shutil.copyfile("best_policy.npy", "current_policy.npy")
    print("Loaded best_policy.npy into current_policy.npy")

    clear_simulation()
    time.sleep(1.0)

    spawn_default_world()
    time.sleep(2.0)

    reset_world()
    time.sleep(1.0)

    start_all_predator_controllers()

    print("Best policy running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping test...")
    finally:
        stop_all()


if __name__ == "__main__":
    main()