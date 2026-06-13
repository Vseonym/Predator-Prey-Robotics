import math


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def pose_xy_yaw(pose):
    return pose.position.x, pose.position.y, quat_to_yaw(pose.orientation)


def odom_xy_yaw(odom_msg):
    return pose_xy_yaw(odom_msg.pose.pose)


class OdomStore:
    """
    Small helper for storing latest ground-truth odometry by robot name.

    Expected topics:
      /predator_0/ground_truth/odom
      /predator_1/ground_truth/odom
      ...
      /prey_0/ground_truth/odom
    """

    def __init__(self, node, robot_names):
        from nav_msgs.msg import Odometry

        self.node = node
        self.robot_names = list(robot_names)
        self.odom = {name: None for name in self.robot_names}

        for name in self.robot_names:
            node.create_subscription(
                Odometry,
                f"/{name}/ground_truth/odom",
                lambda msg, robot=name: self.odom_callback(msg, robot),
                10,
            )

    def odom_callback(self, msg, robot_name):
        self.odom[robot_name] = msg

    def has_all(self, names=None):
        names = self.robot_names if names is None else names
        return all(self.odom.get(name) is not None for name in names)

    def xy_yaw(self, robot_name):
        msg = self.odom.get(robot_name)
        if msg is None:
            return None
        return odom_xy_yaw(msg)