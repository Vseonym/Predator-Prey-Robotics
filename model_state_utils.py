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


def model_pose_dict(model_states_msg):
    return {name: pose for name, pose in zip(model_states_msg.name, model_states_msg.pose)}