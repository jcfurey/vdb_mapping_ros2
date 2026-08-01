#!/usr/bin/env python3
# End-to-end assembler test: feeds one keyframe of synthetic evidence through
# the real node and asserts the three properties the audit found broken:
#  1. survey METADATA survives to the published product (texture, incidence,
#     elevation bounds and resolvedness are the input values, not zeros -
#     pcl::VoxelGrid used to zero every custom field);
#  2. the odom-delta re-anchoring places the survey point at
#     kf_pose * (T_odom_kf^-1 * T_odom_e * p);
#  3. a z-only trajectory correction re-renders the product at the new depth
#     (the xy/yaw gate used to swallow it).
# Exits 0 on success; the launch wrapper asserts the exit code.
import math
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster

T_E = 10.0      # survey evidence stamp
T_KF = 10.5     # keyframe stamp
ODOM_X_E = 1.0  # vehicle odom x at the evidence stamp
ODOM_X_KF = 2.0
MAP_KF = (2.0, 0.0, 0.0)   # optimized keyframe pose (translation)
Z_CORRECTION = -1.0
# survey input point, robot frame at T_E, with distinct metadata everywhere
P_ROBOT = (1.0, 0.0, 0.0)
META = dict(intensity=0.5, range=5.0, incidence=0.7, survey_fallback=0.0,
            texture=0.25, texture_squared=0.0725,  # variance 0.01
            elevation_lo_offset=-0.4, elevation_hi_offset=0.2,
            elevation_resolved=1.0)
# expected map position: T_kf_e = inv([2,0,0]) * [1,0,0] -> p_kf = (0,0,0);
# survey point lands at MAP_KF (then at MAP_KF + z correction)
EXPECT_XYZ = MAP_KF

SURVEY_FIELDS = [
    "x", "y", "z", "intensity", "range", "incidence", "survey_fallback",
    "texture", "texture_squared", "elevation_lo_offset",
    "elevation_hi_offset", "elevation_resolved"]
SURVEY_OFFSETS = [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48]


def make_survey_cloud(stamp_s):
    msg = PointCloud2()
    msg.header.stamp.sec = int(stamp_s)
    msg.header.stamp.nanosec = int((stamp_s - int(stamp_s)) * 1e9)
    msg.header.frame_id = "base_link"
    msg.height = 1
    msg.width = 1
    msg.is_bigendian = False
    msg.is_dense = True
    for name, off in zip(SURVEY_FIELDS, SURVEY_OFFSETS):
        f = PointField()
        f.name = name
        f.offset = off
        f.datatype = PointField.FLOAT32
        f.count = 1
        msg.fields.append(f)
    msg.point_step = 52
    msg.row_step = 52
    row = bytearray(52)
    vals = dict(META)
    vals["x"], vals["y"], vals["z"] = P_ROBOT
    for name, off in zip(SURVEY_FIELDS, SURVEY_OFFSETS):
        struct.pack_into("<f", row, off, float(vals[name]))
    msg.data = bytes(row)
    return msg


def make_xyz_cloud(stamp_s, xyz):
    msg = PointCloud2()
    msg.header.stamp.sec = int(stamp_s)
    msg.header.stamp.nanosec = int((stamp_s - int(stamp_s)) * 1e9)
    msg.header.frame_id = "base_link"
    msg.height = 1
    msg.width = 1
    msg.is_bigendian = False
    msg.is_dense = True
    for i, name in enumerate(("x", "y", "z")):
        f = PointField()
        f.name = name
        f.offset = 4 * i
        f.datatype = PointField.FLOAT32
        f.count = 1
        msg.fields.append(f)
    msg.point_step = 12
    msg.row_step = 12
    msg.data = struct.pack("<3f", *xyz)
    return msg


def make_traj(kf_xyz):
    # [x y z roll pitch yaw i t] float32, t relative to the message stamp
    msg = PointCloud2()
    msg.header.stamp.sec = int(T_KF)
    msg.header.stamp.nanosec = int((T_KF - int(T_KF)) * 1e9)
    msg.header.frame_id = "map"
    msg.height = 1
    msg.width = 1
    msg.is_bigendian = False
    msg.is_dense = True
    for i, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw", "i", "t")):
        f = PointField()
        f.name = name
        f.offset = 4 * i
        f.datatype = PointField.FLOAT32
        f.count = 1
        msg.fields.append(f)
    msg.point_step = 32
    msg.row_step = 32
    msg.data = struct.pack("<8f", kf_xyz[0], kf_xyz[1], kf_xyz[2],
                           0.0, 0.0, 0.0, 0.0, 0.0)
    return msg


def parse_survey(msg):
    fields = {f.name: f.offset for f in msg.fields}
    rows = []
    for i in range(msg.width):
        base = i * msg.point_step
        row = {n: struct.unpack_from("<f", bytes(msg.data), base + off)[0]
               for n, off in fields.items()}
        rows.append(row)
    return rows


class Harness(Node):
    def __init__(self):
        super().__init__("assembler_harness")
        reliable = QoSProfile(depth=10,
                              reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.tfb = TransformBroadcaster(self)
        self.hits_pub = self.create_publisher(PointCloud2, "/test/hits", reliable)
        self.survey_pub = self.create_publisher(PointCloud2, "/test/survey", reliable)
        # the assembler's trajectory subscription is transient_local
        self.traj_pub = self.create_publisher(PointCloud2, "/test/traj", latched)
        self.snapshots = []
        self.create_subscription(
            PointCloud2, "/assembler/survey_pointcloud",
            lambda m: self.snapshots.append(parse_survey(m)), latched)

    def broadcast_odom(self):
        # exact-stamp odom -> base_link samples bracketing both stamps
        for stamp, x in ((T_E - 0.4, ODOM_X_E), (T_E, ODOM_X_E),
                         (T_KF, ODOM_X_KF), (T_KF + 0.4, ODOM_X_KF)):
            t = TransformStamped()
            t.header.stamp.sec = int(stamp)
            t.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
            t.header.frame_id = "odom"
            t.child_frame_id = "base_link"
            t.transform.translation.x = x
            t.transform.rotation.w = 1.0
            self.tfb.sendTransform(t)


def spin_until(node, pred, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if pred():
            return True
    return False


def close(a, b, tol=1e-3):
    return abs(a - b) <= tol


def main():
    rclpy.init()
    node = Harness()

    # let discovery settle, then keep TF fresh while feeding
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
        node.broadcast_odom()

    node.survey_pub.publish(make_survey_cloud(T_E))
    node.hits_pub.publish(make_xyz_cloud(T_KF, (3.0, 0.0, 0.0)))
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.1)
        node.broadcast_odom()
    node.traj_pub.publish(make_traj(MAP_KF))

    def keep_alive():
        node.broadcast_odom()
        return bool(node.snapshots)

    if not spin_until(node, keep_alive, 30.0):
        print("FAIL: no survey snapshot within 30 s", flush=True)
        return 1
    rows = node.snapshots[-1]
    if len(rows) != 1:
        print(f"FAIL: expected 1 survey row, got {len(rows)}", flush=True)
        return 1
    r = rows[0]
    checks = [
        ("x", EXPECT_XYZ[0]), ("y", EXPECT_XYZ[1]), ("z", EXPECT_XYZ[2]),
        ("intensity", META["intensity"]), ("range", META["range"]),
        ("incidence", META["incidence"]), ("texture", META["texture"]),
        ("elevation_lo_offset", META["elevation_lo_offset"]),
        ("elevation_hi_offset", META["elevation_hi_offset"]),
        ("elevation_resolved_fraction", META["elevation_resolved"]),
    ]
    for name, expect in checks:
        got = r.get(name)
        if got is None or not close(got, expect, 0.03):
            print(f"FAIL: survey field {name} = {got}, expected ~{expect} "
                  "(zeros here mean the voxel reduction dropped metadata)",
                  flush=True)
            return 1
    if not close(r.get("texture_variance", -1.0), 0.01, 0.005):
        print(f"FAIL: texture_variance = {r.get('texture_variance')}, "
              "expected ~0.01", flush=True)
        return 1
    if r.get("support", 0.0) < 0.5:
        print(f"FAIL: support = {r.get('support')}", flush=True)
        return 1
    print("[1] survey metadata + odom-delta anchoring OK", flush=True)

    # z-only correction must re-render the product at the new depth
    n_before = len(node.snapshots)
    node.traj_pub.publish(make_traj((MAP_KF[0], MAP_KF[1],
                                     MAP_KF[2] + Z_CORRECTION)))

    def z_moved():
        node.broadcast_odom()
        for snap in node.snapshots[n_before:]:
            if len(snap) == 1 and close(snap[0]["z"],
                                        EXPECT_XYZ[2] + Z_CORRECTION, 0.03):
                return True
        return False

    if not spin_until(node, z_moved, 30.0):
        zs = [s[0]["z"] for s in node.snapshots[n_before:] if len(s) == 1]
        print(f"FAIL: no re-rendered snapshot at z ~ {Z_CORRECTION}; "
              f"saw z = {zs} (a stale z means the pose gate ignored depth)",
              flush=True)
        return 1
    print("[2] z-only correction re-rendered OK", flush=True)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
