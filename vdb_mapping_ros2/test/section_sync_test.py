#!/usr/bin/env python3
# End-to-end remote section sync: mapper A integrates a wall and publishes
# sections; remote B applies them. Pins the whole remote path — including
# the tile-clearing fix: after A's map is RESET (its sections then carry no
# active values over the wall region), B must clear the wall too, even
# though integration pruning may have collapsed it into tiles.
# Exits 0 on success; the launch wrapper asserts the exit code.
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster


def make_wall_cloud(stamp_s):
    # a small wall slab at x = 2.0, spanning y/z, in the sensor frame
    pts = []
    for i in range(9):
        for j in range(5):
            pts.append((2.0, -0.2 + 0.05 * i, -0.1 + 0.05 * j))
    msg = PointCloud2()
    msg.header.stamp.sec = int(stamp_s)
    msg.header.stamp.nanosec = int((stamp_s - int(stamp_s)) * 1e9)
    msg.header.frame_id = "sensor"
    msg.height = 1
    msg.width = len(pts)
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
    msg.row_step = 12 * len(pts)
    msg.data = b"".join(struct.pack("<3f", *p) for p in pts)
    return msg


def count_near_wall(msg):
    fields = {f.name: f.offset for f in msg.fields}
    n = 0
    data = bytes(msg.data)
    for i in range(msg.width * max(1, msg.height)):
        base = i * msg.point_step
        x = struct.unpack_from("<f", data, base + fields["x"])[0]
        if 1.8 <= x <= 2.2:
            n += 1
    return n


class Harness(Node):
    def __init__(self):
        super().__init__("section_harness")
        reliable = QoSProfile(depth=10,
                              reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        self.cloud_pub = self.create_publisher(PointCloud2, "/cloud", reliable)
        self.tfb = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp.sec = 0
        t.header.frame_id = "map"
        t.child_frame_id = "sensor"
        t.transform.rotation.w = 1.0
        self.tfb.sendTransform(t)
        t2 = TransformStamped()
        t2.header.stamp.sec = 0
        t2.header.frame_id = "map"
        t2.child_frame_id = "base_link"
        t2.transform.rotation.w = 1.0
        self.tfb.sendTransform(t2)
        self.remote_wall = None
        self.create_subscription(
            PointCloud2, "/remote/vdb_map_pointcloud",
            lambda m: setattr(self, "remote_wall", count_near_wall(m)),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST))
        self.reset_cli = self.create_client(Trigger, "/vdb_mapping/reset_map")


def spin_until(node, pred, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if pred():
            return True
    return False


def main():
    rclpy.init()
    node = Harness()
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)

    # [1] wall into A -> sections -> B shows it
    stamp = 50.0
    for k in range(6):
        node.cloud_pub.publish(make_wall_cloud(stamp + 0.2 * k))
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.1)
    if not spin_until(node, lambda: (node.remote_wall or 0) > 10, 40.0):
        print(f"FAIL: remote map never showed the wall "
              f"(near-wall points: {node.remote_wall})", flush=True)
        return 1
    print(f"[1] wall propagated to remote ({node.remote_wall} pts)",
          flush=True)

    # [2] reset A -> its sections stop carrying the wall -> B clears it,
    # even where integration pruning collapsed the block into tiles
    if not node.reset_cli.wait_for_service(timeout_sec=10.0):
        print("FAIL: reset_map service unavailable", flush=True)
        return 1
    fut = node.reset_cli.call_async(Trigger.Request())
    if not spin_until(node, lambda: fut.done(), 10.0):
        print("FAIL: reset_map did not answer", flush=True)
        return 1
    node.remote_wall = None
    if not spin_until(node,
                      lambda: node.remote_wall is not None and
                              node.remote_wall <= 2, 40.0):
        print(f"FAIL: remote map kept the wall after the source reset "
              f"(near-wall points: {node.remote_wall})", flush=True)
        return 1
    print("[2] remote cleared after source reset", flush=True)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
