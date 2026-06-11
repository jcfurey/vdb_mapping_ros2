#!/usr/bin/env python3
# -- BEGIN LICENSE BLOCK ----------------------------------------------
# Copyright 2022 FZI Forschungszentrum Informatik
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -- END LICENSE BLOCK ------------------------------------------------
import math
import struct
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from std_msgs.msg import Header


def make_cloud(stamp, frame):
    pts = []
    z = 0.0
    while z <= 1.0:
        y = -1.0
        while y <= 1.0:
            pts.append((2.0, y, z))
            y += 0.02
        z += 0.02
    data = b"".join(struct.pack("fff", *p) for p in pts)
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame)
    msg.height = 1
    msg.width = len(pts)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(pts)
    msg.data = data
    msg.is_dense = True
    return msg


class Tester(Node):
    def __init__(self):
        super().__init__("vdb_smoke_tester")
        self.tf_bc = StaticTransformBroadcaster(self)
        tfs = []
        for child in ("base_link", "velodyne"):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = child
            t.transform.translation.z = 0.5 if child == "velodyne" else 0.0
            t.transform.rotation.w = 1.0
            tfs.append(t)
        self.tf_bc.sendTransform(tfs)

        self.cloud_pub = self.create_publisher(
            PointCloud2, "/cloud", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.occ = None
        self.marker = None
        self.create_subscription(
            OccupancyGrid, "/vdb_mapping_ros2/vdb_map_occupancy", self.on_occ, 1
        )
        self.create_subscription(
            Marker, "/vdb_mapping_ros2/vdb_map_visualization", self.on_marker, 1
        )
        self.sent = 0
        self.create_timer(0.25, self.tick)

    def tick(self):
        if self.sent < 40:
            self.cloud_pub.publish(
                make_cloud(self.get_clock().now().to_msg(), "velodyne")
            )
            self.sent += 1

    def on_occ(self, msg):
        self.occ = msg

    def on_marker(self, msg):
        self.marker = msg


def main():
    rclpy.init()
    n = Tester()
    end = n.get_clock().now().nanoseconds + int(20e9)
    while rclpy.ok() and n.get_clock().now().nanoseconds < end:
        rclpy.spin_once(n, timeout_sec=0.2)
        if n.occ is not None and n.marker is not None and n.sent >= 20:
            break

    ok = True
    if n.occ is None:
        print("FAIL: no occupancy grid received")
        ok = False
    else:
        g = n.occ
        res = g.info.resolution
        ox, oy = g.info.origin.position.x, g.info.origin.position.y
        # origin must sit on a (chunk-aligned index - 0.5) * resolution lattice
        fx = (ox / res) % 32.0
        fy = (oy / res) % 32.0
        counts = {-1: 0, 0: 0, 100: 0}
        for v in g.data:
            counts[v] = counts.get(v, 0) + 1
        print(
            f"occupancy: {g.info.width}x{g.info.height} res={res:.3f} "
            f"origin=({ox:.4f},{oy:.4f}) frac=({fx:.2f},{fy:.2f}) counts={counts}"
        )
        if abs(fx - 31.5) > 1e-3 or abs(fy - 31.5) > 1e-3:
            print("FAIL: origin not on half-voxel chunk lattice")
            ok = False
        if counts[100] == 0:
            print("FAIL: no lethal cells (wall missing)")
            ok = False
        if counts[0] == 0:
            print("FAIL: no free cells (raytraced space missing)")
            ok = False
        # the wall is at x=2.0 -> voxel center index 40 -> lethal cells must
        # appear in grid column round((2.0 - ox)/res - 0.5)
        col = round((2.0 - ox) / res - 0.5)
        lethal_cols = set()
        for i, v in enumerate(g.data):
            if v == 100:
                lethal_cols.add(i % g.info.width)
        print(f"expected wall column={col} lethal columns={sorted(lethal_cols)}")
        if col not in lethal_cols:
            print("FAIL: wall not in expected column")
            ok = False
    if n.marker is None:
        print("FAIL: no marker received")
        ok = False
    else:
        print(f"marker: action={n.marker.action} points={len(n.marker.points)}")
        if len(n.marker.points) == 0 and n.marker.action != Marker.DELETE:
            print("FAIL: empty marker without DELETE action")
            ok = False
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
