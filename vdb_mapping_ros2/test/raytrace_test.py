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
import struct
import rclpy
from rclpy.node import Node
from smoke_test import make_cloud  # reuse wall cloud builder
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vdb_mapping_interfaces.srv import Raytrace, GetMapSection


class RTTester(Node):
    def __init__(self):
        super().__init__("vdb_rt_tester")
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


def main():
    rclpy.init()
    n = RTTester()
    # feed the wall until integrated
    for _ in range(30):
        n.cloud_pub.publish(make_cloud(n.get_clock().now().to_msg(), "velodyne"))
        rclpy.spin_once(n, timeout_sec=0.25)

    cli = n.create_client(Raytrace, "/vdb_mapping_ros2/raytrace")
    if not cli.wait_for_service(timeout_sec=5.0):
        print("FAIL: raytrace service unavailable")
        return 1
    req = Raytrace.Request()
    req.header.frame_id = "map"
    req.ray.origin.x = 0.0
    req.ray.origin.y = 0.0
    req.ray.origin.z = 0.5
    req.ray.direction.x = 1.0
    req.ray.max_ray_length = 10.0
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
    if not fut.done():
        print("FAIL: raytrace timed out")
        return 1
    r = fut.result()
    p = r.end_point
    print(f"raytrace: success={r.success} end_point=({p.x:.3f},{p.y:.3f},{p.z:.3f})")
    ok = r.success and abs(p.x - 2.0) <= 0.075 and abs(p.y) <= 0.075 and abs(p.z - 0.5) <= 0.075
    # also exercise get_map_section
    gcli = n.create_client(GetMapSection, "/vdb_mapping_ros2/get_map_section")
    greq = GetMapSection.Request()
    greq.header.frame_id = "map"
    greq.bounding_box.min_corner.x = -1.0
    greq.bounding_box.min_corner.y = -2.0
    greq.bounding_box.min_corner.z = -1.0
    greq.bounding_box.max_corner.x = 3.0
    greq.bounding_box.max_corner.y = 2.0
    greq.bounding_box.max_corner.z = 2.0
    gfut = gcli.call_async(greq)
    rclpy.spin_until_future_complete(n, gfut, timeout_sec=10.0)
    gr = gfut.result() if gfut.done() else None
    if gr is None or not gr.success or len(gr.section.map) == 0:
        print("FAIL: get_map_section")
        ok = False
    else:
        print(f"get_map_section: success={gr.success} bytes={len(gr.section.map)}")
    print("SERVICE TEST:", "PASS" if ok else "FAIL")
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
