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
import os
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

T_E = 10.0      # survey evidence stamp
T_KF = 10.5     # keyframe stamp
ODOM_X_E = 1.0  # vehicle odom x at the evidence stamp
ODOM_X_KF = 2.0
MAP_KF = (2.0, 0.0, 0.0)   # optimized keyframe pose (translation)
Z_CORRECTION = -1.0
NAV_EXPORT = "/tmp/assembler_test_navigation.pcd"
SURFEL_EXPORT = "/tmp/assembler_test_navigation_surfels.pcd"
# survey input point, robot frame at T_E, with distinct metadata everywhere
P_ROBOT = (1.0, 0.0, 0.0)
META = dict(intensity=0.5, range=5.0, incidence=0.7, survey_fallback=0.0,
            texture=0.25, texture_squared=0.0725,  # variance 0.01
            elevation_lo_offset=-0.4, elevation_hi_offset=0.2,
            elevation_resolved=1.0)
# expected map position: T_kf_e = inv([2,0,0]) * [1,0,0] -> p_kf = (0,0,0);
# survey point lands at MAP_KF (then at MAP_KF + z correction)
EXPECT_XYZ = MAP_KF
# A dense, survey-only wall patch proves the live/export navigation product is
# sourced from the centimetre supported survey, not the independent 20 cm
# aperture-ribbon reconstruction used by the occupancy test below. The odom
# delta adds +1 m in map x, so robot-frame (6,3,0) lands at map (7,3,0).
NAV_SURVEY_XYZ = (7.0, 3.0, 0.0)
# A separate supported return outside every clearing ray deliberately has no
# neighbors. Dense navigation must keep it with normal_valid=0; strict surfels
# must omit it. The same +1 m odom delta places it at map (9,-3,0).
ISOLATED_SURVEY_XYZ = (9.0, -3.0, 0.0)

SURVEY_FIELDS = [
    "x", "y", "z", "intensity", "range", "incidence", "survey_fallback",
    "texture", "texture_squared", "elevation_lo_offset",
    "elevation_hi_offset", "elevation_resolved"]
SURVEY_OFFSETS = [0, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48]
TILE_FIELDS = [
    "x", "y", "z", "intensity", "range", "azimuth",
    "vertical_uncertainty"]
RECONSTRUCTION_FIELDS = TILE_FIELDS + [
    "range_sigma", "prominence", "echo_width"]
TILE_ANGLES = (-10.0, 0.0, 10.0)


def make_survey_cloud(stamp_s):
    msg = PointCloud2()
    msg.header.stamp.sec = int(stamp_s)
    msg.header.stamp.nanosec = int((stamp_s - int(stamp_s)) * 1e9)
    msg.header.frame_id = "base_link"
    msg.height = 1
    points = [P_ROBOT, (8.0, -3.0, 0.0)]
    for dy in (-0.08, -0.04, 0.0, 0.04, 0.08):
        for dz in (-0.08, -0.04, 0.0, 0.04, 0.08):
            points.append((6.0, 3.0 + dy, dz))
    msg.width = len(points)
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
    msg.row_step = msg.point_step * msg.width
    rows = []
    for xyz in points:
        row = bytearray(52)
        vals = dict(META)
        vals["x"], vals["y"], vals["z"] = xyz
        for name, off in zip(SURVEY_FIELDS, SURVEY_OFFSETS):
            struct.pack_into("<f", row, off, float(vals[name]))
        rows.append(bytes(row))
    msg.data = b"".join(rows)
    return msg


def make_xyz_cloud(stamp_s, xyzs):
    msg = PointCloud2()
    msg.header.stamp.sec = int(stamp_s)
    msg.header.stamp.nanosec = int((stamp_s - int(stamp_s)) * 1e9)
    msg.header.frame_id = "base_link"
    msg.height = 1
    msg.width = len(xyzs)
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
    msg.row_step = msg.point_step * msg.width
    msg.data = b"".join(struct.pack("<3f", *xyz) for xyz in xyzs)
    return msg


def make_tile_cloud(stamp_s, frame, reconstruction=False):
    msg = PointCloud2()
    msg.header.stamp.sec = int(stamp_s)
    msg.header.stamp.nanosec = int((stamp_s - int(stamp_s)) * 1e9)
    msg.header.frame_id = frame
    msg.height = 1
    # Two physical targets receive the same three independent elevation views.
    # The first will be contradicted by a clearing ray; the second remains a
    # valid navigation surface and proves the mask is selective.
    optical_targets = [(0.0, 0.0, 5.0)]
    # A connected 3x3 patch around the uncontradicted target supplies enough
    # local geometry for the real surfel plane fit. The contradicted target is
    # deliberately isolated: occupancy still sees it before the clear-space
    # mask, while the clean cloud independently refuses isolated speckle.
    for dy in (-0.2, 0.0, 0.2):
        for dz in (-0.2, 0.0, 0.2):
            optical_targets.append((0.0, 2.0 + dy, 5.0 + dz))
    msg.width = len(optical_targets)
    msg.is_bigendian = False
    msg.is_dense = True
    fields = RECONSTRUCTION_FIELDS if reconstruction else TILE_FIELDS
    for i, name in enumerate(fields):
        f = PointField()
        f.name = name
        f.offset = 4 * i
        f.datatype = PointField.FLOAT32
        f.count = 1
        msg.fields.append(f)
    msg.point_step = 4 * len(fields)
    msg.row_step = msg.point_step * msg.width
    half = math.radians(10.0)
    rows = []
    for x, y, z in optical_targets:
        measured_range = math.sqrt(x * x + y * y + z * z)
        values = [x, y, z, 0.6, measured_range,
                  math.atan2(y, z), measured_range * math.tan(half)]
        if reconstruction:
            values += [0.03, 0.4, 0.08]
        rows.append(struct.pack("<" + "f" * len(values), *values))
    msg.data = b"".join(rows)
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
        self.static_tfb = StaticTransformBroadcaster(self)
        self.hits_pub = self.create_publisher(PointCloud2, "/test/hits", reliable)
        self.clear_pub = self.create_publisher(PointCloud2, "/test/clear", reliable)
        self.survey_pub = self.create_publisher(PointCloud2, "/test/survey", reliable)
        self.tile_pub = self.create_publisher(PointCloud2, "/test/tile", reliable)
        self.reconstruction_pub = self.create_publisher(
            PointCloud2, "/test/reconstruction", reliable)
        # the assembler's trajectory subscription is transient_local
        self.traj_pub = self.create_publisher(PointCloud2, "/test/traj", latched)
        self.snapshots = []
        self.supported_survey_snapshots = []
        self.tile_snapshots = []
        self.surface_snapshots = []
        self.navigation_snapshots = []
        self.navigation_stamps = []
        self.navigation_surfel_snapshots = []
        self.navigation_surfel_stamps = []
        self.occupancy_snapshots = []
        self.latched_qos = latched
        self.create_subscription(
            PointCloud2, "/assembler/survey_pointcloud",
            lambda m: self.snapshots.append(parse_survey(m)), latched)
        self.create_subscription(
            PointCloud2, "/assembler/supported_survey_pointcloud",
            lambda m: self.supported_survey_snapshots.append(
                parse_survey(m)), latched)
        self.tile_sub = self.create_subscription(
            PointCloud2, "/assembler/tile_pointcloud",
            lambda m: self.tile_snapshots.append(parse_survey(m)), latched)
        self.create_subscription(
            PointCloud2, "/assembler/surface_pointcloud",
            lambda m: self.surface_snapshots.append(parse_survey(m)), latched)
        self.create_subscription(
            PointCloud2, "/assembler/navigation_pointcloud",
            self.capture_navigation, latched)
        self.create_subscription(
            PointCloud2, "/assembler/navigation_surfel_pointcloud",
            self.capture_navigation_surfels, latched)
        self.create_subscription(
            OccupancyGrid, "/assembler/vdb_map_occupancy",
            lambda m: self.occupancy_snapshots.append(m), latched)
        self.navigation_export = self.create_client(
            Trigger, "/assembler/export_navigation_surface")
        self.navigation_surfel_export = self.create_client(
            Trigger, "/assembler/export_navigation_surfels")

        transforms = []
        for angle in TILE_ANGLES:
            t = TransformStamped()
            t.header.frame_id = "base_link"
            t.child_frame_id = f"tile_{int(angle):+d}"
            # Camera optical +Z points horizontally at angle elevation;
            # optical +X remains the independent elevation-aperture axis.
            theta = math.pi / 2.0 - math.radians(angle)
            t.transform.rotation.y = math.sin(theta / 2.0)
            t.transform.rotation.w = math.cos(theta / 2.0)
            transforms.append(t)
        self.static_tfb.sendTransform(transforms)

    def capture_navigation(self, msg):
        self.navigation_snapshots.append(parse_survey(msg))
        self.navigation_stamps.append(
            (msg.header.stamp.sec, msg.header.stamp.nanosec))

    def capture_navigation_surfels(self, msg):
        self.navigation_surfel_snapshots.append(parse_survey(msg))
        self.navigation_surfel_stamps.append(
            (msg.header.stamp.sec, msg.header.stamp.nanosec))

    def broadcast_odom(self):
        # exact-stamp odom -> base_link samples bracketing both stamps
        for stamp, x in ((T_E - 0.4, ODOM_X_E),
                         (T_E, ODOM_X_E), (T_E + 0.1, ODOM_X_E),
                         (T_E + 0.2, ODOM_X_E),
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


def occupancy_at(msg, x, y):
    col = math.floor((x - msg.info.origin.position.x) / msg.info.resolution)
    row = math.floor((y - msg.info.origin.position.y) / msg.info.resolution)
    if col < 0 or row < 0 or col >= msg.info.width or row >= msg.info.height:
        return None
    return msg.data[row * msg.info.width + col]


def read_navigation_pcd(path):
    """Read the all-float binary PCD emitted by SurfaceExportPoint."""
    fields = []
    point_count = 0
    with open(path, "rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PCD header ended before DATA")
            text = line.decode("ascii").strip()
            if text.startswith("FIELDS "):
                fields = text.split()[1:]
            elif text.startswith("POINTS "):
                point_count = int(text.split()[1])
            elif text == "DATA binary":
                payload = stream.read()
                break
    if not fields or len(payload) < point_count * 4 * len(fields):
        raise ValueError("malformed navigation PCD")
    rows = []
    for index in range(point_count):
        values = struct.unpack_from(
            "<" + "f" * len(fields), payload, index * 4 * len(fields))
        rows.append(dict(zip(fields, values)))
    return fields, rows


def main():
    rclpy.init()
    node = Harness()

    # let discovery settle, then keep TF fresh while feeding
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
        node.broadcast_odom()

    node.survey_pub.publish(make_survey_cloud(T_E))
    for index, angle in enumerate(TILE_ANGLES):
        stamp = T_E + 0.1 * index
        frame = f"tile_{int(angle):+d}"
        node.tile_pub.publish(make_tile_cloud(stamp, frame))
        node.reconstruction_pub.publish(
            make_tile_cloud(stamp, frame, reconstruction=True))
    # A three-cell raw-hit line survives the legacy occupancy projection's
    # isolated-cell filter. confirmed_surface mode must nevertheless remove it
    # from the persistent planning map and mark the reconstructed target.
    node.hits_pub.publish(make_xyz_cloud(
        T_KF, [(2.9, 0.0, 0.0), (3.0, 0.0, 0.0), (3.1, 0.0, 0.0)]))
    # At the keyframe pose this ray runs from map x=2 to x=7.5 and therefore
    # marks the reconstructed (6,0,0) hypothesis free. It does not touch the
    # equally strong (6,2,0) target.
    node.clear_pub.publish(make_xyz_cloud(T_KF, [(5.5, 0.0, 0.0)]))
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.1)
        node.broadcast_odom()
    node.traj_pub.publish(make_traj(MAP_KF))

    def keep_alive():
        node.broadcast_odom()
        return bool(node.snapshots) and bool(node.supported_survey_snapshots)

    if not spin_until(node, keep_alive, 30.0):
        print("FAIL: no survey snapshot within 30 s", flush=True)
        return 1
    rows = node.snapshots[-1]
    r = next((p for p in rows
              if close(p.get("x", math.nan), EXPECT_XYZ[0], 0.03)
              and close(p.get("y", math.nan), EXPECT_XYZ[1], 0.03)
              and close(p.get("z", math.nan), EXPECT_XYZ[2], 0.03)), None)
    if r is None:
        print(f"FAIL: expected anchored survey row near {EXPECT_XYZ}; "
              f"got {len(rows)} rows", flush=True)
        return 1
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
    supported_rows = node.supported_survey_snapshots[-1]
    if not any(close(p.get("x", math.nan), EXPECT_XYZ[0], 0.03)
               and close(p.get("y", math.nan), EXPECT_XYZ[1], 0.03)
               for p in supported_rows):
        print(f"FAIL: supported survey mismatch: {supported_rows}", flush=True)
        return 1
    print("[1] survey metadata + odom-delta anchoring OK", flush=True)

    if not spin_until(
            node,
            lambda: bool(node.tile_snapshots) and bool(node.surface_snapshots)
            and bool(node.navigation_snapshots)
            and bool(node.navigation_surfel_snapshots),
            10.0):
        print("FAIL: reconstruction products were not published", flush=True)
        return 1
    tiles = node.tile_snapshots[-1]
    if len(tiles) < 3 or not all("intensity" in p and "support" in p
                                 for p in tiles):
        print(f"FAIL: malformed tile mosaic: {tiles}", flush=True)
        return 1
    surfaces = node.surface_snapshots[-1]
    target = next((p for p in surfaces
                   if close(p["x"], 6.0, 0.25)
                   and close(p["y"], 0.0, 0.25)
                   and close(p["z"], 0.0, 0.25)), None)
    if target is None:
        print(f"FAIL: no multi-view intersection near (6,0,0): {surfaces}",
              flush=True)
        return 1
    if target.get("support", 0.0) < 3.0 or \
            target.get("view_span_deg", 0.0) < 19.0 or \
            not 0.5 < target.get("confidence", 0.0) < 1.0:
        print(f"FAIL: weak/non-diverse surface evidence: {target}", flush=True)
        return 1
    contradicted_navigation_target = next(
        (p for p in node.navigation_snapshots[-1]
         if close(p["x"], 6.0, 0.25)
         and close(p["y"], 0.0, 0.25)
         and close(p["z"], 0.0, 0.25)), None)
    if contradicted_navigation_target is not None:
        print("FAIL: ribbon-only hypothesis leaked into dense survey-derived "
              f"navigation returns: {contradicted_navigation_target}",
              flush=True)
        return 1
    isolated_navigation_target = next(
        (p for p in node.navigation_snapshots[-1]
         if close(p["x"], ISOLATED_SURVEY_XYZ[0], 0.03)
         and close(p["y"], ISOLATED_SURVEY_XYZ[1], 0.03)
         and close(p["z"], ISOLATED_SURVEY_XYZ[2], 0.03)), None)
    if isolated_navigation_target is None:
        print("FAIL: dense navigation dropped a supported ranged return just "
              "because it had no planar neighborhood", flush=True)
        return 1
    for field, expected in (
            ("range", META["range"]), ("incidence", META["incidence"]),
            ("texture", META["texture"]),
            ("elevation_lo_offset", META["elevation_lo_offset"]),
            ("elevation_hi_offset", META["elevation_hi_offset"]),
            ("elevation_resolved_fraction", META["elevation_resolved"])):
        if not close(isolated_navigation_target.get(field, math.nan),
                     expected, 0.03):
            print(f"FAIL: dense navigation lost survey field {field}: "
                  f"{isolated_navigation_target}", flush=True)
            return 1
    if isolated_navigation_target.get("normal_valid") != 0.0 or any(
            isolated_navigation_target.get(field) != 0.0
            for field in ("normal_x", "normal_y", "normal_z")) or \
            isolated_navigation_target.get("curvature") != -1.0 or \
            isolated_navigation_target.get("residual") != -1.0:
        print("FAIL: dense navigation did not explicitly mark an unavailable "
              f"normal: {isolated_navigation_target}", flush=True)
        return 1
    navigation_target = next((p for p in node.navigation_snapshots[-1]
                              if close(p["x"], NAV_SURVEY_XYZ[0], 0.15)
                              and close(p["y"], NAV_SURVEY_XYZ[1], 0.15)
                              and close(p["z"], NAV_SURVEY_XYZ[2], 0.15)), None)
    if navigation_target is None or navigation_target["confidence"] < 0.45:
        print(f"FAIL: clean navigation surface omitted the uncontradicted target: "
              f"{node.navigation_snapshots[-1]}", flush=True)
        return 1
    for field in ("normal_x", "normal_y", "normal_z", "curvature",
                  "residual", "range", "incidence", "normal_valid"):
        if field not in navigation_target or not math.isfinite(
                navigation_target[field]):
            print(f"FAIL: dense navigation field {field} is missing/invalid: "
                  f"{navigation_target}", flush=True)
            return 1
    if navigation_target["normal_valid"] != 1.0:
        print("FAIL: dense wall return did not receive its available normal: "
              f"{navigation_target}", flush=True)
        return 1
    strict_target = next(
        (p for p in node.navigation_surfel_snapshots[-1]
         if close(p["x"], NAV_SURVEY_XYZ[0], 0.15)
         and close(p["y"], NAV_SURVEY_XYZ[1], 0.15)
         and close(p["z"], NAV_SURVEY_XYZ[2], 0.15)), None)
    if strict_target is None:
        print("FAIL: strict surfel product omitted the fitted wall: "
              f"{node.navigation_surfel_snapshots[-1]}", flush=True)
        return 1
    if any(close(p["x"], ISOLATED_SURVEY_XYZ[0], 0.03)
           and close(p["y"], ISOLATED_SURVEY_XYZ[1], 0.03)
           and close(p["z"], ISOLATED_SURVEY_XYZ[2], 0.03)
           for p in node.navigation_surfel_snapshots[-1]):
        print("FAIL: isolated return leaked into strict surfel product",
              flush=True)
        return 1
    if node.navigation_stamps[-1] != (0, 0):
        print("FAIL: retained global navigation surface is time-bound to an "
              f"expiring TF sample: {node.navigation_stamps[-1]}", flush=True)
        return 1
    if node.navigation_surfel_stamps[-1] != (0, 0):
        print("FAIL: retained strict surfels are time-bound to an expiring TF "
              f"sample: {node.navigation_surfel_stamps[-1]}", flush=True)
        return 1
    if not spin_until(node, lambda: bool(node.occupancy_snapshots), 10.0):
        print("FAIL: no global occupancy snapshot", flush=True)
        return 1
    occupancy = node.occupancy_snapshots[-1]
    if occupancy_at(occupancy, 6.0, 0.0) == 100:
        print("FAIL: observed-free surface hypothesis overwrote global "
              "occupancy at (6,0)", flush=True)
        return 1
    if occupancy_at(occupancy, 6.0, 2.0) != 100:
        print("FAIL: uncontradicted multi-view surface did not mark global "
              "occupancy at (6,2)", flush=True)
        return 1
    if occupancy_at(occupancy, 5.0, 0.0) == 100:
        print("FAIL: provisional raw hit remained lethal in "
              "confirmed_surface mode", flush=True)
        return 1

    if not node.navigation_export.wait_for_service(timeout_sec=5.0):
        print("FAIL: navigation export service unavailable", flush=True)
        return 1
    future = node.navigation_export.call_async(Trigger.Request())
    if not spin_until(node, future.done, 10.0) or not future.result().success:
        result = future.result() if future.done() else None
        print(f"FAIL: navigation export failed: {result}", flush=True)
        return 1
    try:
        fields, exported = read_navigation_pcd(NAV_EXPORT)
    except (OSError, ValueError) as exc:
        print(f"FAIL: cannot read navigation export: {exc}", flush=True)
        return 1
    expected_fields = [
        "x", "y", "z", "intensity", "range", "incidence", "support",
        "confidence", "pose_sigma", "texture", "texture_variance",
        "elevation_lo_offset", "elevation_hi_offset",
        "elevation_resolved_fraction", "normal_x", "normal_y", "normal_z",
        "curvature", "residual", "normal_valid"]
    if fields != expected_fields:
        print(f"FAIL: navigation PCD schema is wrong: {fields}", flush=True)
        return 1
    if any(close(p["x"], 6.0, 0.25) and close(p["y"], 0.0, 0.25)
           and close(p["z"], 0.0, 0.25) for p in exported):
        print("FAIL: navigation export retained an observed-free hypothesis",
              flush=True)
        return 1
    if not any(close(p["x"], NAV_SURVEY_XYZ[0], 0.15)
               and close(p["y"], NAV_SURVEY_XYZ[1], 0.15)
               and close(p["z"], NAV_SURVEY_XYZ[2], 0.15)
               for p in exported):
        print("FAIL: navigation export omitted the supported-survey wall",
              flush=True)
        return 1
    exported_isolated = next(
        (p for p in exported
         if close(p["x"], ISOLATED_SURVEY_XYZ[0], 0.03)
         and close(p["y"], ISOLATED_SURVEY_XYZ[1], 0.03)
         and close(p["z"], ISOLATED_SURVEY_XYZ[2], 0.03)), None)
    if exported_isolated is None or exported_isolated["normal_valid"] != 0.0:
        print("FAIL: dense export omitted or mislabelled isolated evidence: "
              f"{exported_isolated}", flush=True)
        return 1

    if not node.navigation_surfel_export.wait_for_service(timeout_sec=5.0):
        print("FAIL: navigation surfel export service unavailable", flush=True)
        return 1
    future = node.navigation_surfel_export.call_async(Trigger.Request())
    if not spin_until(node, future.done, 10.0) or not future.result().success:
        result = future.result() if future.done() else None
        print(f"FAIL: navigation surfel export failed: {result}", flush=True)
        return 1
    try:
        surfel_fields, exported_surfels = read_navigation_pcd(SURFEL_EXPORT)
    except (OSError, ValueError) as exc:
        print(f"FAIL: cannot read navigation surfel export: {exc}", flush=True)
        return 1
    expected_surfel_fields = [
        "x", "y", "z", "intensity", "support", "view_span_deg",
        "confidence", "normal_x", "normal_y", "normal_z", "curvature",
        "residual", "range_sigma", "echo_width", "echo_prominence",
        "peak_prominence"]
    if surfel_fields != expected_surfel_fields:
        print(f"FAIL: surfel PCD schema is wrong: {surfel_fields}", flush=True)
        return 1
    if any(close(p["x"], ISOLATED_SURVEY_XYZ[0], 0.03)
           and close(p["y"], ISOLATED_SURVEY_XYZ[1], 0.03)
           and close(p["z"], ISOLATED_SURVEY_XYZ[2], 0.03)
           for p in exported_surfels):
        print("FAIL: strict surfel export retained isolated evidence",
              flush=True)
        return 1
    if not any(close(p["x"], NAV_SURVEY_XYZ[0], 0.15)
               and close(p["y"], NAV_SURVEY_XYZ[1], 0.15)
               and close(p["z"], NAV_SURVEY_XYZ[2], 0.15)
               for p in exported_surfels):
        print("FAIL: strict surfel export omitted the fitted wall", flush=True)
        return 1
    # Prove the later file is from the shutdown path, not this service call.
    for path in (NAV_EXPORT, SURFEL_EXPORT):
        try:
            os.unlink(path)
        except OSError as exc:
            print("FAIL: cannot clear service export before shutdown test: "
                  f"{exc}", flush=True)
            return 1

    # The full tile cache is intentionally released when nobody views it, but
    # its raw keyframe evidence must remain recoverable. Disconnect, give the
    # assembler time to release/replace its durable sample, then reconnect and
    # require the exact product to be rebuilt from spill.
    node.destroy_subscription(node.tile_sub)
    node.tile_sub = None
    node.tile_snapshots.clear()
    disconnect_deadline = time.monotonic() + 1.5
    while time.monotonic() < disconnect_deadline:
        node.broadcast_odom()
        rclpy.spin_once(node, timeout_sec=0.1)
    node.tile_sub = node.create_subscription(
        PointCloud2, "/assembler/tile_pointcloud",
        lambda m: node.tile_snapshots.append(parse_survey(m)),
        node.latched_qos)
    if not spin_until(
            node, lambda: any(len(snap) >= 3 for snap in node.tile_snapshots),
            10.0):
        print("FAIL: tile product was not rebuilt after viewer reconnected",
              flush=True)
        return 1
    print("[2] tile retained on disk + diagnostic/live/export surfaces OK",
          flush=True)

    # z-only correction must re-render the product at the new depth
    n_before = len(node.snapshots)
    n_surface_before = len(node.surface_snapshots)
    n_navigation_before = len(node.navigation_snapshots)
    n_surfel_before = len(node.navigation_surfel_snapshots)
    node.traj_pub.publish(make_traj((MAP_KF[0], MAP_KF[1],
                                     MAP_KF[2] + Z_CORRECTION)))

    def z_moved():
        node.broadcast_odom()
        for snap in node.snapshots[n_before:]:
            if any(close(p["x"], EXPECT_XYZ[0], 0.03)
                   and close(p["y"], EXPECT_XYZ[1], 0.03)
                   and close(p["z"], EXPECT_XYZ[2] + Z_CORRECTION, 0.03)
                   for p in snap):
                return True
        return False

    if not spin_until(node, z_moved, 30.0):
        zs = [p["z"] for s in node.snapshots[n_before:] for p in s
              if close(p["x"], EXPECT_XYZ[0], 0.03)
              and close(p["y"], EXPECT_XYZ[1], 0.03)]
        print(f"FAIL: no re-rendered snapshot at z ~ {Z_CORRECTION}; "
              f"saw z = {zs} (a stale z means the pose gate ignored depth)",
              flush=True)
        return 1
    def surface_z_moved():
        node.broadcast_odom()
        for snap in node.surface_snapshots[n_surface_before:]:
            if any(close(p["x"], 6.0, 0.25) and
                   close(p["z"], Z_CORRECTION, 0.25) for p in snap):
                return True
        return False

    if not spin_until(node, surface_z_moved, 10.0):
        print("FAIL: multi-view surface did not follow graph z correction",
              flush=True)
        return 1
    def navigation_z_moved():
        node.broadcast_odom()
        for snap in node.navigation_snapshots[n_navigation_before:]:
            if any(close(p["x"], NAV_SURVEY_XYZ[0], 0.15) and
                   close(p["y"], NAV_SURVEY_XYZ[1], 0.15) and
                   close(p["z"], Z_CORRECTION, 0.25) for p in snap):
                return True
        return False

    if not spin_until(node, navigation_z_moved, 10.0):
        print("FAIL: navigation surface did not follow graph z correction",
              flush=True)
        return 1
    def surfel_z_moved():
        node.broadcast_odom()
        for snap in node.navigation_surfel_snapshots[n_surfel_before:]:
            if any(close(p["x"], NAV_SURVEY_XYZ[0], 0.15) and
                   close(p["y"], NAV_SURVEY_XYZ[1], 0.15) and
                   close(p["z"], Z_CORRECTION, 0.25) for p in snap):
                return True
        return False

    if not spin_until(node, surfel_z_moved, 10.0):
        print("FAIL: strict surfels did not follow graph z correction",
              flush=True)
        return 1
    print("[3] z-only correction re-rendered every product OK", flush=True)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
