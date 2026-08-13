#!/usr/bin/env python3
# launch_testing harness for the end-to-end map assembler test. Starts a
# vdb_map_assembler_node wired to the /test/* synthetic topics, runs
# test/assembler_test.py as a standalone process and asserts it exits 0.
# The assembler keeps the node name "assembler" so its ~/ outputs resolve
# under /assembler/, which is what assembler_test.py subscribes to.
import os
import sys
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
from launch_testing.asserts import assertExitCodes


def generate_test_description():
    test_dir = os.path.dirname(__file__)

    assembler_node = launch_ros.actions.Node(
        package='vdb_mapping_ros2',
        executable='vdb_map_assembler_node',
        name='assembler',
        output='screen',
        parameters=[{
            'hits_topic': '/test/hits',
            'clear_topic': '/test/clear',
            'survey_topic': '/test/survey',
            'tile_topic': '/test/tile',
            'traj_topic': '/test/traj',
            'robot_frame': 'base_link',
            'odom_frame': 'odom',
            'map_frame': 'map',
            # one synthetic point per cloud: mask/threshold friction off
            'survey_occupancy_mask': False,
            'tile_resolution': 0.1,
            'surface_resolution': 0.2,
            'surface_min_observations': 3,
            'surface_min_view_span_deg': 6.0,
            'surface_max_samples_per_return': 5,
            'prob_thres_min': 0.49,
            'prob_thres_max': 0.51,
            # exact-stamp TF only: a fallback to latest would silently
            # collapse the odom deltas this test pins
            'allow_latest_tf_fallback': False,
            'render_min_period': 0.2,
            'input_reliable': True,
            'export_on_shutdown': False,
            # spill evidence to disk so the z-correction re-render in stage 2
            # exercises the readSpill round-trip, not just resident clouds
            'spill_dir': '/tmp/assembler_test_spill',
        }],
    )

    harness = launch.actions.ExecuteProcess(
        cmd=[sys.executable, os.path.join(test_dir, 'assembler_test.py')],
        output='screen',
    )

    return launch.LaunchDescription([
        assembler_node,
        harness,
        launch_testing.actions.ReadyToTest(),
    ]), {'harness': harness, 'assembler_node': assembler_node}


class TestAssembler(unittest.TestCase):
    def test_harness_exits_clean(self, proc_info, harness):
        proc_info.assertWaitForShutdown(process=harness, timeout=120)


@launch_testing.post_shutdown_test()
class TestAfterShutdown(unittest.TestCase):
    def test_exit_code(self, proc_info, harness):
        assertExitCodes(proc_info, allowable_exit_codes=[0], process=harness)
