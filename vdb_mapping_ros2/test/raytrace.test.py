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
#
# launch_testing harness for the raytrace / get_map_section service test.
# Starts a vdb_mapping_ros_node, runs test/raytrace_test.py as a standalone
# process and asserts it exits 0. raytrace_test.py imports make_cloud from
# smoke_test, so the test directory is put on PYTHONPATH.
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

    mapping_node = launch_ros.actions.Node(
        package='vdb_mapping_ros2',
        executable='vdb_mapping_ros_node',
        name='vdb_mapping_ros2',
        output='screen',
        parameters=[{
            'map_frame': 'map',
            'robot_frame': 'base_link',
            'sources': ['velodyne'],
            'velodyne.topic': '/cloud',
            'velodyne.sensor_origin_frame': 'velodyne',
            'velodyne.reliable': True,
            'accumulate_updates': False,
            'publish_pointcloud': True,
            'publish_vis_marker': True,
            'publish_occupancy_grid': True,
            'visualization_rate': 5.0,
        }],
    )

    raytrace = launch.actions.ExecuteProcess(
        cmd=[sys.executable, os.path.join(test_dir, 'raytrace_test.py')],
        name='raytrace_test',
        output='screen',
        additional_env={
            'PYTHONUNBUFFERED': '1',
            'PYTHONPATH': test_dir + os.pathsep + os.environ.get('PYTHONPATH', ''),
        },
    )

    return (
        launch.LaunchDescription([
            mapping_node,
            raytrace,
            launch_testing.actions.ReadyToTest(),
        ]),
        {'raytrace': raytrace},
    )


class RaytraceTestRuns(unittest.TestCase):
    def test_raytrace_completes(self, proc_info, raytrace):
        proc_info.assertWaitForShutdown(process=raytrace, timeout=120)


@launch_testing.post_shutdown_test()
class RaytraceTestExit(unittest.TestCase):
    def test_exit_code(self, proc_info, raytrace):
        assertExitCodes(proc_info, allowable_exit_codes=[0], process=raytrace)
