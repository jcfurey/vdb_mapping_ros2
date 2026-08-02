#!/usr/bin/env python3
# launch_testing harness for the remote section-sync test: mapper A
# (node name vdb_mapping, publishing sections) feeds remote B (node name
# remote, applying them); test/section_sync_test.py drives A with a wall
# cloud, resets it, and asserts B tracks both transitions.
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

    mapper = launch_ros.actions.Node(
        package='vdb_mapping_ros2',
        executable='vdb_mapping_ros_node',
        name='vdb_mapping',
        output='screen',
        parameters=[{
            'map_frame': 'map',
            'robot_frame': 'base_link',
            'resolution': 0.05,
            'prob_thres_min': 0.49,
            'prob_thres_max': 0.51,
            'sources': ['wall'],
            'wall.topic': '/cloud',
            'wall.sensor_origin_frame': 'sensor',
            'wall.reliable': True,
            'accumulate_updates': False,
            'publish_pointcloud': True,
            'visualization_rate': 5.0,
            'publish_sections': True,
            'section_update.rate': 4.0,
            'section_update.min_coord.x': -5.0,
            'section_update.min_coord.y': -5.0,
            'section_update.min_coord.z': -5.0,
            'section_update.max_coord.x': 5.0,
            'section_update.max_coord.y': 5.0,
            'section_update.max_coord.z': 5.0,
        }],
    )

    remote = launch_ros.actions.Node(
        package='vdb_mapping_ros2',
        executable='vdb_mapping_ros_node',
        name='remote',
        output='screen',
        parameters=[{
            'map_frame': 'map',
            'robot_frame': 'base_link',
            'resolution': 0.05,
            'prob_thres_min': 0.49,
            'prob_thres_max': 0.51,
            'apply_raw_sensor_data': False,
            'publish_pointcloud': True,
            'visualization_rate': 5.0,
            'remote_sources': ['mapper'],
            'mapper.namespace': '/vdb_mapping',
            'mapper.apply_remote_sections': True,
        }],
    )

    harness = launch.actions.ExecuteProcess(
        cmd=[sys.executable, os.path.join(test_dir, 'section_sync_test.py')],
        output='screen',
    )

    return launch.LaunchDescription([
        mapper,
        remote,
        harness,
        launch_testing.actions.ReadyToTest(),
    ]), {'harness': harness}


class TestSectionSync(unittest.TestCase):
    def test_harness_exits_clean(self, proc_info, harness):
        proc_info.assertWaitForShutdown(process=harness, timeout=150)


@launch_testing.post_shutdown_test()
class TestAfterShutdown(unittest.TestCase):
    def test_exit_code(self, proc_info, harness):
        assertExitCodes(proc_info, allowable_exit_codes=[0], process=harness)
