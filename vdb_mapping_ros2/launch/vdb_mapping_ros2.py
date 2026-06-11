import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
            get_package_share_directory('vdb_mapping_ros2'),
            'config',
            'vdb_params.yaml'
            )

    container = ComposableNodeContainer(
        name='Container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='vdb_mapping_ros2',
                plugin='vdb_mapping_ros2::VDBMappingROS2',
                name='vdb_mapping',
                parameters=[config],
            )
        ],
        output='screen',
    )

    ld.add_action(container)

    return ld
