"""Publish the solved ZED2i<->RS-16 extrinsic as a static TF."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_yaml = os.path.join(os.getcwd(), 'calib_debug',
                                'extrinsic_zed_rslidar.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('extrinsic', default_value=default_yaml,
                              description='path to the solved extrinsic yaml'),
        Node(
            package='charuco_lidar_calib',
            executable='tf_publisher',
            name='charuco_lidar_tf_publisher',
            output='screen',
            parameters=[{'extrinsic': LaunchConfiguration('extrinsic')}],
        ),
    ])
