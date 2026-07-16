"""FSG sensor bringup: RS-16 + ZED under the /sensors namespace (+ extrinsic TF).

Public topics (IDENTICAL in race and calib mode):
  /sensors/lidar/points                  PointCloud2 (frame: rslidar)
  /sensors/camera/left/compressed        CompressedImage (JPEG, 30 Hz, lazy)
  /sensors/camera/left/info              CameraInfo
  /sensors/camera/right/compressed       CompressedImage
  /sensors/camera/right/info             CameraInfo

mode:=race additionally publishes the extrinsic TF; mode:=calib does not.
Raw images live on the hidden wrapper topics (/_zed_hidden/zed/...) which the
capture/calibration tools subscribe directly.

Usage:
  ros2 launch fsg_sensors sensors.launch.py                  # race: sensors + extrinsic TF
  ros2 launch fsg_sensors sensors.launch.py mode:=calib      # calibration capture: no TF
  ros2 launch fsg_sensors sensors.launch.py extrinsic:=/path/to/extrinsic.yaml

The default extrinsic is the file `charuco_lidar_calib` writes, so the loop is:
calibrate -> relaunch -> TF is live. Archive good results into
src/fsg_sensors/config/extrinsics/history/<date>_<mount>.yaml.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

DEFAULT_EXTRINSIC = os.path.join(os.getcwd(), 'calib_debug', 'extrinsic_zed_rslidar.yaml')


def _setup(context):
    share = get_package_share_directory('fsg_sensors')
    mode = context.launch_configurations['mode']
    extrinsic = context.launch_configurations['extrinsic']
    camera_model = context.launch_configurations['camera_model']
    actions = []

    # --- RS-16: vendor node, team config (topics under /sensors/lidar/*) ---
    actions.append(Node(
        package='rslidar_sdk', executable='rslidar_sdk_node',
        name='lidar_driver', namespace='sensors', output='screen',
        parameters=[{'config_path': os.path.join(share, 'config', 'rslidar.yaml')}]))

    # --- ZED: vendor launch, unmodified, ALWAYS under the hidden namespace ---
    # (leading '_' = not shown by `ros2 topic list`). The lazy relay exposes
    # the same short public topics in BOTH modes:
    #   /sensors/camera/{left,right}/{compressed,info}
    # Calibration tools (sync_capture, latency probes) subscribe the hidden
    # raw topics directly — hidden only means unlisted, not unreachable.
    zed_launch = os.path.join(get_package_share_directory('zed_wrapper'),
                              'launch', 'zed_camera.launch.py')
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_launch),
        launch_arguments={
            'camera_model': camera_model,
            'namespace': '_zed_hidden',
            'enable_ipc': 'false',
            'ros_params_override_path': os.path.join(share, 'config',
                                                     'zed_override.yaml'),
        }.items()))
    actions.append(Node(
        package='fsg_sensors', executable='zed_relay',
        name='zed_relay', namespace='sensors', output='screen'))

    # --- extrinsic TF (race mode only) ---
    if mode == 'race':
        if os.path.exists(extrinsic):
            actions.append(Node(
                package='charuco_lidar_calib', executable='tf_publisher',
                name='extrinsic_tf', namespace='sensors', output='screen',
                parameters=[{'extrinsic': extrinsic}]))
        else:
            actions.append(LogInfo(msg=(
                f'[fsg_sensors] extrinsic not found: {extrinsic} — TF NOT '
                f'published. Run the calibration first, or pass extrinsic:=<yaml>.')))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='race',
                              choices=['race', 'calib'],
                              description='race: publish extrinsic TF; calib: sensors only'),
        DeclareLaunchArgument('extrinsic', default_value=DEFAULT_EXTRINSIC,
                              description='extrinsic yaml from charuco_lidar_calib'),
        DeclareLaunchArgument('camera_model', default_value='zed'),
        OpaqueFunction(function=_setup),
    ])
