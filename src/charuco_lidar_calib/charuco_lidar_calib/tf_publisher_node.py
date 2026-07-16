"""Publish a solved extrinsic as a static TF (camera_optical -> lidar).

  ros2 run charuco_lidar_calib tf_publisher \
       --ros-args -p extrinsic:=calib_debug/extrinsic_zed_rslidar.yaml
"""
import sys

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

from . import solve


class ExtrinsicTfPublisher(Node):
    def __init__(self):
        super().__init__('charuco_lidar_tf_publisher')
        self.declare_parameter('extrinsic', 'calib_debug/extrinsic_zed_rslidar.yaml')
        path = self.get_parameter('extrinsic').get_parameter_value().string_value
        with open(path) as f:
            e = yaml.safe_load(f)
        l2c = e['lidar_to_camera']
        R = np.array(l2c['R'], float)
        t = np.array(l2c['t'], float)
        parent = l2c.get('parent_frame', 'zed_left_camera_frame_optical')
        child = l2c.get('child_frame', 'rslidar')
        q = solve.matrix_to_quaternion(R)

        self._br = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])
        tf.transform.rotation.x = float(q[0])
        tf.transform.rotation.y = float(q[1])
        tf.transform.rotation.z = float(q[2])
        tf.transform.rotation.w = float(q[3])
        self._br.sendTransform(tf)
        self.get_logger().info(
            f"static TF {parent} -> {child}: t={np.round(t,4).tolist()} "
            f"q={np.round(q,4).tolist()}  (from {path})")


def main(argv=None):
    rclpy.init(args=argv)
    node = ExtrinsicTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
