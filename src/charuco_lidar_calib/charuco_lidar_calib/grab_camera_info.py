"""Save a CameraInfo message from a live topic into a yaml (K/D) for calibrate.

  ros2 run charuco_lidar_calib grab_camera_info \
       --topic /zed/zed_node/rgb/color/rect/camera_info --out calib_debug/zed_K.yaml
"""
import argparse
import sys
import time

import yaml
import rclpy
from sensor_msgs.msg import CameraInfo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--topic', default='/sensors/camera/left/info')
    ap.add_argument('--right-topic',
                    default='/sensors/camera/right/info',
                    help='right camera_info; its P[0,3] gives the baseline')
    ap.add_argument('--out', default='calib_debug/zed_K.yaml')
    ap.add_argument('--timeout', type=float, default=8.0)
    args = ap.parse_args(argv)

    rclpy.init(args=argv)
    node = rclpy.create_node('grab_camera_info')
    got = {}

    def cb(m):
        got['msg'] = m

    def cb_r(m):
        got['right'] = m
    node.create_subscription(CameraInfo, args.topic, cb, 10)
    node.create_subscription(CameraInfo, args.right_topic, cb_r, 10)
    t0 = time.time()
    while rclpy.ok() and 'msg' not in got and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    # give the right info a short extra window
    t0 = time.time()
    while rclpy.ok() and 'right' not in got and time.time() - t0 < 2.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    if 'msg' not in got:
        print(f"no CameraInfo on {args.topic}"); return 1
    m = got['msg']
    data = {
        'width': int(m.width), 'height': int(m.height),
        'frame_id': m.header.frame_id,
        'distortion_model': m.distortion_model,
        'K': [float(x) for x in m.k],
        'D': [float(x) for x in m.d],
    }
    if 'right' in got:
        r = got['right']
        # ROS stereo convention: right P[0,3] = -fx * baseline
        fx_r = float(r.p[0])
        baseline = -float(r.p[3]) / fx_r if fx_r else 0.0
        data['baseline'] = float(baseline)
        print(f"  baseline = {baseline*1000:.2f} mm (from {args.right_topic})")
    else:
        print(f"  (no right camera_info on {args.right_topic} — baseline not saved; "
              f"is publish_left_right enabled?)")
    import os
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"wrote {args.out}")
    print(f"  fx={m.k[0]:.3f} fy={m.k[4]:.3f} cx={m.k[2]:.3f} cy={m.k[5]:.3f}")
    print(f"  D={list(m.d)}  frame={m.header.frame_id}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
