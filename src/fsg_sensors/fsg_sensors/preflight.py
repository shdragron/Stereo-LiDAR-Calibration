"""FSG pre-flight sensor check: one command, PASS/FAIL verdict.

Checks (thresholds tuned for this rig: RS-16 10 Hz + ZED gen1 30 Hz on AGX Orin):
  1. power  : nvpmodel is MAXN, CPU clocks locked (min == max)
  2. lidar  : rate ~10 Hz, pipeline latency (now - stamp) small
  3. camera : rate >= ~28 Hz, pipeline latency stable
  4. pairing: ApproximateTime lidar<->camera dt consistent (low jitter)
  5. tf     : extrinsic TF camera_optical -> rslidar exists (race mode)

Usage:
  ros2 run fsg_sensors preflight                # 10 s measurement
  ros2 run fsg_sensors preflight --duration 20
  ros2 run fsg_sensors preflight --no-tf        # during calibration (no TF yet)
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import PointCloud2, CompressedImage

LIDAR_TOPIC = '/sensors/lidar/points'
# compressed: the only camera stream public in race mode (see zed_relay)
CAM_TOPIC = '/sensors/camera/left/compressed'
CAM_FRAME = 'zed_left_camera_frame_optical'
LIDAR_FRAME = 'rslidar'

QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST, depth=10)

_results = []


def check(name, ok, detail):
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<28} {detail}")
    return ok


def warn(name, detail):
    print(f"  [WARN] {name:<28} {detail}")


def check_power():
    print("\n== power / clocks ==")
    try:
        out = subprocess.run(['nvpmodel', '-q'], capture_output=True,
                             text=True, timeout=5).stdout
        mode = next((ln.split(':')[1].strip() for ln in out.splitlines()
                     if 'NV Power Mode' in ln), '?')
        check('nvpmodel', mode == 'MAXN', f'mode={mode} (want MAXN)')
    except Exception as e:
        warn('nvpmodel', f'query failed ({e}) — check manually')
    try:
        base = '/sys/devices/system/cpu/cpu0/cpufreq/'
        lo = int(open(base + 'scaling_min_freq').read())
        hi = int(open(base + 'scaling_max_freq').read())
        check('cpu clock lock', lo == hi,
              f'min={lo/1000:.0f} max={hi/1000:.0f} MHz '
              + ('(locked)' if lo == hi else '(jetson_clocks not applied)'))
    except Exception as e:
        warn('cpu clock lock', f'sysfs read failed ({e})')


class Probe(Node):
    def __init__(self, lidar_topic, cam_topic):
        super().__init__('fsg_preflight')
        self.lid, self.cam, self.dts = [], [], []
        ls = message_filters.Subscriber(self, PointCloud2, lidar_topic,
                                        qos_profile=QOS)
        cs = message_filters.Subscriber(self, CompressedImage, cam_topic,
                                        qos_profile=QOS)
        ls.registerCallback(self._mk(self.lid))
        cs.registerCallback(self._mk(self.cam))
        self.sync = message_filters.ApproximateTimeSynchronizer([ls, cs],
                                                                queue_size=10,
                                                                slop=0.06)
        self.sync.registerCallback(self._pair)

    def _mk(self, store):
        def cb(msg):
            recv = self.get_clock().now().nanoseconds
            st = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            store.append((recv, st))
        return cb

    def _pair(self, cloud, img):
        tl = cloud.header.stamp.sec + cloud.header.stamp.nanosec * 1e-9
        tc = img.header.stamp.sec + img.header.stamp.nanosec * 1e-9
        self.dts.append((tl - tc) * 1000.0)


def _rate_lat(rows, dur):
    # rate over the stream's own first->last span, not the full measurement
    # window: subscription discovery / lazy-relay start-up would otherwise
    # bias the average down.
    recv = np.array([r[0] for r in rows], np.float64)
    st = np.array([r[1] for r in rows], np.float64)
    span = (recv[-1] - recv[0]) / 1e9
    hz = (len(rows) - 1) / span if span > 0 else 0.0
    return hz, (recv - st) / 1e6                   # Hz, latency ms


def check_streams(node, dur):
    print(f"\n== streams ({dur:.0f}s measurement) ==")
    if len(node.lid) < 3:
        check('lidar stream', False, f'only {len(node.lid)} clouds — topic up?')
    else:
        hz, lat = _rate_lat(node.lid, dur)
        check('lidar rate', 9.0 <= hz <= 11.0, f'{hz:.1f} Hz (want 10±1)')
        check('lidar latency', lat.mean() < 20,
              f'now-stamp {lat.mean():.0f} ms (want <20)')
    if len(node.cam) < 3:
        check('camera stream', False, f'only {len(node.cam)} images — topic up?')
    else:
        hz, lat = _rate_lat(node.cam, dur)
        check('camera rate', hz >= 27.0, f'{hz:.1f} Hz (want >=27)')
        check('camera latency', lat.mean() < 80 and lat.std() < 10,
              f'now-stamp {lat.mean():.0f}±{lat.std():.1f} ms (want <80, stable)')
    if len(node.dts) < 3:
        check('lidar-cam pairing', False, f'only {len(node.dts)} pairs')
    else:
        # dt is dominated by the camera pipeline latency (~35 ms) plus a
        # free-running phase term up to one camera period (33 ms) — both are
        # constant/bounded and absorbed by deskewing downstream. This check
        # only guards regressions: pairing coverage and bounded offset/jitter.
        a = np.array(node.dts)
        n_lid = max(len(node.lid), 1)
        check('lidar-cam pairing',
              a.std() < 20 and abs(a.mean()) < 70 and len(a) >= 0.9 * n_lid,
              f'dt={a.mean():+.1f}±{a.std():.1f} ms, matched '
              f'{len(a)}/{n_lid} clouds (want jitter<20, |dt|<70, >=90%)')


def check_tf(node):
    print("\n== extrinsic TF ==")
    try:
        from tf2_ros import Buffer, TransformListener
        buf = Buffer()
        TransformListener(buf, node, spin_thread=False)
        end = time.time() + 3.0
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            if buf.can_transform(CAM_FRAME, LIDAR_FRAME, rclpy.time.Time()):
                t = buf.lookup_transform(CAM_FRAME, LIDAR_FRAME,
                                         rclpy.time.Time()).transform.translation
                check('extrinsic TF', True,
                      f'{CAM_FRAME} -> {LIDAR_FRAME} '
                      f't=({t.x:+.3f}, {t.y:+.3f}, {t.z:+.3f}) m')
                return
        check('extrinsic TF', False,
              f'{CAM_FRAME} -> {LIDAR_FRAME} not available — '
              f'launch mode:=race with a valid extrinsic yaml')
    except Exception as e:
        check('extrinsic TF', False, f'lookup failed: {e}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--duration', type=float, default=10.0)
    ap.add_argument('--lidar-topic', default=LIDAR_TOPIC)
    ap.add_argument('--cam-topic', default=CAM_TOPIC)
    ap.add_argument('--no-tf', action='store_true',
                    help='skip the extrinsic TF check (calibration mode)')
    args = ap.parse_args(argv)

    print("FSG sensor pre-flight check")
    print("=" * 60)
    check_power()

    rclpy.init()
    node = Probe(args.lidar_topic, args.cam_topic)
    t0 = time.time()
    while time.time() - t0 < args.duration:
        rclpy.spin_once(node, timeout_sec=0.05)
    check_streams(node, args.duration)
    if not args.no_tf:
        check_tf(node)
    node.destroy_node()
    rclpy.try_shutdown()

    ok = all(_results)
    print("\n" + "=" * 60)
    print(f"VERDICT: {'PASS — ready to run' if ok else 'FAIL — fix the items above'}"
          f"  ({sum(_results)}/{len(_results)} checks)")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
