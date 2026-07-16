#!/usr/bin/env python3
"""
LiDAR + ZED 지연/타임스탬프 진단 프로브.
각 토픽에 대해:
  - 지연(now - header.stamp) : 파이프라인 지연 (수신시점 - 헤더에 찍힌 시각)
  - 캡처간격(stamp diff)      : 헤더 시각의 간격 = 실제 캡처/조립 주기(규칙적인가)
  - 도착간격(recv diff)       : 우리 노드 수신 간격 = 전달이 밀리는가(버스트/파일업)
사용:  python3 latency_probe.py [측정초=12]
"""
import sys, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image
import numpy as np

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
TOPICS = [('cam',   Image,       '/_zed_hidden/zed/rgb/color/rect/image'),
          ('lidar', PointCloud2, '/sensors/lidar/points')]


class Probe(Node):
    def __init__(self):
        super().__init__('latency_probe')
        self.rows = {n: [] for n, _, _ in TOPICS}
        for n, typ, tp in TOPICS:
            self.create_subscription(typ, tp, self._mk(n), qos_profile_sensor_data)

    def _mk(self, name):
        def cb(msg):
            recv = self.get_clock().now().nanoseconds
            st = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            self.rows[name].append((recv, st))
        return cb


def rep(name, rows):
    if len(rows) < 3:
        print(f"\n[{name}]  수신 부족 (n={len(rows)}) — 토픽/QoS 확인")
        return
    recv = np.array([r[0] for r in rows], np.float64)
    st   = np.array([r[1] for r in rows], np.float64)
    lat  = (recv - st) / 1e6
    dtr  = np.diff(recv) / 1e6
    dts  = np.diff(st) / 1e6
    dur  = (recv[-1] - recv[0]) / 1e9
    print(f"\n[{name}]  n={len(rows)}  rate={len(rows)/max(dur,1e-6):.1f} Hz  (측정 {dur:.1f}s)")
    print(f"  지연 now-stamp : mean={lat.mean():6.0f}  min={lat.min():6.0f}  max={lat.max():6.0f}  std={lat.std():5.0f} ms   ← 파이프라인 지연")
    print(f"  캡처간격 stamp : mean={dts.mean():6.1f}  min={dts.min():6.1f}  max={dts.max():6.1f}  std={dts.std():5.1f} ms   ← 발행 규칙성")
    print(f"  도착간격 recv  : mean={dtr.mean():6.1f}  min={dtr.min():6.1f}  max={dtr.max():6.1f}  std={dtr.std():5.1f} ms   ← 밀림/버스트")


def main():
    rclpy.init()
    node = Probe()
    import time
    end = node.get_clock().now().nanoseconds + DUR * 1e9
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.02)
    print("=" * 70)
    print(f"지연 진단 결과 (측정 {DUR:.0f}s)")
    print("=" * 70)
    for n, _, _ in TOPICS:
        rep(n, node.rows[n])
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
