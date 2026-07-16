"""Identify the ArUco dictionary of a board from an image (or a live topic).

  # from an image file:
  ros2 run charuco_lidar_calib dict_sniffer --image captures/2026.../0000.png
  # from a live ZED frame:
  ros2 run charuco_lidar_calib dict_sniffer --topic /zed/zed_node/rgb/color/rect/image
"""
import argparse
import sys

import cv2
import cv2.aruco as aruco


DICTS = [n for n in dir(aruco) if n.startswith('DICT_')]


def _params():
    try:
        return aruco.DetectorParameters_create()
    except AttributeError:
        return aruco.DetectorParameters()


def _get_dict(name):
    try:
        return aruco.Dictionary_get(getattr(aruco, name))
    except AttributeError:
        return aruco.getPredefinedDictionary(getattr(aruco, name))


def _detect(gray, dname, params):
    d = _get_dict(dname)
    try:
        corners, ids, _ = aruco.detectMarkers(gray, d, parameters=params)
    except Exception:
        det = aruco.ArucoDetector(d, params)
        corners, ids, _ = det.detectMarkers(gray)
    n = 0 if ids is None else len(ids)
    return n, (sorted(ids.flatten().tolist()) if ids is not None else [])


def sniff(gray):
    params = _params()
    h, w = gray.shape[:2]
    variants = [gray, cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)]
    results = []
    for dn in DICTS:
        best, ids = 0, []
        for v in variants:
            n, i = _detect(v, dn, params)
            if n > best:
                best, ids = n, i
        if best:
            results.append((dn, best, ids))
    results.sort(key=lambda x: -x[1])
    return results


def _grab_topic(topic, timeout=8.0):
    import rclpy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    rclpy.init()
    node = rclpy.create_node('dict_sniffer_grab')
    br = CvBridge()
    got = {}

    def cb(m):
        got['img'] = br.imgmsg_to_cv2(m, desired_encoding='bgr8')
    node.create_subscription(Image, topic, cb, 10)
    import time
    t0 = time.time()
    while rclpy.ok() and 'img' not in got and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node(); rclpy.shutdown()
    return got.get('img')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', default=None)
    ap.add_argument('--topic', default=None)
    args = ap.parse_args(argv)

    if args.image:
        img = cv2.imread(args.image)
    elif args.topic:
        img = _grab_topic(args.topic)
    else:
        print("give --image PATH or --topic TOPIC"); return 1
    if img is None:
        print("could not obtain image"); return 1
    gray = img if img.ndim == 2 else cv2.cvtColor(
        img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR),
        cv2.COLOR_BGR2GRAY)

    res = sniff(gray)
    if not res:
        print("No ArUco markers detected in ANY dictionary."); return 1
    print("dictionary            markers  id-range")
    for dn, n, ids in res[:8]:
        rng = f"{min(ids)}..{max(ids)}" if ids else "-"
        print(f"  {dn:20s} {n:4d}    {rng}")
    # prefer smallest dict of the winning bit-size (matches typical generators)
    top_n = res[0][1]
    winners = [r for r in res if r[1] == top_n]
    winners.sort(key=lambda r: int(r[0].split('_')[-1]))  # by count suffix
    print(f"\nBEST: {winners[0][0]}  ({top_n} markers)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
