"""Paint a camera image onto the LiDAR cloud using the solved extrinsic.

Outputs a colored XYZRGB PCD (open with pcl_viewer / CloudCompare) and,
optionally, rendered novel-view PNGs.

  ros2 run charuco_lidar_calib colorize captures/<ts>/0000_L.png \
       --extrinsic calib_debug/extrinsic_zed_rslidar.yaml
"""
import argparse
import os
import sys

import numpy as np
import cv2
import yaml

from . import pcd_io
from .calibrate import load_config
from .verify import load_extrinsic


def colorize(png, pcd, R, t, K, brightness=1.0):
    """Return (P (N,3), colors_bgr (N,3) uint8, inview mask)."""
    img = cv2.imread(png)
    if img is None:
        raise FileNotFoundError(png)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    xyzi = pcd_io.load_xyzi(pcd)
    P = xyzi[:, :3]
    Pc = (R @ P.T).T + t
    z = Pc[:, 2]
    uv = (K @ Pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    inview = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    colors = np.zeros((len(P), 3), np.uint8)
    ui = uv[inview].astype(int)
    colors[inview] = img[ui[:, 1], ui[:, 0]]
    if brightness != 1.0:
        colors = np.clip(colors.astype(np.float32) * brightness + 15,
                         0, 255).astype(np.uint8)
    return P, colors, inview


def write_colored_pcd(path, P, colors_bgr):
    """Write XYZRGB pcd (rgb packed as float, PCL convention)."""
    packed = ((colors_bgr[:, 2].astype(np.uint32) << 16) |
              (colors_bgr[:, 1].astype(np.uint32) << 8) |
              colors_bgr[:, 0].astype(np.uint32)).astype(np.uint32)
    packed_f = np.frombuffer(np.ascontiguousarray(packed).tobytes(),
                             dtype=np.float32)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
                "TYPE F F F F\nCOUNT 1 1 1 1\n"
                f"WIDTH {len(P)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {len(P)}\nDATA ascii\n")
        for (x, y, z), c in zip(P, packed_f):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {c:.9e}\n")


def render_view(P, colors, eye, look, fov_deg=60, size=1100, rad=3):
    """Painter's-sort point splat from a virtual camera (Z-up world)."""
    eye = np.asarray(eye, float)
    look = np.asarray(look, float)
    fwd = look - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, fwd)
    Rv = np.stack([right, -up, fwd])
    p = (Rv @ (P - eye).T).T
    m = p[:, 2] > 0.15
    pp, cc = p[m], colors[m]
    f = size / (2 * np.tan(np.radians(fov_deg) / 2))
    u = f * pp[:, 0] / pp[:, 2] + size / 2
    v = f * pp[:, 1] / pp[:, 2] + size * 0.48
    canvas = np.full((size, size, 3), 8, np.uint8)
    order = np.argsort(-pp[:, 2])
    for x_, y_, col in zip(u[order].astype(int), v[order].astype(int), cc[order]):
        if 2 <= x_ < size - 2 and 2 <= y_ < size - 2:
            cv2.circle(canvas, (x_, y_), rad,
                       (int(col[0]), int(col[1]), int(col[2])), -1)
    return canvas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('png', help='image (NNNN.png or NNNN_L.png; pcd sibling used)')
    ap.add_argument('--pcd', default=None)
    ap.add_argument('--extrinsic', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--camera-info', default=None)
    ap.add_argument('--out', default=None, help='output colored pcd path')
    ap.add_argument('--views', action='store_true',
                    help='also render front/three-quarter view PNGs')
    args = ap.parse_args(argv)

    cfg, _ = load_config(args.config)
    K = np.array(cfg['camera']['K'], float).reshape(3, 3)
    if args.camera_info:
        with open(args.camera_info) as f:
            ci = yaml.safe_load(f)
        K = np.array(ci.get('K', cfg['camera']['K']), float).reshape(3, 3)

    if args.pcd:
        pcd = args.pcd
    elif args.png.endswith('_L.png'):
        pcd = args.png[:-6] + '.pcd'
    else:
        pcd = args.png[:-4] + '.pcd'

    R, t = load_extrinsic(args.extrinsic)
    debug_dir = cfg['output'].get('debug_dir', 'calib_debug')
    out = args.out or os.path.join(debug_dir, 'colored_cloud.pcd')

    P, colors, inview = colorize(args.png, pcd, R, t, K)
    write_colored_pcd(out, P, colors)
    print(f"wrote {out}  ({int(inview.sum())}/{len(P)} points colored)")

    if args.views:
        Pv, Cv = P[inview], np.clip(
            colors[inview].astype(np.float32) * 1.6 + 15, 0, 255).astype(np.uint8)
        c = Pv.mean(0)
        root = os.path.splitext(out)[0]
        for name, eye in [('front', [-0.8, 0.1, 0.3]),
                          ('right34', c + [-1.7, -1.9, 1.1]),
                          ('left34', c + [-1.4, 2.0, 1.2])]:
            img = render_view(Pv, Cv, eye, c)
            cv2.imwrite(f"{root}_{name}.png", img)
            print(f"wrote {root}_{name}.png")
    return 0


if __name__ == '__main__':
    sys.exit(main())
