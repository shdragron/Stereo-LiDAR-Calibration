"""Project the LiDAR cloud onto the ZED image with a solved extrinsic.

A well-calibrated result shows LiDAR points landing on the matching image
structures (board edges, floor, objects). Saves an overlay to the debug dir.

Usage:
  ros2 run charuco_lidar_calib verify captures/2026.../0000.png \
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


def load_extrinsic(path):
    with open(path) as f:
        e = yaml.safe_load(f)
    l2c = e['lidar_to_camera']
    R = np.array(l2c['R'], float)
    t = np.array(l2c['t'], float)
    return R, t


def project(png, pcd, R, t, K, D, out_path, max_range=6.0, point_size=2,
            color='depth', zoom_crop=None):
    """color: 'depth' or 'intensity'. With a good extrinsic, intensity
    coloring makes the board's white/black checker pattern visible in the
    LiDAR points, aligned with the image checkers — a sharp visual check.

    zoom_crop: optional (x0, y0, x1, y1) image rect; additionally writes a
    2x-magnified crop next to out_path (suffix _zoom)."""
    img = cv2.imread(png)
    if img is None:
        raise FileNotFoundError(png)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    xyzi = pcd_io.load_xyzi(pcd)
    P = xyzi[:, :3]
    inten = xyzi[:, 3]
    Pc = (R @ P.T).T + t                      # camera frame
    front = Pc[:, 2] > 0.05
    Pc = Pc[front]
    inten = inten[front]
    rvec = np.zeros(3); tvec = np.zeros(3)
    uv, _ = cv2.projectPoints(Pc, rvec, tvec, K.astype(float), D.astype(float))
    uv = uv.reshape(-1, 2)
    if color == 'intensity':
        lo, hi = np.percentile(inten, [2, 98])
        cn = np.clip((inten - lo) / max(hi - lo, 1e-9), 0, 1)
        cmap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
        cols = cv2.applyColorMap((cn * 255).astype(np.uint8), cmap).reshape(-1, 3)
        label = f"intensity-colored (p2={lo:.0f}..p98={hi:.0f})"
    else:
        depth = np.linalg.norm(Pc, axis=1)
        dn = np.clip(depth / max_range, 0, 1)
        cols = cv2.applyColorMap((dn * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET).reshape(-1, 3)
        label = "depth-colored"
    overlay = img.copy()
    n_drawn = 0
    for (u, v), c in zip(uv, cols):
        iu, iv = int(round(u)), int(round(v))
        if 0 <= iu < w and 0 <= iv < h:
            cv2.circle(overlay, (iu, iv), point_size,
                       (int(c[0]), int(c[1]), int(c[2])), -1)
            n_drawn += 1
    blended = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
    cv2.putText(blended, f"projected {n_drawn} lidar pts ({label})",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, blended)
    if zoom_crop is not None:
        x0, y0, x1, y1 = [int(v) for v in zoom_crop]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            crop = blended[y0:y1, x0:x1]
            crop = cv2.resize(crop, None, fx=2, fy=2,
                              interpolation=cv2.INTER_NEAREST)
            root, ext = os.path.splitext(out_path)
            cv2.imwrite(root + '_zoom' + ext, crop)
    return n_drawn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('png', help='image (its NNNN.pcd sibling is used)')
    ap.add_argument('--pcd', default=None)
    ap.add_argument('--extrinsic', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--camera-info', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--color', choices=['depth', 'intensity'], default='depth')
    ap.add_argument('--zoom-board', action='store_true',
                    help='also write a 2x crop around the detected ChArUco board')
    args = ap.parse_args(argv)

    cfg, _ = load_config(args.config)
    K = np.array(cfg['camera']['K'], float).reshape(3, 3)
    D = np.array(cfg['camera']['D'], float)
    if args.camera_info:
        with open(args.camera_info) as f:
            ci = yaml.safe_load(f)
        K = np.array(ci.get('K', cfg['camera']['K']), float).reshape(3, 3)
        D = np.array(ci.get('D', cfg['camera']['D']), float)

    if args.pcd:
        pcd = args.pcd
    elif args.png.endswith('_L.png'):          # stereo layout: NNNN_L.png
        pcd = args.png[:-6] + '.pcd'
    else:                                      # mono layout: NNNN.png
        pcd = args.png[:-4] + '.pcd'
    R, t = load_extrinsic(args.extrinsic)
    debug_dir = cfg['output'].get('debug_dir', 'calib_debug')
    out = args.out or os.path.join(debug_dir,
                                   'fusion_' + os.path.basename(args.png))

    zoom = None
    if args.zoom_board:
        # find the board in the image to crop around it
        from .calibrate import build_board
        from . import charuco_pose as CP
        img = cv2.imread(args.png)
        res = CP.estimate(img, K, D, build_board(cfg))
        if res.ok:
            proj, _ = cv2.projectPoints(res.corners_cam, np.zeros(3),
                                        np.zeros(3), K, np.zeros(5))
            proj = proj.reshape(-1, 2)
            m = 60
            zoom = (proj[:, 0].min() - m, proj[:, 1].min() - m,
                    proj[:, 0].max() + m, proj[:, 1].max() + m)
        else:
            print("(zoom-board: ChArUco not detected, skipping crop)")

    n = project(args.png, pcd, R, t, K, D, out, color=args.color,
                zoom_crop=zoom, point_size=(3 if args.color == 'intensity' else 2))
    print(f"wrote {out}  ({n} points projected)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
