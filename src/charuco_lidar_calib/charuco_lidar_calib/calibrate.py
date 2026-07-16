"""Offline ZED2i <-> RS-16 extrinsic calibration from captured png+pcd pairs.

Pipeline per capture pair:
  1. camera : ChArUco -> board pose, 4 outer corners, metric plane (n_c, d_c)
  2. lidar  : ROI (interactive / auto) -> RANSAC plane -> ordered corners + flags
  3. build a PlaneSample (plane + corners-if-reliable)
Then solve_plane_based over all poses -> extrinsic (lidar -> camera).

Usage (offline):
  ros2 run charuco_lidar_calib calibrate captures/2026*  --roi interactive
  # headless / board isolated:
  ros2 run charuco_lidar_calib calibrate captures/2026*  --roi auto
"""
import argparse
import glob
import os
import sys

import numpy as np
import cv2
import yaml

from . import board as B
from . import charuco_pose as CP
from . import pcd_io
from . import lidar_board as LB
from . import solve


# --------------------------------------------------------------------------
def find_config(path_arg):
    if path_arg and os.path.exists(path_arg):
        return path_arg
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory('charuco_lidar_calib'),
                         'config', 'calib.yaml')
        if os.path.exists(p):
            return p
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, 'config', 'calib.yaml')


def load_config(path_arg):
    cfg_path = find_config(path_arg)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg_path


def build_board(cfg):
    b = cfg['board']
    keys = ['squares_x', 'squares_y', 'square_length', 'marker_length',
            'dictionary', 'legacy', 'min_charuco_corners']
    return B.BoardModel(**{k: b[k] for k in keys})


def camera_plane(res):
    """Return (n_c, d_c) with n_c.X + d_c = 0, n_c pointing toward the camera."""
    n = np.asarray(res.normal_cam, float)
    p0 = np.asarray(res.corners_cam[0], float)
    d = -float(n @ p0)
    return n, d


# approx camera-optical -> lidar mount rotation (X_fwd, Y_left, Z_up)
R_CAM_TO_LIDAR = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def load_seed_extrinsic(path):
    """Read camera->lidar (R, t) from a previously solved extrinsic yaml."""
    with open(path) as f:
        e = yaml.safe_load(f)
    c2l = e['camera_to_lidar']
    return np.array(c2l['R'], float), np.array(c2l['t'], float)


def auto_roi_from_camera(res, seed=None, margin=0.3):
    """Bounding-box ROI in the lidar frame from the camera board corners.

    With `seed` = (R, t) camera->lidar from a previous calibration the box is
    placed exactly; without it the nominal mount rotation with ZERO translation
    is used, which only works when camera and lidar sit close together
    (< margin apart) — on-car rigs should pass --seed-extrinsic or use
    --roi interactive."""
    if seed is not None:
        R_cl, t_cl = seed
        cl = (R_cl @ res.corners_cam.T).T + t_cl
    else:
        cl = (R_CAM_TO_LIDAR @ res.corners_cam.T).T
    lo = cl.min(0) - margin
    hi = cl.max(0) + margin
    return ([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]])


def list_pairs(inputs):
    """Expand dirs/globs into (png_left, png_right_or_None, pcd) triplets.

    Supports two capture layouts:
      NNNN.png + NNNN.pcd                     (mono)
      NNNN_L.png + NNNN_R.png + NNNN.pcd      (stereo)
    """
    triplets = []
    paths = []
    for it in inputs:
        paths.extend(sorted(glob.glob(it)))

    def add_from_png(png):
        if png.endswith('_view.png') or png.endswith('_R.png'):
            return
        if png.endswith('_L.png'):
            base = png[:-6]
            right = base + '_R.png'
            pcd = base + '.pcd'
            if os.path.exists(pcd):
                triplets.append((png, right if os.path.exists(right) else None, pcd))
        else:
            pcd = png[:-4] + '.pcd'
            if os.path.exists(pcd):
                triplets.append((png, None, pcd))

    for p in paths:
        if os.path.isdir(p):
            for png in sorted(glob.glob(os.path.join(p, '*.png'))):
                add_from_png(png)
        elif p.endswith('.png'):
            add_from_png(p)
    return triplets


# --------------------------------------------------------------------------
# stereo acceptance gate: beyond these the rectification/baseline is suspect
# and the mono PnP pose is more trustworthy than a bad triangulation.
STEREO_MAX_EPI_PX = 1.5
STEREO_MAX_FIT_M = 0.015


def process_pair(png, pcd, cfg, bm, K, D, roi_mode, debug_dir, tag,
                 png_right=None, baseline=None, disp_offset=None,
                 roi_seed=None):
    img = cv2.imread(png)
    if img is None:
        print(f"  [skip] cannot read {png}")
        return None

    stereo_note = ''
    res = None
    if png_right is not None and baseline:
        from . import stereo_pose as SP
        img_r = cv2.imread(png_right)
        if img_r is not None:
            sres = SP.estimate_stereo(img, img_r, K, D, baseline, bm,
                                      disp_offset=disp_offset)
            if sres.ok and (sres.epipolar_rms > STEREO_MAX_EPI_PX
                            or sres.fit_rmse > STEREO_MAX_FIT_M):
                print(f"  [stereo->mono] {tag}: quality gate failed "
                      f"(epi={sres.epipolar_rms:.2f}px "
                      f"fit={sres.fit_rmse*1000:.1f}mm) — using mono PnP")
            elif sres.ok:
                res = sres            # duck-typed: has corners_cam/normal_cam
                stereo_note = (f" [stereo: matched={sres.n_matched} "
                               f"epi={sres.epipolar_rms:.2f}px "
                               f"fit={sres.fit_rmse*1000:.1f}mm "
                               f"dd={sres.disp_offset:+.2f}px]")
                if debug_dir:
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir, f'pair_{tag}_stereo.png'),
                                SP.annotate_stereo(img, img_r, sres, K))
    if res is None:
        res = CP.estimate(img, K, D, bm)
        if not res.ok:
            print(f"  [skip] {tag}: ChArUco not detected (markers={res.n_markers})")
            return None

    n_c, d_c = camera_plane(res)
    xyzi = pcd_io.load_xyzi(pcd)
    bw = bm.squares_x * bm.square_length
    bh = bm.squares_y * bm.square_length

    # --- ROI selection ---
    # capture-time ROI (drawn right after SPACE in sync_capture) wins: it was
    # drawn on exactly this cloud, so calibrate can run fully headless.
    mask = None
    roi_file = pcd[:-4] + '_roi.npy'
    if os.path.exists(roi_file):
        m = np.load(roi_file)
        if m.shape[0] == xyzi.shape[0]:
            mask = m.astype(bool)
            print(f"  [roi] capture-time ROI ({int(mask.sum())} pts)")
        else:
            print(f"  [roi] {os.path.basename(roi_file)} size mismatch "
                  f"({m.shape[0]} vs {xyzi.shape[0]}) — falling back to {roi_mode}")
    if mask is None and roi_mode == 'interactive':
        try:
            mask = LB.select_polygon(xyzi, cfg['lidar_extract'])
        except RuntimeError as e:
            print(f"  [skip] {tag}: {e}")
            return None
    elif mask is None:  # auto
        xr, yr, zr = auto_roi_from_camera(res, seed=roi_seed)
        mask = LB.box_roi_mask(xyzi[:, :3], xr, yr, zr)

    out = LB.extract_board(xyzi, mask, cfg['lidar_extract'], bw, bh)
    if out is None:
        print(f"  [skip] {tag}: lidar board extraction failed (roi pts={int(mask.sum())})")
        return None

    # camera corners -> lidar-normal must be toward lidar; already handled.
    if stereo_note:
        cam_desc = stereo_note.strip()
    else:
        cam_desc = f"cam corners={res.n_corners} reproj={res.reproj_err:.2f}px"
    print(f"  [ok] {tag}: {cam_desc} | "
          f"lidar inliers={out['n_inliers']} extent={out['meas_extent'][0]:.2f}x"
          f"{out['meas_extent'][1]:.2f} elev[{out['elev_range'][0]:.1f},"
          f"{out['elev_range'][1]:.1f}] "
          f"{'CLIPPED' if out['fov_clipped'] else 'full-FOV'} "
          f"corners_reliable={out['corners_reliable']}")

    # debug images
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        if not stereo_note:      # stereo debug image already saved above
            cv2.imwrite(os.path.join(debug_dir, f'pair_{tag}_cam.png'),
                        CP.annotate(img, K, D, bm, res))
        _save_lidar_debug(os.path.join(debug_dir, f'pair_{tag}_lidar.png'),
                          xyzi, mask, out, cfg['lidar_extract'])

    return solve.PlaneSample(
        lidar_pts=out['inlier_pts'], lidar_normal=out['normal'],
        cam_normal=n_c, cam_d=d_c,
        lidar_corners=out['corners'], cam_corners=res.corners_cam,
        corners_reliable=out['corners_reliable'], name=tag)


def _save_lidar_debug(path, xyzi, mask, out, lecfg):
    img, uv, idx = LB.front_view(xyzi, int(lecfg.get('front_view_size', 900)),
                                 lecfg.get('depth_min', 0.4), lecfg.get('depth_max', 4.0))
    # highlight inliers
    inlier_set = set(map(tuple, np.round(out['inlier_pts'], 4)))
    xyz = xyzi[:, :3]
    for k, gi in enumerate(idx):
        p = tuple(np.round(xyz[gi], 4))
        if p in inlier_set:
            cv2.circle(img, (int(uv[k, 0]), int(uv[k, 1])), 2, (255, 255, 255), -1)
    cv2.putText(img, f"inliers={out['n_inliers']} extent={out['meas_extent'][0]:.2f}"
                f"x{out['meas_extent'][1]:.2f} {'CLIPPED' if out['fov_clipped'] else 'full'}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imwrite(path, img)


# --------------------------------------------------------------------------
def write_extrinsic(cfg, R, t, metrics, out_path):
    Ri, ti = solve.invert(R, t)
    q = solve.matrix_to_quaternion(R)
    rpy = solve.euler_rpy(R)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    parent = cfg['camera']['optical_frame']
    child = cfg['lidar']['frame']
    stp = ("ros2 run tf2_ros static_transform_publisher "
           f"--x {t[0]:.6f} --y {t[1]:.6f} --z {t[2]:.6f} "
           f"--qx {q[0]:.6f} --qy {q[1]:.6f} --qz {q[2]:.6f} --qw {q[3]:.6f} "
           f"--frame-id {parent} --child-frame-id {child}")
    def fl(x):
        return [[float(v) for v in row] for row in x] if np.ndim(x) == 2 \
            else [float(v) for v in np.ravel(x)]
    data = {
        'note': 'camera_point = R * lidar_point + t  (lidar -> camera optical)',
        'lidar_to_camera': {
            'parent_frame': parent, 'child_frame': child,
            'R': fl(R), 't': fl(t),
            'quaternion_xyzw': fl(q),
            'rpy_rad': fl(np.array(rpy)),
            'matrix_4x4': fl(T),
        },
        'camera_to_lidar': {'R': fl(Ri), 't': fl(ti)},
        'static_transform_publisher': stp,
        'metrics': metrics,
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
    return stp


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='capture dirs / globs / png files')
    ap.add_argument('--config', default=None)
    ap.add_argument('--camera-info', default=None, help='yaml with K/D override')
    ap.add_argument('--roi', choices=['interactive', 'auto'], default='interactive')
    ap.add_argument('--out', default=None, help='output extrinsic yaml')
    ap.add_argument('--debug-dir', default=None)
    ap.add_argument('--no-refine', action='store_true')
    ap.add_argument('--no-corners', action='store_true')
    ap.add_argument('--min-frames', type=int, default=None,
                    help='override solve.min_frames (e.g. 2 for a quick test)')
    ap.add_argument('--trust-corners', action='store_true',
                    help='treat lidar corners as reliable even when FOV-clipped '
                         '(test runs only: clipping is ~symmetric so the '
                         'known-size fit center is still decent)')
    ap.add_argument('--corner-weight', type=float, default=0.05,
                    help='refine weight of corner residuals vs plane (default '
                         '0.05 = noise-balanced; corners are ring-quantized '
                         'to ~2 cm while planes are mm-accurate)')
    ap.add_argument('--seed-extrinsic', default=None,
                    help='previous extrinsic yaml to place the auto ROI '
                         '(required for --roi auto when camera and lidar are '
                         'far apart, e.g. on-car)')
    args = ap.parse_args(argv)

    cfg, cfg_path = load_config(args.config)
    print(f"config: {cfg_path}")
    bm = build_board(cfg)
    K = np.array(cfg['camera']['K'], float).reshape(3, 3)
    D = np.array(cfg['camera']['D'], float)
    baseline = float(cfg['camera'].get('baseline', 0.0))
    if args.camera_info:
        with open(args.camera_info) as f:
            ci = yaml.safe_load(f)
        K = np.array(ci.get('K', cfg['camera']['K']), float).reshape(3, 3)
        D = np.array(ci.get('D', cfg['camera']['D']), float)
        baseline = float(ci.get('baseline', baseline))
        print(f"camera_info override from {args.camera_info}")
    else:
        print(f"[WARN] no --camera-info: using the config K fallback "
              f"(fx={K[0,0]:.2f}). If the camera was swapped or recalibrated "
              f"this silently produces a WRONG extrinsic — grab live "
              f"intrinsics with `ros2 run charuco_lidar_calib "
              f"grab_camera_info` and pass the yaml.")
    if baseline:
        print(f"stereo baseline = {baseline*1000:.2f} mm")

    roi_seed = None
    if args.roi == 'auto':
        seed_path = args.seed_extrinsic
        if seed_path is None:
            default_seed = cfg['output'].get('extrinsic_yaml', '')
            if default_seed and os.path.exists(default_seed):
                seed_path = default_seed
        if seed_path and os.path.exists(seed_path):
            roi_seed = load_seed_extrinsic(seed_path)
            print(f"auto ROI seeded from {seed_path}")
        else:
            print("[WARN] auto ROI without a seed extrinsic assumes camera "
                  "and lidar are nearly co-located; on-car use "
                  "--seed-extrinsic <yaml> or --roi interactive.")

    debug_dir = args.debug_dir or cfg['output'].get('debug_dir', 'calib_debug')
    out_path = args.out or cfg['output'].get('extrinsic_yaml',
                                             'calib_debug/extrinsic_zed_rslidar.yaml')

    pairs = list_pairs(args.inputs)
    if not pairs:
        print("No (png,pcd) pairs found in inputs."); return 1
    print(f"found {len(pairs)} capture pair(s). ROI mode = {args.roi}\n")

    # --- rig-wide disparity-offset pre-pass (rectification toe-in) ---------
    # The offset is a property of the rig, not of a frame: estimate it per
    # stereo frame from the board's known size and use the median everywhere.
    disp_offset = None
    stereo_frames = [(p, r) for p, r, _ in pairs if r and baseline]
    if stereo_frames:
        from . import stereo_pose as SP
        offs = []
        for p, r in stereo_frames:
            il, ir_ = cv2.imread(p), cv2.imread(r)
            if il is None or ir_ is None:
                continue
            s = SP.estimate_stereo(il, ir_, K, D, baseline, bm)
            if s.ok and abs(s.disp_offset) > 1e-6:
                offs.append(s.disp_offset)
        if offs:
            disp_offset = float(np.median(offs))
            print(f"stereo disparity offset (rig-wide, median of "
                  f"{len(offs)} confident frame(s)) = {disp_offset:+.2f} px")

    samples = []
    for i, (png, png_r, pcd) in enumerate(pairs):
        tag = f"{i:03d}"
        mode = 'stereo' if (png_r and baseline) else 'mono'
        print(f"[{i+1}/{len(pairs)}] {os.path.relpath(png)} ({mode})")
        s = process_pair(png, pcd, cfg, bm, K, D, args.roi, debug_dir, tag,
                         png_right=png_r, baseline=baseline,
                         disp_offset=disp_offset, roi_seed=roi_seed)
        if s is not None:
            if args.trust_corners:
                s.corners_reliable = True
            samples.append(s)

    min_frames = args.min_frames or cfg['solve'].get('min_frames', 3)
    print(f"\nusable poses: {len(samples)}")
    if len(samples) < max(1, min_frames):
        print(f"Need >= {min_frames} good board poses (varied orientation). "
              f"Capture more poses and rerun.")
        return 1

    n_reliable = sum(s.corners_reliable for s in samples)
    try:
        res = solve.solve_plane_based(samples, use_corners=(not args.no_corners),
                                      refine=(not args.no_refine),
                                      w_corner=args.corner_weight)
    except ValueError as e:
        print(f"\n[FAIL] {e}")
        return 1

    print("\n==================== RESULT (lidar -> camera) ====================")
    print("R =\n", np.round(res.R, 5))
    print("t =", np.round(res.t, 5), "m")
    print(f"poses={res.n_poses}  corners_reliable={n_reliable}")
    print(f"plane RMSE = {res.rmse_plane*1000:.1f} mm")
    print(f"normal RMS = {res.normal_deg_rms:.3f} deg")
    if np.isfinite(res.corner_rmse):
        print(f"corner RMSE = {res.corner_rmse*1000:.1f} mm")
    print(f"translation conditioning = {res.t_condition:.1f} "
          f"({'GOOD' if res.t_condition < 50 else 'WEAK - add more orientation variety'})")
    print(f"rotation normal spread = {res.rot_spread:.3f} "
          f"({'GOOD' if res.rot_spread > 0.15 else 'WEAK - add yaw/pitch tilt variety'})")
    for pp in res.per_pose:
        print(f"   {pp['name']}: plane={pp['plane_mm']:.1f}mm normal={pp['normal_deg']:.2f}deg"
              + (f" corner={pp['corner_mm']:.1f}mm" if pp['corner_mm'] else ""))

    metrics = {
        'rmse_plane_mm': round(res.rmse_plane * 1000, 2),
        'normal_deg_rms': round(res.normal_deg_rms, 4),
        'corner_rmse_mm': (round(res.corner_rmse * 1000, 2)
                           if np.isfinite(res.corner_rmse) else None),
        'n_poses': res.n_poses, 'n_corners_reliable': n_reliable,
        'translation_condition': round(res.t_condition, 1),
        'rotation_normal_spread': round(res.rot_spread, 4),
    }
    stp = write_extrinsic(cfg, res.R, res.t, metrics, out_path)
    print(f"\nwrote {out_path}")
    print("\npublish the extrinsic as a static TF with:\n  " + stp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
