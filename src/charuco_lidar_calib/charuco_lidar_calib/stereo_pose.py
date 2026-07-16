"""Stereo (left/right) ChArUco: triangulated metric board geometry.

Uses the RECTIFIED left/right pair with the factory baseline (epipolar
geometry): matched interior corners are triangulated per-id, giving each
corner an independent metric 3D position in the LEFT camera frame — no PnP
depth/normal ambiguity. The board pose is then a rigid (Kabsch) fit of the
known board-frame corner layout to the triangulated points, which also yields
the 4 outer corners and the plane.

Rectified-pair epipolar constraint (same row) is used as a quality check.
"""
from dataclasses import dataclass
import numpy as np
import cv2

from .board import BoardModel, detect_charuco_multiscale, chessboard_obj_points
from .solve import solve_rigid_3d3d


@dataclass
class StereoCharucoResult:
    ok: bool
    n_matched: int = 0               # corners matched L<->R by id
    epipolar_rms: float = float('nan')   # |v_L - v_R| RMS [px] (rectified: ~0)
    fit_rmse: float = float('nan')   # Kabsch fit residual board->triangulated [m]
    disp_offset: float = 0.0         # estimated systematic disparity bias [px]
    R: np.ndarray = None             # board -> left-camera
    t: np.ndarray = None
    corners_cam: np.ndarray = None   # 4x3 outer corners, left-cam frame [m]
    normal_cam: np.ndarray = None    # unit normal (toward camera)
    tri_pts: np.ndarray = None       # (N,3) triangulated interior corners
    tri_ids: np.ndarray = None       # (N,) their charuco ids


def _detect_corners(img, board: BoardModel, K, D):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    n, chc, chi = detect_charuco_multiscale(gray, board, K, D)
    if n == 0 or chi is None:
        return None, None
    return chc.reshape(-1, 2), chi.ravel()


def estimate_stereo(img_l, img_r, K, D, baseline, board: BoardModel,
                    disp_offset=None):
    """K, D: LEFT rectified intrinsics (D should be ~0). baseline [m] > 0.

    Rectified triangulation:  d = uL - uR - disp_offset,  Z = fx*B/d,
                              X = (uL-cx)Z/fx,  Y = (vL-cy)Z/fy.

    disp_offset: systematic disparity bias [px] (rectification toe-in).
      None  -> self-estimate from the board's known metric size (accepted only
               when it clearly improves the rigid fit; a rig-wide median over
               several frames is more robust — see calibrate.py).
      float -> use the given value as-is.
    """
    K = np.asarray(K, np.float64).reshape(3, 3)
    D = np.asarray(D, np.float64).ravel()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    cl, il = _detect_corners(img_l, board, K, D)
    cr, ir = _detect_corners(img_r, board, K, D)
    if cl is None or cr is None:
        return StereoCharucoResult(ok=False)

    common, li, ri = np.intersect1d(il, ir, return_indices=True)
    if len(common) < board.min_charuco_corners:
        return StereoCharucoResult(ok=False, n_matched=len(common))

    pl = cl[li]                       # (N,2) left pixels
    pr = cr[ri]                       # (N,2) right pixels

    # epipolar (rectified): rows must agree
    dv = pl[:, 1] - pr[:, 1]
    epi_rms = float(np.sqrt(np.mean(dv ** 2)))

    disparity = pl[:, 0] - pr[:, 0]
    good = disparity > 0.5            # in front, sane
    if good.sum() < board.min_charuco_corners:
        return StereoCharucoResult(ok=False, n_matched=int(good.sum()),
                                   epipolar_rms=epi_rms)
    pl, pr, disparity, common = pl[good], pr[good], disparity[good], common[good]
    v_avg = 0.5 * (pl[:, 1] + pr[:, 1])   # average row: halves row noise

    obj_all = chessboard_obj_points(board)
    obj = obj_all[common.astype(int)]

    def triangulate(dd):
        d_eff = disparity - dd
        Z = fx * baseline / d_eff
        X = (pl[:, 0] - cx) * Z / fx
        Y = (v_avg - cy) * Z / fy
        return np.column_stack([X, Y, Z])

    # --- estimate the systematic disparity bias (rectification toe-in) -----
    # The board is a metric ruler: a constant disparity offset warps the
    # triangulated grid so it no longer rigid-fits the known layout. Find the
    # offset that minimizes the rigid-fit RMSE (independent of mono PnP).
    def fit_rmse_for(dd):
        d_eff = disparity - dd
        if np.any(d_eff < 0.5):
            return 1e9
        return solve_rigid_3d3d(obj, triangulate(dd)).rmse

    if disp_offset is None:
        disp_offset = 0.0
        try:
            from scipy.optimize import minimize_scalar
            # ±8 px: this rig's gen1 ZED shows a real ~4-5 px toe-in (verified
            # against lidar plane distances 2026-07-17); ±4 pegged at the bound
            opt = minimize_scalar(fit_rmse_for, bounds=(-8.0, 8.0),
                                  method='bounded', options={'xatol': 1e-3})
            # accept only a clear improvement (avoid overfitting noise)
            if opt.fun < 0.8 * fit_rmse_for(0.0):
                disp_offset = float(opt.x)
        except Exception:
            pass
    else:
        disp_offset = float(disp_offset)

    tri = triangulate(disp_offset)        # (N,3) left-cam frame

    # rigid fit: board-frame corner layout -> triangulated points
    rr = solve_rigid_3d3d(obj, tri)       # "lidar"=board, "camera"=cam frame
    R, t = rr.R, rr.t

    outer = board.outer_corners_board_frame()
    corners_cam = (R @ outer.T + t.reshape(3, 1)).T
    normal = R @ np.array([0.0, 0.0, 1.0])
    normal /= (np.linalg.norm(normal) + 1e-12)
    if normal @ (-corners_cam.mean(0)) < 0:   # make it face the camera
        normal = -normal

    return StereoCharucoResult(
        ok=True, n_matched=int(len(common)), epipolar_rms=epi_rms,
        fit_rmse=rr.rmse, disp_offset=disp_offset, R=R, t=t,
        corners_cam=corners_cam, normal_cam=normal, tri_pts=tri, tri_ids=common)


def annotate_stereo(img_l, img_r, res: StereoCharucoResult, K=None):
    """Side-by-side debug image with matched corners + metrics."""
    def prep(im):
        if im.ndim == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[2] == 4:
            return cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        return im.copy()
    L, R = prep(img_l), prep(img_r)
    canvas = np.hstack([L, R])
    txt = (f"matched={res.n_matched} epipolarRMS={res.epipolar_rms:.2f}px "
           f"fitRMSE={res.fit_rmse*1000:.1f}mm" if res.ok else "stereo charuco FAILED")
    cv2.putText(canvas, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 255) if res.ok else (0, 0, 255), 2)
    if res.ok and K is not None:
        K = np.asarray(K, np.float64).reshape(3, 3)
        uv, _ = cv2.projectPoints(res.tri_pts, np.zeros(3), np.zeros(3), K,
                                  np.zeros(5))
        for (u, v) in uv.reshape(-1, 2).astype(int):
            cv2.circle(canvas, (u, v), 5, (0, 255, 0), 2)
        # outer corners on the left half
        proj, _ = cv2.projectPoints(res.corners_cam, np.zeros(3), np.zeros(3),
                                    K, np.zeros(5))
        proj = proj.reshape(-1, 2).astype(int)
        for i, p in enumerate(proj):
            cv2.circle(canvas, tuple(p), 9, (0, 0, 255), -1)
            cv2.putText(canvas, f"C{i}", tuple(p + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.polylines(canvas, [proj.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
    return canvas
