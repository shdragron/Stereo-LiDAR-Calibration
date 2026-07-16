"""Camera-side: detect the ChArUco board and return its pose + outer corners.

Returns everything in the ZED *left optical* frame (X right, Y down, Z forward).
"""
from dataclasses import dataclass
import numpy as np
import cv2

from .board import (BoardModel, detect_markers, detect_charuco_multiscale,
                    estimate_board_pose, chessboard_obj_points)


@dataclass
class CharucoResult:
    ok: bool
    n_markers: int = 0
    n_corners: int = 0
    rvec: np.ndarray = None          # board->camera (Rodrigues)
    tvec: np.ndarray = None          # board origin in camera frame [m]
    R: np.ndarray = None             # 3x3 board->camera
    corners_cam: np.ndarray = None   # 4x3 outer corners in camera frame [m]
    normal_cam: np.ndarray = None    # unit board normal in camera frame
    reproj_err: float = float('nan') # mean reprojection error [px]


def _to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def estimate(img, K, D, board: BoardModel):
    """img: BGR/BGRA/gray ndarray. Returns CharucoResult."""
    bgr = _to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).ravel()

    corners, ids = detect_markers(gray, board)
    n_markers = 0 if ids is None else len(ids)
    n_corners, ch_corners, ch_ids = detect_charuco_multiscale(gray, board, K, D)
    if n_corners < board.min_charuco_corners:
        return CharucoResult(ok=False, n_markers=n_markers, n_corners=n_corners)

    ok, rvec, tvec = estimate_board_pose(ch_corners, ch_ids, board, K, D)
    if not ok:
        return CharucoResult(ok=False, n_markers=n_markers, n_corners=n_corners)

    R, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape(3)
    outer = board.outer_corners_board_frame()               # 4x3, board frame
    corners_cam = (R @ outer.T + tvec.reshape(3, 1)).T       # 4x3, camera frame
    normal_cam = R @ np.array([0.0, 0.0, 1.0])
    normal_cam = normal_cam / (np.linalg.norm(normal_cam) + 1e-12)
    # planar-pose ambiguity can flip the board z axis; the solver requires the
    # normal to point toward the camera (same guard as stereo_pose)
    if normal_cam @ corners_cam.mean(0) > 0:
        normal_cam = -normal_cam

    # reprojection error over the interior charuco corners (sanity metric)
    err = _reproj_error(ch_corners, ch_ids, board, R, tvec, K, D)

    return CharucoResult(ok=True, n_markers=n_markers, n_corners=n_corners,
                         rvec=rvec.reshape(3), tvec=tvec, R=R,
                         corners_cam=corners_cam, normal_cam=normal_cam,
                         reproj_err=err)


def _reproj_error(ch_corners, ch_ids, board, R, tvec, K, D):
    """Mean reprojection error [px] over the interior charuco corners."""
    try:
        cb = chessboard_obj_points(board)
        obj = cb[ch_ids.ravel()].reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(obj, rvec, tvec.reshape(3, 1), K, D)
        obs = ch_corners.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - obs, axis=1)))
    except Exception:
        return float('nan')


def annotate(img, K, D, board: BoardModel, res: CharucoResult):
    """Return a BGR debug image with markers, corners, axes and C0..C3."""
    import cv2.aruco as aruco
    bgr = _to_bgr(img).copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).ravel()
    corners, ids = detect_markers(gray, board)
    if ids is not None:
        aruco.drawDetectedMarkers(bgr, corners, ids)
    if not res.ok:
        cv2.putText(bgr, "ChArUco NOT detected", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return bgr
    rvec = np.asarray(res.rvec, float).reshape(3, 1)
    tvec = np.asarray(res.tvec, float).reshape(3, 1)
    cv2.drawFrameAxes(bgr, K, D, rvec, tvec, 0.3, 4)
    outer = board.outer_corners_board_frame()
    proj, _ = cv2.projectPoints(outer, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2).astype(int)
    for i, p in enumerate(proj):
        cv2.circle(bgr, tuple(p), 9, (0, 0, 255), -1)
        cv2.putText(bgr, f"C{i}", tuple(p + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.polylines(bgr, [proj.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
    cv2.putText(bgr, f"corners={res.n_corners}  reproj={res.reproj_err:.2f}px",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return bgr
