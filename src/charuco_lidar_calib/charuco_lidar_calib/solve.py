"""3D-3D rigid solver (Kabsch/Umeyama) + aggregation.

Ported verbatim (math-wise) from the legacy ``calc_RT`` in Find_RT.h:
    cov = camera_centered @ lidar_centered^T
    U, S, Vt = svd(cov)
    R = U @ Vt   (with det<0 sign correction on the last column)
    t = mu_camera - R @ mu_lidar
    R, t map LiDAR-frame points into the CAMERA frame.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class RigidResult:
    R: np.ndarray       # 3x3, maps lidar -> camera
    t: np.ndarray       # 3,   camera = R @ lidar + t
    rmse: float
    n: int


def solve_rigid_3d3d(lidar_pts, camera_pts):
    """lidar_pts, camera_pts: (N,3). Returns RigidResult (lidar->camera)."""
    L = np.asarray(lidar_pts, dtype=np.float64).reshape(-1, 3).T   # 3xN
    C = np.asarray(camera_pts, dtype=np.float64).reshape(-1, 3).T  # 3xN
    n = L.shape[1]
    assert n == C.shape[1] and n >= 3, "need >=3 matched points"

    mu_l = L.mean(axis=1, keepdims=True)
    mu_c = C.mean(axis=1, keepdims=True)
    Lc = L - mu_l
    Cc = C - mu_c

    cov = Cc @ Lc.T
    U, _, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        D = np.diag([1.0, 1.0, -1.0])
        R = U @ D @ Vt
    t = (mu_c - R @ mu_l).ravel()

    resid = C - (R @ L + t.reshape(3, 1))
    rmse = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=0))))
    return RigidResult(R=R, t=t, rmse=rmse, n=n)


def invert(R, t):
    """Return (R_inv, t_inv) mapping the other direction."""
    R = np.asarray(R, float)
    t = np.asarray(t, float).ravel()
    R_inv = R.T
    t_inv = -R.T @ t
    return R_inv, t_inv


def matrix_to_quaternion(R):
    """3x3 rotation -> (qx, qy, qz, qw)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw])
    return q / (np.linalg.norm(q) + 1e-12)


def euler_rpy(R):
    """Return (roll, pitch, yaw) in radians (XYZ / rpy, ZYX convention)."""
    R = np.asarray(R, float)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


# ==========================================================================
#  Plane-based (point-to-plane) solver  --  robust for sparse / FOV-clipped
#  LiDAR where board corners are not observable.
#
#  Per board pose we have:
#    - LiDAR board inlier points  P_i (in lidar frame)
#    - LiDAR board plane normal   n_l_i
#    - Camera board plane         (n_c_i, d_c_i):  n_c_i . X + d_c_i = 0
#    - (optional) ordered 4 corners on both sides, when the board is fully in
#      the LiDAR FOV (corners_reliable).
#
#  Solve T (lidar->camera) so that:
#    R n_l_i ~= n_c_i                                   (rotation from normals)
#    n_c_i . (R P_ij + t) + d_c_i ~= 0                  (point-to-plane -> t)
#    R c_l + t ~= c_c    for reliable corners           (fixes in-plane t)
# ==========================================================================
from dataclasses import dataclass, field   # noqa: E402


@dataclass
class PlaneSample:
    lidar_pts: np.ndarray                 # (M,3) board inliers, lidar frame
    lidar_normal: np.ndarray              # (3,) unit, toward lidar sensor
    cam_normal: np.ndarray               # (3,) unit, toward camera
    cam_d: float                          # camera plane offset
    lidar_corners: np.ndarray = None      # (4,3) ordered C0..C3 or None
    cam_corners: np.ndarray = None        # (4,3) ordered C0..C3 or None
    corners_reliable: bool = False
    name: str = ''


def solve_rotation_from_normals(n_lidar, n_cam):
    """Kabsch on unit normals: find R minimizing ||R n_l - n_c||. >=2 non-
    parallel pairs needed (>=3 recommended)."""
    Nl = np.asarray(n_lidar, float).reshape(-1, 3)
    Nc = np.asarray(n_cam, float).reshape(-1, 3)
    cov = Nc.T @ Nl
    U, _, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def solve_translation_point_to_plane(R, samples):
    """Given R, solve t (3,) minimizing sum point-to-plane residuals.
    Returns (t, condition_number)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for s in samples:
        P = np.asarray(s.lidar_pts, float).reshape(-1, 3)
        n = np.asarray(s.cam_normal, float)
        RP = (R @ P.T).T
        c = RP @ n + s.cam_d            # per-point residual const part
        A += len(P) * np.outer(n, n)
        b += -n * c.sum()
    # condition of A tells us if the board normals were diverse enough
    w = np.linalg.eigvalsh(A)
    cond = float(w.max() / max(w.min(), 1e-12))
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    return t, cond


def _residuals(x, samples, w_plane, w_corner):
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    t = x[3:6]
    res = []
    for s in samples:
        P = np.asarray(s.lidar_pts, float).reshape(-1, 3)
        n = np.asarray(s.cam_normal, float)
        RP = (R @ P.T).T + t
        r = (RP @ n + s.cam_d)
        # normalise per-sample so dense clouds don't dominate
        res.append(w_plane * r / np.sqrt(len(P)))
        if s.corners_reliable and s.lidar_corners is not None and s.cam_corners is not None:
            cl = np.asarray(s.lidar_corners, float)
            cc = np.asarray(s.cam_corners, float)
            rc = ((R @ cl.T).T + t) - cc
            res.append(w_corner * rc.ravel())
    return np.concatenate(res)


def refine_joint(R0, t0, samples, w_plane=1.0, w_corner=0.05):
    """Nonlinear refine of (R,t) over point-to-plane (+corner) residuals.

    w_corner default 0.05 balances the noise scales: corners (sigma ~2 cm,
    RS-16 ring quantization) vs plane (sigma ~3 mm, per-pose normalized) ->
    w ~ 0.003 / (0.02 * sqrt(12)) ~ 0.05. Validated on synthetic data: larger
    weights let corner noise drag the rotation (~1 deg at w=1..5) while 0.05
    keeps <0.2 deg AND still fixes the in-plane translation DOF that planes
    cannot observe when the board normals are near-parallel.
    """
    try:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except Exception:
        return R0, t0
    x0 = np.concatenate([Rotation.from_matrix(R0).as_rotvec(), np.asarray(t0, float)])
    sol = least_squares(_residuals, x0, args=(samples, w_plane, w_corner),
                        method='lm', max_nfev=200)
    R = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    t = sol.x[3:6]
    return R, t


@dataclass
class PlaneCalibResult:
    R: np.ndarray
    t: np.ndarray
    rmse_plane: float            # mean |point-to-plane| after solve [m]
    normal_deg_rms: float        # RMS angle between R n_l and n_c [deg]
    corner_rmse: float           # mean corner-to-corner error [m] (reliable only)
    n_poses: int
    t_condition: float           # conditioning of translation system
    rot_spread: float            # s2/s1 of the normal set (0 = normals parallel
                                 # -> rotation weakly constrained)
    per_pose: list = field(default_factory=list)


def _normals_rank(samples, tol=0.05):
    """Effective rank of the lidar-normal set (1 = all parallel)."""
    N = np.array([s.lidar_normal for s in samples], float)
    s = np.linalg.svd(N, compute_uv=False)
    return int((s > tol * s[0]).sum())


def solve_plane_based(samples, use_corners=True, refine=True, w_corner=0.05):
    if len(samples) < 1:
        raise ValueError("plane-based solve needs >=1 board pose")

    # rotation observability: singular values of the lidar-normal set.
    # s2/s1 near 0 means the board normals are nearly parallel and the
    # rotation is weakly constrained (mirror of t_condition for rotation).
    N = np.array([s.lidar_normal for s in samples], float)
    sv = np.linalg.svd(N, compute_uv=False)
    rot_spread = float(sv[1] / sv[0]) if len(sv) > 1 and sv[0] > 0 else 0.0

    rank = _normals_rank(samples)
    reliable = [s for s in samples
                if s.corners_reliable and s.lidar_corners is not None
                and s.cam_corners is not None]
    if rank >= 2:
        R = solve_rotation_from_normals([s.lidar_normal for s in samples],
                                        [s.cam_normal for s in samples])
        t, cond = solve_translation_point_to_plane(R, samples)
    elif reliable:
        # Degenerate normals (all board poses parallel): initialise the full
        # 6DOF from the reliable corner correspondences instead.
        L = np.vstack([s.lidar_corners for s in reliable])
        C = np.vstack([s.cam_corners for s in reliable])
        rr = solve_rigid_3d3d(L, C)
        R, t = rr.R, rr.t
        cond = float('inf')     # translation not plane-constrained -> flag it
    else:
        raise ValueError(
            "board normals are (nearly) parallel across all poses and no "
            "reliable corners are available - capture poses with varied tilt")
    if refine:
        wc = w_corner if use_corners else 0.0
        R, t = refine_joint(R, t, samples, w_corner=wc)

    # diagnostics
    plane_err, ang_err, corner_err = [], [], []
    per_pose = []
    for s in samples:
        P = np.asarray(s.lidar_pts, float).reshape(-1, 3)
        n = np.asarray(s.cam_normal, float)
        r = np.abs(((R @ P.T).T + t) @ n + s.cam_d)
        Rn = R @ np.asarray(s.lidar_normal, float)
        cosang = np.clip(abs(Rn @ n), -1, 1)
        ang = np.degrees(np.arccos(cosang))
        ce = np.nan
        if s.corners_reliable and s.lidar_corners is not None and s.cam_corners is not None:
            ce = float(np.mean(np.linalg.norm(
                ((R @ s.lidar_corners.T).T + t) - s.cam_corners, axis=1)))
            corner_err.append(ce)
        plane_err.append(r.mean()); ang_err.append(ang)
        per_pose.append(dict(name=s.name, plane_mm=float(r.mean() * 1000),
                             normal_deg=float(ang), corner_mm=(float(ce * 1000)
                             if np.isfinite(ce) else None)))
    return PlaneCalibResult(
        R=R, t=np.asarray(t, float),
        rmse_plane=float(np.sqrt(np.mean(np.concatenate(
            [np.atleast_1d(e) for e in plane_err]) ** 2))),
        normal_deg_rms=float(np.sqrt(np.mean(np.array(ang_err) ** 2))),
        corner_rmse=(float(np.mean(corner_err)) if corner_err else float('nan')),
        n_poses=len(samples), t_condition=cond, rot_spread=rot_spread,
        per_pose=per_pose)
