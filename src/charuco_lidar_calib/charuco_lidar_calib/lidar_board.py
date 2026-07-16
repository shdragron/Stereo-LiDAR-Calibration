"""LiDAR-side board extraction: ROI select -> RANSAC plane -> ordered 4 corners.

RoboSense frame convention: X forward, Y left, Z up.
Board corners are ordered to MATCH the camera side:
    C0 bottom-left, C1 bottom-right, C2 top-right, C3 top-left
using gravity (+Z = up) and (+Y = left) projected onto the board plane.
"""
import numpy as np
import cv2

Y_LEFT = np.array([0.0, 1.0, 0.0])    # lidar +Y  -> physical LEFT of the board
Z_UP = np.array([0.0, 0.0, 1.0])      # lidar +Z  -> physical UP


# --------------------------------------------------------------------------
# ROI masks
# --------------------------------------------------------------------------
def box_roi_mask(xyz, x_range=None, y_range=None, z_range=None):
    m = np.ones(len(xyz), dtype=bool)
    for i, rng in enumerate((x_range, y_range, z_range)):
        if rng is not None:
            m &= (xyz[:, i] >= rng[0]) & (xyz[:, i] <= rng[1])
    return m


# --------------------------------------------------------------------------
# BEV (top-down) view for ROI: the board is a line segment spatially separated
# from the background wall, so lassoing just the board is easy.
# --------------------------------------------------------------------------
def bev_view(xyzi, size=900, range_m=4.0):
    """Top-down view: forward(+X)=up, left(+Y)=left, colored by INTENSITY —
    the board's black/white pattern stands out against the surroundings.
    (The ROI only needs to hand RANSAC an easy region; rough lasso is fine.)
    Returns (img, uv, keep_idx) like front_view."""
    xyz = xyzi[:, :3]
    keep = (np.abs(xyz[:, 0]) < range_m) & (np.abs(xyz[:, 1]) < range_m)
    idx = np.where(keep)[0]
    P = xyz[keep]
    inten = xyzi[keep, 3] if xyzi.shape[1] > 3 else np.zeros(len(P))
    img = np.zeros((size, size, 3), np.uint8)
    c = size / 2.0
    scale = size / (2.0 * range_m)
    step = 1 if range_m <= 8 else 2
    for r in range(step, int(range_m) + 1, step):        # 거리 링
        cv2.circle(img, (int(c), int(c)), int(r * scale), (45, 45, 45), 1)
    cv2.line(img, (int(c), 0), (int(c), size), (45, 45, 45), 1)
    cv2.line(img, (0, int(c)), (size, int(c)), (45, 45, 45), 1)
    if len(P):
        u = c - P[:, 1] * scale
        v = c - P[:, 0] * scale
        uv = np.column_stack([u, v])
        # robust intensity normalization (RS-16: ~0..255, scene-dependent)
        hi = max(float(np.percentile(inten, 99)), 1e-6)
        ic = np.clip(inten / hi, 0, 1)
        cols = cv2.applyColorMap((ic * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET).reshape(-1, 3)
        for (uu, vv), col in zip(uv.astype(int), cols):
            if 0 <= uu < size and 0 <= vv < size:
                cv2.circle(img, (uu, vv), 2,
                           (int(col[0]), int(col[1]), int(col[2])), -1)
    else:
        uv = np.zeros((0, 2))
    cv2.circle(img, (int(c), int(c)), 4, (255, 255, 255), -1)   # ego
    return img, uv, idx


# --------------------------------------------------------------------------
# Front view (project onto image looking along +X) for ROI + debug
# --------------------------------------------------------------------------
def front_view(xyzi, size=900, depth_min=0.4, depth_max=4.0, pad=0.15):
    """Render a front view (image X = -Y = right, image Y = -Z = down),
    colored by depth. Returns (img, uv, keep_idx) where uv[k] is the pixel of
    point keep_idx[k], so a polygon test on uv maps back to point indices."""
    xyz = xyzi[:, :3]
    keep = (xyz[:, 0] > depth_min) & (xyz[:, 0] < depth_max)
    idx = np.where(keep)[0]
    P = xyz[keep]
    if len(P) == 0:
        return np.zeros((size, size, 3), np.uint8), np.zeros((0, 2)), idx
    horiz = -P[:, 1]        # right = -Y
    vert = -P[:, 2]         # down  = -Z
    hmin, hmax = horiz.min() - pad, horiz.max() + pad
    vmin, vmax = vert.min() - pad, vert.max() + pad
    span = max(hmax - hmin, vmax - vmin) + 1e-9
    u = ((horiz - hmin) / span * (size - 20) + 10)
    v = ((vert - vmin) / span * (size - 20) + 10)
    uv = np.column_stack([u, v])
    img = np.zeros((size, size, 3), np.uint8)
    d = P[:, 0]
    dn = np.clip((d - depth_min) / (depth_max - depth_min + 1e-9), 0, 1)
    cols = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_JET).reshape(-1, 3)
    for (uu, vv), c in zip(uv.astype(int), cols):
        cv2.circle(img, (int(uu), int(vv)), 2, (int(c[0]), int(c[1]), int(c[2])), -1)
    return img, uv, idx


def select_polygon(xyzi, cfg):
    """Interactive polygon ROI. Returns a boolean mask over xyzi.
    Requires a display; raises RuntimeError if cancelled (ESC).

    Default view is the top-down BEV (board = a line segment, spatially
    separated from the background wall — easy to lasso). 'v' toggles to the
    camera-like front view. The 'range' trackbar zooms the BEV (half-range)
    / caps the front-view depth; changing range or view clears the clicks."""
    win = 'lidar board ROI'
    size = int(cfg.get('front_view_size', 900))
    state = {'mode': 'bev',
             'range': float(cfg.get('depth_max', 4.0)),
             'dmin': float(cfg.get('depth_min', 0.4))}
    poly = []
    view = {}                                  # img, uv, idx of current render

    def project():
        if state['mode'] == 'bev':
            view['img'], view['uv'], view['idx'] = bev_view(
                xyzi, size, state['range'])
        else:
            view['img'], view['uv'], view['idx'] = front_view(
                xyzi, size, state['dmin'], state['range'])

    def redraw():
        d = view['img'].copy()
        for p in poly:
            cv2.circle(d, p, 4, (255, 255, 255), -1)
        if len(poly) >= 2:
            cv2.polylines(d, [np.array(poly)], False, (255, 255, 255), 1)
        label = ('BEV(top-down, fwd=up)' if state['mode'] == 'bev'
                 else 'FRONT(cam-like)')
        cv2.putText(d, f"{label}  range {state['range']:.1f}m   L-click: add  "
                    "ENTER: done  u: undo  v: view  ESC: skip",
                    (10, size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, d)

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            poly.append((x, y)); redraw()

    def on_range(_=None):
        r = max(0.5, cv2.getTrackbarPos('range [dm]', win) / 10.0)
        if r != state['range']:
            state['range'] = r
            poly.clear()                       # projection changed
            project(); redraw()

    cv2.namedWindow(win)
    cv2.createTrackbar('range [dm]', win, int(state['range'] * 10), 150, on_range)
    cv2.setMouseCallback(win, on_mouse)
    project(); redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(poly) >= 3:      # ENTER
            break
        if k == ord('u') and poly:
            poly.pop(); redraw()
        if k == ord('v'):                          # BEV <-> front
            state['mode'] = 'front' if state['mode'] == 'bev' else 'bev'
            poly.clear()
            project(); redraw()
        if k == 27:                                # ESC
            cv2.destroyWindow(win)
            raise RuntimeError("ROI selection cancelled")
    cv2.destroyWindow(win)

    contour = np.array(poly, np.int32).reshape(-1, 1, 2)
    uv, idx = view['uv'], view['idx']
    inside = np.array([cv2.pointPolygonTest(contour, (float(u), float(v)), False) >= 0
                       for (u, v) in uv])
    mask = np.zeros(len(xyzi), bool)
    mask[idx[inside]] = True
    return mask


# --------------------------------------------------------------------------
# RANSAC plane
# --------------------------------------------------------------------------
def ransac_plane(P, thr=0.02, iters=2000, seed=0):
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return None, None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best_mask, best_cnt = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(n, 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue
        nrm = nrm / norm
        d = -nrm @ a
        inl = np.abs(P @ nrm + d) < thr
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_mask = cnt, inl
    # SVD refit on inliers
    Q = P[best_mask]
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c)
    nrm = Vt[2]
    d = -nrm @ c
    mask = np.abs(P @ nrm + d) < thr
    return nrm, d, mask


# --------------------------------------------------------------------------
# Corner extraction + ordering
# --------------------------------------------------------------------------
def _order_corners(corners3, centroid, normal):
    """Order 4 coplanar 3D corners into C0 bl, C1 br, C2 tr, C3 tl using
    (+Y = left, +Z = up) projected onto the plane.

    Returns (ordered_corners, ok). ok=False when the board is tilted ~45°
    (diamond pose): the gravity-based bl/br/tr/tl labeling is then ambiguous
    and the corners must NOT be used for point-to-point correspondence."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    e_left = Y_LEFT - (Y_LEFT @ n) * n
    e_left /= (np.linalg.norm(e_left) + 1e-12)
    e_up = Z_UP - (Z_UP @ n) * n
    e_up /= (np.linalg.norm(e_up) + 1e-12)
    rel = corners3 - centroid
    u = rel @ e_left        # + = left
    v = rel @ e_up          # + = up
    C0 = int(np.argmax(u - v))    # left,  down
    C1 = int(np.argmax(-u - v))   # right, down
    C2 = int(np.argmax(-u + v))   # right, up
    C3 = int(np.argmax(u + v))    # left,  up
    order = [C0, C1, C2, C3]
    ok = len(set(order)) == 4
    if not ok:                    # degenerate fallback: angular sort (display
        ang = np.arctan2(v, u)    # only; correspondence would be unreliable)
        order = list(np.argsort(ang))
    return corners3[order], ok


def extract_board(xyzi, roi_mask, cfg, board_w, board_h):
    """Return dict with ordered corners in the lidar frame + diagnostics,
    or None on failure."""
    xyz = xyzi[:, :3]
    P = xyz[roi_mask]
    if len(P) < cfg.get('min_plane_inliers', 200):
        return None
    normal, d, inl = ransac_plane(
        P, cfg.get('ransac_dist_thresh', 0.02), int(cfg.get('ransac_iters', 2000)))
    board_pts = P[inl]
    if len(board_pts) < cfg.get('min_plane_inliers', 200):
        return None

    # --- trim coplanar background (e.g. a wall right behind the board): keep
    # only inliers within a board-sized disk around the robust (median) center,
    # then refit the plane on the kept points. Two iterations settle the center.
    r_keep = 0.5 * float(np.hypot(board_w, board_h)) + \
        float(cfg.get('trim_margin', 0.08))
    centroid = board_pts.mean(0)
    _, _, Vt = np.linalg.svd(board_pts - centroid)
    e1, e2 = Vt[0], Vt[1]
    uv = np.column_stack([(board_pts - centroid) @ e1, (board_pts - centroid) @ e2])
    c2 = np.median(uv, axis=0)
    for _ in range(2):
        keep = np.linalg.norm(uv - c2, axis=1) < r_keep
        if keep.sum() < cfg.get('min_plane_inliers', 200):
            break
        c2 = np.median(uv[keep], axis=0)
    keep = np.linalg.norm(uv - c2, axis=1) < r_keep
    if keep.sum() >= cfg.get('min_plane_inliers', 200):
        board_pts = board_pts[keep]
        # refit plane on the trimmed set
        c = board_pts.mean(0)
        _, _, Vt = np.linalg.svd(board_pts - c)
        normal = Vt[2]
        d = -normal @ c
        thr = cfg.get('ransac_dist_thresh', 0.02)
        m2 = np.abs(board_pts @ normal + d) < thr
        if m2.sum() >= cfg.get('min_plane_inliers', 200):
            board_pts = board_pts[m2]

    # make normal point toward the sensor (origin)
    centroid = board_pts.mean(0)
    if normal @ (-centroid) < 0:
        normal = -normal
        d = -d

    # in-plane 2D basis
    _, _, Vt = np.linalg.svd(board_pts - centroid)
    e1, e2 = Vt[0], Vt[1]
    uv = np.column_stack([(board_pts - centroid) @ e1, (board_pts - centroid) @ e2])
    rect = cv2.minAreaRect(uv.astype(np.float32))
    (rw, rh) = rect[1]
    meas_long, meas_short = max(rw, rh), min(rw, rh)

    if cfg.get('fit_known_size', True):
        box_uv = _known_size_box(uv, rect, e1, e2, normal, board_w, board_h)
    else:
        box_uv = cv2.boxPoints(rect)
    corners3 = np.array([centroid + p[0] * e1 + p[1] * e2 for p in box_uv])
    corners3, order_ok = _order_corners(corners3, centroid, normal)

    # --- FOV-clipping check: are the board top/bottom inside the RS-16 FOV? --
    # If the inlier elevation angles touch the sensor limit, the board extends
    # beyond the vertical FOV and the top/bottom corners are NOT observed, so
    # they are unreliable for point-to-point matching.
    elev = np.degrees(np.arctan2(board_pts[:, 2],
                                 np.hypot(board_pts[:, 0], board_pts[:, 1])))
    fov_limit = float(cfg.get('lidar_fov_deg', 15.0))
    margin = float(cfg.get('fov_margin_deg', 0.7))
    clipped = (elev.max() > fov_limit - margin) or (elev.min() < -fov_limit + margin)
    # corners reliable only if not clipped, measured extent ~ known size, and
    # the bl/br/tr/tl labeling was unambiguous (fails at ~45° diamond tilt).
    # extent check is orientation-agnostic (long vs long, short vs short).
    long_b, short_b = max(board_w, board_h), min(board_w, board_h)
    ext_ok = (abs(meas_long - long_b) < 0.15) and (abs(meas_short - short_b) < 0.15)
    corners_reliable = bool((not clipped) and ext_ok and order_ok)

    return dict(corners=corners3, normal=normal, d=float(d), centroid=centroid,
                n_inliers=int(len(board_pts)),
                meas_extent=(float(meas_long), float(meas_short)),
                elev_range=(float(elev.min()), float(elev.max())),
                fov_clipped=bool(clipped), corners_reliable=corners_reliable,
                inlier_pts=board_pts)


def _known_size_box(uv, rect, e1, e2, normal, board_w, board_h):
    """Keep the minAreaRect center+angle but force the extents to the known
    board size, assigning W/H to the axis nearest horizontal/vertical."""
    (cx, cy), (rw, rh), ang = rect
    th = np.deg2rad(ang)
    ax1 = np.array([np.cos(th), np.sin(th)])       # rect axis in (e1,e2)
    ax2 = np.array([-np.sin(th), np.cos(th)])
    # which axis is more "up"? compare their 3D direction to +Z projected
    n = normal / (np.linalg.norm(normal) + 1e-12)
    e_up = Z_UP - (Z_UP @ n) * n
    e_up /= (np.linalg.norm(e_up) + 1e-12)
    up2 = np.array([e_up @ e1, e_up @ e2])
    up2 /= (np.linalg.norm(up2) + 1e-12)
    ax1_is_vert = abs(ax1 @ up2) >= abs(ax2 @ up2)
    # vertical axis -> board_h, horizontal -> board_w
    if ax1_is_vert:
        h1, h2 = board_h / 2.0, board_w / 2.0
    else:
        h1, h2 = board_w / 2.0, board_h / 2.0
    c = np.array([cx, cy])
    return np.array([c - h1 * ax1 - h2 * ax2,
                     c + h1 * ax1 - h2 * ax2,
                     c + h1 * ax1 + h2 * ax2,
                     c - h1 * ax1 + h2 * ax2])
