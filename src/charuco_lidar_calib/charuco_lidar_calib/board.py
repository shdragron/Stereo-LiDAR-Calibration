"""ChArUco board model + OpenCV-version-agnostic helpers.

Verified on this rig with OpenCV (python) 4.5.4:
  8x7 squares, square=0.12 m, DICT_5X5_50, legacy layout, 28 markers (id 0..27).

The ChArUco *pose* is recovered from the interior chessboard corners, whose 3D
positions depend only on ``square_length`` (not ``marker_length``), so the pose
is accurate even if ``marker_length`` is a rough guess.
"""
from dataclasses import dataclass
import numpy as np
import cv2
import cv2.aruco as aruco


# --- dictionary name -> enum, tolerant to both API generations -------------
def resolve_dictionary(name):
    """Return a cv2.aruco dictionary object for a name like 'DICT_5X5_50'."""
    if not hasattr(aruco, name):
        raise ValueError(f"Unknown ArUco dictionary '{name}'")
    enum = getattr(aruco, name)
    if hasattr(aruco, 'getPredefinedDictionary'):        # 4.7+
        return aruco.getPredefinedDictionary(enum)
    return aruco.Dictionary_get(enum)                     # 4.5.x


@dataclass
class BoardModel:
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary: str
    legacy: bool = True
    min_charuco_corners: int = 8

    def __post_init__(self):
        self._dict = resolve_dictionary(self.dictionary)
        self._board = self._make_board()

    # --- board object, tolerant to both API generations -------------------
    def _make_board(self):
        sx, sy = self.squares_x, self.squares_y
        sq, mk = self.square_length, self.marker_length
        if hasattr(aruco, 'CharucoBoard_create'):            # 4.5.x (legacy)
            return aruco.CharucoBoard_create(sx, sy, sq, mk, self._dict)
        # 4.7+ : (cols, rows) == (squares_x, squares_y)
        board = aruco.CharucoBoard((sx, sy), sq, mk, self._dict)
        if hasattr(board, 'setLegacyPattern'):
            board.setLegacyPattern(self.legacy)
        return board

    @property
    def cv_board(self):
        return self._board

    @property
    def cv_dict(self):
        return self._dict

    @property
    def n_interior_corners(self):
        return (self.squares_x - 1) * (self.squares_y - 1)

    def outer_corners_board_frame(self):
        """The 4 outer corners of the *checker region* in the board frame.

        Order (validated against the camera image):
            C0 = (0,0)          bottom-left
            C1 = (W,0)          bottom-right
            C2 = (W,H)          top-right
            C3 = (0,H)          top-left
        with W = squares_x*square_length, H = squares_y*square_length, z = 0.
        """
        w = self.squares_x * self.square_length
        h = self.squares_y * self.square_length
        return np.array([[0, 0, 0],
                         [w, 0, 0],
                         [w, h, 0],
                         [0, h, 0]], dtype=np.float64)


# --- detection helpers, tolerant to both API generations -------------------
def detect_markers(gray, board: BoardModel):
    d = board.cv_dict
    if hasattr(aruco, 'ArucoDetector'):                      # 4.7+
        det = aruco.ArucoDetector(d, aruco.DetectorParameters())
        corners, ids, rejected = det.detectMarkers(gray)
    else:                                                    # 4.5.x
        corners, ids, rejected = aruco.detectMarkers(gray, d)
    return corners, ids


def interpolate_charuco(corners, ids, gray, board: BoardModel, K=None, D=None):
    """Return (n_found, charuco_corners, charuco_ids)."""
    if ids is None or len(ids) == 0:
        return 0, None, None
    b = board.cv_board
    if hasattr(aruco, 'CharucoDetector'):                    # 4.7+
        det = aruco.CharucoDetector(b)
        ch_corners, ch_ids, _, _ = det.detectBoard(gray)
        n = 0 if ch_ids is None else len(ch_ids)
        return n, ch_corners, ch_ids
    # 4.5.x
    n, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
        corners, ids, gray, b, cameraMatrix=K, distCoeffs=D)
    return (n if n is not None else 0), ch_corners, ch_ids


def chessboard_obj_points(board: BoardModel):
    """(K,3) board-frame coords of the interior corners, indexed by charuco id.
    Handles both OpenCV API generations."""
    b = board.cv_board
    if callable(getattr(b, 'getChessboardCorners', None)):     # 4.7+
        return np.asarray(b.getChessboardCorners(), np.float64).reshape(-1, 3)
    return np.asarray(b.chessboardCorners, np.float64).reshape(-1, 3)  # 4.5.x


def detect_charuco_multiscale(gray, board: BoardModel, K=None, D=None,
                              scales=(1, 2), good_enough=30):
    """Detect interior ChArUco corners, retrying on upscaled images for
    small/far boards. Returns (n, corners, ids) with corner coordinates in
    the ORIGINAL image scale."""
    import cv2 as _cv2
    best = (0, None, None)
    h, w = gray.shape[:2]
    for s in scales:
        g = gray if s == 1 else _cv2.resize(
            gray, (w * s, h * s), interpolation=_cv2.INTER_CUBIC)
        # K scales with the image for the optional-refinement path
        Ks = None
        if K is not None and s != 1:
            Ks = np.asarray(K, np.float64).copy()
            Ks[:2, :] *= s
        elif K is not None:
            Ks = np.asarray(K, np.float64)
        corners, ids = detect_markers(g, board)
        n, chc, chi = interpolate_charuco(corners, ids, g, board, Ks, D)
        if n > best[0]:
            chc_scaled = None if chc is None else (chc / float(s))
            best = (n, chc_scaled, chi)
        if best[0] >= good_enough:
            break
    return best


def estimate_board_pose(ch_corners, ch_ids, board: BoardModel, K, D):
    """Return (ok, rvec, tvec) for board->camera, or (False, None, None)."""
    if ch_ids is None or len(ch_ids) < board.min_charuco_corners:
        return False, None, None
    b = board.cv_board
    if hasattr(aruco, 'estimatePoseCharucoBoard'):           # 4.5.x
        ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, b, K, D, None, None)
        return bool(ok), rvec, tvec
    # 4.7+ : match interpolated corners to object points and solvePnP
    obj_pts, img_pts = b.matchImagePoints(ch_corners, ch_ids)
    if obj_pts is None or len(obj_pts) < board.min_charuco_corners:
        return False, None, None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)
    return bool(ok), rvec, tvec
