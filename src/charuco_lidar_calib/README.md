# charuco_lidar_calib

Extrinsic calibration between a **ZED2i** camera and a **RoboSense RS-16** LiDAR
using an **8×7 ChArUco board** (ROS 2 Humble, Python / rclpy).

ChArUco gives a robust, metric camera-side board pose (sub-pixel). The LiDAR
board is extracted semi-automatically (ROI click → RANSAC plane). A multi-pose
**plane-based** solve (normal-Kabsch rotation + point-to-plane translation +
joint refine) yields the rigid transform `lidar → camera`.

## Why plane-based (important)

The RS-16 has only a **±15° (30°) vertical FOV**. At range *d* it can see a
board slice only `2·d·tan15°` tall — e.g. **0.61 m at 1.1 m**. Our board is
**0.84 m** tall, so at close range the board top/bottom fall **outside** the
LiDAR FOV and their corners are *never observed* (see
`calib_debug/04_lidar_fov_clip.png`). Corner-to-corner matching is therefore
unreliable. The plane-based solver only needs the board **plane band** the
LiDAR *does* see, so it is robust to this clipping. When the board is far
enough that it fully fits the FOV, the 4 corners are additionally used
(`corners_reliable`).

## Verified rig facts (2026-07-04)

| | |
|---|---|
| camera image | `/sensors/zed/rgb/color/rect/image` · **bgra8** · 1280×720 (left, rectified) |
| camera_info | `/sensors/zed/rgb/color/rect/camera_info` · **grab live — camera swapped to ZED gen1 2026-07-16 (fx≈676.58)** · **D=0** |
| camera frame | `zed_left_camera_frame_optical` |
| lidar | `/sensors/lidar/points` · fields x y z intensity ring timestamp · frame `rslidar` |
| board | 8×7 · square **0.12 m** · **DICT_5X5_50** · legacy · 28 markers (id 0–27) |

Edit `config/calib.yaml` if any of these change.

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select charuco_lidar_calib
source install/setup.bash
```

## Data collection (do this first)

Use the existing `./capture.sh` (SPACE saves a synchronized pair). With the
right image topic enabled (`./launch.sh` does this via `zed_lr_override.yaml`)
it saves **stereo** `NNNN_L.png + NNNN_R.png + NNNN.pcd`; otherwise mono
`NNNN.png + NNNN.pcd`. The capture window shows `stereo(R): ON/OFF`.
`calibrate` auto-detects either layout — stereo frames use epipolar
triangulation (more accurate depth), mono frames fall back to PnP.

**Capture 6–10 board poses on a stand (not handheld), fully still before SPACE:**

- distance ≈ 2 m (detection limit ~2.5 m at 720p; full-board LiDAR FOV needs ≥1.6 m),
- frontal, tilted left/right (yaw), **and 2–3 poses pitched back — required**
  (all-upright boards leave the vertical translation unconstrained),
- keep the whole board inside the camera view.

Rotation/translation observability comes from **orientation variety** — moving
the board around without tilting it is not enough. Watch `translation
conditioning` in the output (<50 = good).

Note: stereo frames also self-calibrate a rig-wide **disparity offset**
(ZED factory rectification toe-in; ~+1.75 px on this rig) using the board's
known size as a metric ruler — reported as `dd=` per frame.

## Calibrate

```bash
# (optional) refresh camera intrinsics from the live topic
ros2 run charuco_lidar_calib grab_camera_info --out calib_debug/zed_K.yaml

# solve — interactive ROI: for each frame, click a polygon around the board
ros2 run charuco_lidar_calib calibrate captures/<ts> \
     --roi interactive --camera-info calib_debug/zed_K.yaml

# board isolated / headless: auto ROI from the camera board pose
ros2 run charuco_lidar_calib calibrate captures/<ts> --roi auto
```

Output: `calib_debug/extrinsic_zed_rslidar.yaml` (R, t, quaternion, 4×4, both
directions, metrics) and a ready `static_transform_publisher` command. Per-pose
debug images `pair_NNN_cam.png` / `pair_NNN_lidar.png` land in `calib_debug/`.

Watch the metrics: `plane RMSE` (mm), `normal RMS` (deg), and **translation
conditioning** — if it is high (WEAK), add more orientation variety.

## Verify (fusion overlay)

```bash
# depth-colored overlay
ros2 run charuco_lidar_calib verify captures/<ts>/0000_L.png \
     --extrinsic calib_debug/extrinsic_zed_rslidar.yaml
# intensity-colored + 2x board crop — the sharpest visual check: the board's
# white/black checkers appear in the lidar reflectivity, aligned with the image
ros2 run charuco_lidar_calib verify captures/<ts>/0000_L.png \
     --extrinsic calib_debug/extrinsic_zed_rslidar.yaml --color intensity --zoom-board
```

## Colorize (paint the image onto the cloud)

```bash
ros2 run charuco_lidar_calib colorize captures/<ts>/0000_L.png \
     --extrinsic calib_debug/extrinsic_zed_rslidar.yaml --views
# -> calib_debug/colored_cloud.pcd (XYZRGB; open with pcl_viewer/CloudCompare)
```

## Publish the extrinsic as a TF

```bash
ros2 launch charuco_lidar_calib tf_publisher.launch.py \
     extrinsic:=$PWD/calib_debug/extrinsic_zed_rslidar.yaml
# or paste the static_transform_publisher line printed by `calibrate`
```

## Identify the board dictionary (utility)

```bash
ros2 run charuco_lidar_calib dict_sniffer --image captures/<ts>/0000.png
ros2 run charuco_lidar_calib dict_sniffer --topic /zed/zed_node/rgb/color/rect/image
```

## Executables

| exe | purpose |
|---|---|
| `calibrate` | offline solve over captured (mono or stereo) png+pcd pairs |
| `verify` | reproject lidar onto the image (`--color depth\|intensity`, `--zoom-board`) |
| `colorize` | paint the camera image onto the cloud → XYZRGB pcd (`--views`) |
| `tf_publisher` | publish the extrinsic as a static TF |
| `dict_sniffer` | identify the ArUco dictionary of a board |
| `grab_camera_info` | dump live CameraInfo (K/D + stereo baseline) to yaml |

Full session log, calibration history and troubleshooting notes:
[`~/ros2_ws/CALIBRATION_REPORT.md`](../../CALIBRATION_REPORT.md)

## Notes

- All debug visualizations go to `calib_debug/` (workspace root).
- ChArUco pose depends only on `square_length`, so `marker_length` may be a
  rough guess without hurting accuracy.
- The camera-side geometry (C0 bl, C1 br, C2 tr, C3 tl outer corners) was
  validated against real images — see `calib_debug/01_charuco_pose_camera.png`.
