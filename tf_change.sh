#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  활성 extrinsic(TF) 교체:  ./tf_change.sh [세션폴더 | yaml파일]
#
#   ./tf_change.sh captures/20260717_025414   그 세션의 캘 결과로 교체
#   ./tf_change.sh src/fsg_sensors/config/extrinsics/history/2026-07-04_*.yaml
#   ./tf_change.sh                            인자 없음 = 현재 활성 yaml 그대로 재발행
#
#  하는 일:
#   ① 지정한 extrinsic yaml을 활성 파일(calib_debug/extrinsic_zed_rslidar.yaml)로 복사
#   ② 떠 있는 TF 발행자(launch가 띄운 것 포함)를 내리고 새 값으로 다시 발행
#      (센서 launch 는 건드리지 않음 — TF만 교체됨)
#  ※ 다음 ./launch.sh 도 활성 파일을 읽으므로 재부팅/재기동 후에도 유지된다.
# ─────────────────────────────────────────────────────────────
set -o pipefail        # 주의: ROS setup.bash 가 미정의 변수를 참조하므로 -u 는 쓰지 않음
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

ACTIVE="$WS/calib_debug/extrinsic_zed_rslidar.yaml"

resolve() {
  if [ -e "$1" ]; then realpath "$1"
  elif [ -e "$WS/$1" ]; then realpath "$WS/$1"
  else echo "$1"; fi
}

# ── 소스 yaml 결정 ─────────────────────────────────────────
if [ $# -ge 1 ]; then
  SRC="$(resolve "$1")"
  if [ -d "$SRC" ]; then
    if   [ -f "$SRC/extrinsic.yaml" ]; then SRC="$SRC/extrinsic.yaml"
    elif [ -f "$SRC/extrinsic_zed_rslidar.yaml" ]; then SRC="$SRC/extrinsic_zed_rslidar.yaml"
    else echo "✗ $SRC 안에 extrinsic.yaml 이 없습니다 (그 세션으로 ./calib.sh 먼저)."; exit 1
    fi
  fi
  [ -f "$SRC" ] || { echo "✗ 파일 없음: $SRC"; exit 1; }
  [ "$SRC" -ef "$ACTIVE" ] || cp "$SRC" "$ACTIVE"
  echo "▶ 활성 extrinsic 교체: $SRC"
else
  [ -f "$ACTIVE" ] || { echo "✗ 활성 extrinsic 없음: $ACTIVE — ./calib.sh 먼저"; exit 1; }
  echo "▶ 인자 없음 → 현재 활성 yaml 재발행: $ACTIVE"
fi

# ── 발행 교체: 기존 TF 발행자 전부 내리고 하나만 새로 ──────
pkill -f 'charuco_lidar_calib/tf_publisher' 2>/dev/null || true
nohup setsid ros2 run charuco_lidar_calib tf_publisher \
      --ros-args -p extrinsic:="$ACTIVE" \
      > /tmp/tf_publisher_$(date +%s).log 2>&1 &
disown

python3 - "$ACTIVE" <<'EOF'
import sys, yaml
e = yaml.safe_load(open(sys.argv[1]))
t = e['lidar_to_camera']['t']; m = e.get('metrics', {})
print(f"▶ 발행됨: {e['lidar_to_camera']['parent_frame']} -> {e['lidar_to_camera']['child_frame']}")
print(f"  t = ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m   metrics: plane={m.get('rmse_plane_mm')}mm poses={m.get('n_poses')}")
EOF
echo "▶ 확인:  ros2 run tf2_ros tf2_echo zed_left_camera_frame_optical rslidar"
