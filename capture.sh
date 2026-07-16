#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  캡처 전용:  LiDAR + ZED 실시간 동기 캡처 창
#
#   ./capture.sh     캡처 창 실행 (센서는 ./launch.sh 로 미리 띄워둘 것)
#
#  창에서:  [SPACE]/[C] = 그 순간 동기 쌍 저장,  [Q]/[ESC] = 종료
#  저장:    ~/ros2_ws/captures/<날짜_시각>/  (NNNN.png + NNNN.pcd + index.csv)
# ─────────────────────────────────────────────────────────────
set -o pipefail        # 주의: ROS setup.bash 가 미정의 변수를 참조하므로 -u 는 쓰지 않음
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

have_topic() { ros2 topic list --include-hidden-topics 2>/dev/null | grep -qx "$1"; }
CAM=/_zed_hidden/zed/rgb/color/rect/image
LID=/sensors/lidar/points

miss=0
have_topic "$LID" || { echo "✗ $LID 없음"; miss=1; }
have_topic "$CAM" || { echo "✗ ZED 이미지 없음 (enable_ipc:=false 로 떠 있어야 함)"; miss=1; }
if [ "$miss" = 1 ]; then
  echo
  echo "센서가 안 떠 있습니다. 다른 터미널에서 먼저 실행하세요:"
  echo "  ./launch.sh"
  exit 1
fi

echo "▶ 캡처 창 실행:  [SPACE]=저장   [Q]=종료"
exec python3 "$WS/sync_capture.py"
