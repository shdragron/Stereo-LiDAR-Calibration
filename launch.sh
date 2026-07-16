#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  센서 실행:  ros2 launch fsg_sensors sensors.launch.py 위임 래퍼
#
#   ./launch.sh          race 모드 (extrinsic TF 포함, 있으면)
#   ./launch.sh calib    calib 모드 (TF 없이 센서만 — 캘 캡처용)
#   [Ctrl+C]             센서 깔끔히 종료 (카메라 해제)
#
#  토픽:  /sensors/lidar/points , /sensors/camera/{left,right}/{compressed,info}
#  이 창은 열어두고, 다른 터미널에서 ./capture.sh 를 실행하세요.
# ─────────────────────────────────────────────────────────────
set -o pipefail        # 주의: ROS setup.bash 가 미정의 변수를 참조하므로 -u 는 쓰지 않음
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
if [ -f "$WS/install/setup.bash" ]; then
  source "$WS/install/setup.bash"
else
  echo "✗ 먼저 빌드하세요:  cd $WS && colcon build"; exit 1
fi
cd "$WS"    # 기본 extrinsic 경로(calib_debug/…)가 워크스페이스 기준이 되도록

# 카메라는 한 번에 하나만! 이미 떠 있으면 재실행 금지
if pgrep -f 'sensors.launch.py|zed_camera.launch|component_container_isolated' >/dev/null 2>&1; then
  echo "▶ 센서(ZED)가 이미 실행 중 — 중복 실행 방지를 위해 종료합니다."
  echo "  기존 것을 내리려면 해당 터미널에서 Ctrl+C 후 다시 실행하세요."
  exit 1
fi

# tf_change.sh 가 남긴 독립 TF 발행자가 있으면 정리 (launch 가 새로 발행하므로 중복 방지)
pkill -f 'charuco_lidar_calib/tf_publisher' 2>/dev/null || true

MODE="${1:-race}"
echo "▶ fsg_sensors 실행 (mode:=${MODE})  —  종료: [Ctrl+C]"
exec ros2 launch fsg_sensors sensors.launch.py mode:="${MODE}"
