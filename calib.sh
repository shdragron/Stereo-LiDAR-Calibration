#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  캘리브레이션 원커맨드:  ./calib.sh [캡처경로]
#
#   ./calib.sh                          captures/ 최신 세션으로 실행
#   ./calib.sh captures/20260717_0254*  세션 지정 (glob 가능)
#
#  하는 일:
#   ① 카메라 인트린식 확보 (센서 떠 있으면 라이브로 새로 grab,
#      아니면 기존 calib_debug/zed_K.yaml 사용)
#   ② calibrate 실행 — 캡처 때 그린 ROI(NNNN_roi.npy)가 있으면 창 없이
#      끝까지 자동, 없는 프레임만 ROI 창(BEV)이 뜸
#   ③ 성공 시 결과를 fsg_sensors/config/extrinsics/history/에 날짜로 보관
#
#  다음 단계:  Ctrl+C로 calib 모드 내리고  ./launch.sh  (race, TF 자동 발행)
# ─────────────────────────────────────────────────────────────
set -o pipefail        # 주의: ROS setup.bash 가 미정의 변수를 참조하므로 -u 는 쓰지 않음
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

K_YAML="$WS/calib_debug/zed_K.yaml"
EXTRINSIC="$WS/calib_debug/extrinsic_zed_rslidar.yaml"
HISTORY="$WS/src/fsg_sensors/config/extrinsics/history"

# ── 세션 결정 (절대/상대 경로 모두 지원, 어느 디렉토리에서 실행해도 됨) ──
resolve() {
  # 그대로 존재하면 절대경로화 → 아니면 $WS 기준으로 시도 → 글롭 등은 원본 유지
  if [ -e "$1" ]; then realpath "$1"
  elif [ -e "$WS/$1" ]; then realpath "$WS/$1"
  else echo "$1"; fi
}
if [ $# -ge 1 ]; then
  SESS=()
  for a in "$@"; do SESS+=("$(resolve "$a")"); done
else
  SESS=("$(ls -td "$WS"/captures/2026*/ 2>/dev/null | head -1)")
  [ -n "${SESS[0]}" ] || { echo "✗ captures/ 에 세션이 없습니다. 먼저 ./capture.sh"; exit 1; }
  echo "▶ 캡처 경로 미지정 → 최신 세션 사용: ${SESS[0]}"
fi

# calibrate 의 기본 출력/디버그 경로(calib_debug/…)가 항상 워크스페이스에
# 떨어지도록 고정 — 다른 디렉토리에서 실행해도 결과 위치가 같다.
cd "$WS"

# ── ① 인트린식 ─────────────────────────────────────────────
echo "▶ 카메라 인트린식 확보 중…"
if ros2 run charuco_lidar_calib grab_camera_info --out "$K_YAML" --timeout 6 >/dev/null 2>&1; then
  echo "  ✓ 라이브 grab 성공: $K_YAML"
elif [ -f "$K_YAML" ]; then
  echo "  ⚠ 센서가 안 떠 있어 기존 파일 사용: $K_YAML"
  echo "    (카메라를 교체/재보정했다면 센서 켜고 다시 실행할 것)"
else
  echo "✗ 인트린식 없음: 센서를 켜거나($K_YAML 생성) 다시 실행하세요."; exit 1
fi

# ── ② 솔브 ────────────────────────────────────────────────
echo "▶ calibrate 실행: ${SESS[*]}"
ros2 run charuco_lidar_calib calibrate "${SESS[@]}" --camera-info "$K_YAML" || {
  echo "✗ 캘리브레이션 실패 — 위 로그의 WEAK/skip 항목을 확인하세요."; exit 1; }

# ── ③ 이력 보관 + 세션 폴더 사본 ──────────────────────────
mkdir -p "$HISTORY"
TAG="$(date +%Y-%m-%d_%H%M)"
cp "$EXTRINSIC" "$HISTORY/${TAG}.yaml"
for s in "${SESS[@]}"; do             # 세션 폴더에도 사본 → ./tf_change.sh <세션> 가능
  d="$s"; [ -d "$d" ] || d="$(dirname "$d")"
  if [ -d "$d" ] && [ "$d" != "$WS" ]; then
    cp "$EXTRINSIC" "$d/extrinsic.yaml"
    echo "▶ 세션 사본: $d/extrinsic.yaml"
    break
  fi
done
echo
echo "▶ 완료. 결과: $EXTRINSIC"
echo "▶ 이력 보관: $HISTORY/${TAG}.yaml"
echo "▶ 적용: ./tf_change.sh (즉시 TF 교체)  또는  ./launch.sh 재기동(race 모드)"
