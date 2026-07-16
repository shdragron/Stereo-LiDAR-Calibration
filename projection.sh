#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  라이다 → 이미지 투영 확인:  ./projection.sh [이미지.png ...]
#
#   ./projection.sh captures/20260717_025414/0003_L.png   지정 이미지
#   ./projection.sh /path/to/captures/.../0*.png     절대경로·글롭 OK
#   ./projection.sh                                        최신 세션의 최신 이미지
#
#  같은 이름의 pcd(NNNN.pcd — _L/_R 자동 처리)를 활성 extrinsic 으로
#  이미지에 투영해 calib_debug/fusion_<이름>.png 저장 후 뷰어로 띄운다.
# ─────────────────────────────────────────────────────────────
set -o pipefail        # 주의: ROS setup.bash 가 미정의 변수를 참조하므로 -u 는 쓰지 않음
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

EXTRINSIC="$WS/calib_debug/extrinsic_zed_rslidar.yaml"
K_YAML="$WS/calib_debug/zed_K.yaml"
[ -f "$EXTRINSIC" ] || { echo "✗ 활성 extrinsic 없음 — ./calib.sh 먼저"; exit 1; }
CAMINFO=(); [ -f "$K_YAML" ] && CAMINFO=(--camera-info "$K_YAML")

resolve() {
  if [ -e "$1" ]; then realpath "$1"
  elif [ -e "$WS/$1" ]; then realpath "$WS/$1"
  else echo "$1"; fi
}

# ── 대상 이미지 결정 ───────────────────────────────────────
IMGS=()
if [ $# -ge 1 ]; then
  for a in "$@"; do IMGS+=("$(resolve "$a")"); done
else
  SESS="$(ls -td "$WS"/captures/2026*/ 2>/dev/null | head -1)"
  [ -n "$SESS" ] || { echo "✗ captures/ 에 세션이 없습니다."; exit 1; }
  LATEST="$(ls -t "$SESS"/*.png 2>/dev/null | grep -v '_view\.png\|_R\.png' | head -1)"
  [ -n "$LATEST" ] || { echo "✗ $SESS 에 이미지가 없습니다."; exit 1; }
  IMGS=("$LATEST")
  echo "▶ 이미지 미지정 → 최신 사용: $LATEST"
fi

cd "$WS"
OUTS=()
for img in "${IMGS[@]}"; do
  [ -f "$img" ] || { echo "  ✗ 파일 없음: $img"; continue; }
  base="${img%.png}"; base="${base%_L}"; base="${base%_R}"
  pcd="$base.pcd"
  [ -f "$pcd" ] || { echo "  ✗ 짝 pcd 없음: $pcd"; continue; }
  out="$WS/calib_debug/fusion_$(basename "$base").png"
  ros2 run charuco_lidar_calib verify "$img" --pcd "$pcd" \
       --extrinsic "$EXTRINSIC" "${CAMINFO[@]}" --out "$out" \
    && OUTS+=("$out")
done

[ ${#OUTS[@]} -ge 1 ] || { echo "✗ 생성된 투영 이미지 없음"; exit 1; }
echo "▶ 생성: ${OUTS[*]}"
command -v xdg-open >/dev/null && xdg-open "${OUTS[0]}" >/dev/null 2>&1 &
exit 0
