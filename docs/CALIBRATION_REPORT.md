# ZED2i ↔ RS-16 외부 캘리브레이션 — 작업 정리 보고서

> 2026-07-04 세션. Formula Student Driverless 콘 검출 융합을 위한
> 카메라-라이다 강체변환(R|t) 산출 파이프라인 구축·검증 기록.
> 결과물: ROS2 패키지 `src/charuco_lidar_calib/`, 최종 extrinsic
> `calib_debug/extrinsic_zed_rslidar.yaml` (+ `calib_debug/history/` 이력).

---

## 1. 시스템 구성 (전부 라이브로 실측 확정)

| 항목 | 값 |
|---|---|
| 플랫폼 | Jetson (L4T/tegra), ROS 2 Humble, python cv2 4.5.4 |
| 카메라 | ZED2i · `/zed/zed_node/rgb/color/rect/image` = **좌측 rectified** · bgra8 · 1280×720 |
| 카메라 K | fx=fy=**524.79**, cx=625.76, cy=357.88, **D=0** (rectified) — `grab_camera_info`로 라이브 취득 |
| 스테레오 baseline | **119.75 mm** (우측 camera_info의 P[0,3]=−fx·B에서 자동 추출) |
| 카메라 프레임 | `zed_left_camera_frame_optical` (X우, Y하, Z전방) |
| 라이다 | RoboSense RS-16 · `/rslidar_points` · frame `rslidar` (X전방, Y좌, Z상) · 10 Hz |
| 라이다 포인트 | ~~XYZI~~ → **XYZIRT** (per-point ring+timestamp, 2026-07-04 전환) |
| 보드 | **8×7 ChArUco**, square 0.12 m, marker 0.09 m, **DICT_5X5_50**, legacy 레이아웃, 마커 28개(id 0–27) — `dict_sniffer`로 판별 |
| 캡처 | `./launch.sh`(센서) + `./capture.sh`(SPACE 저장) → `captures/<ts>/NNNN_L.png+_R.png+.pcd` |

레거시 `src/camera_lidar_calibrate`는 순수 ROS1(catkin)이라 빌드 불가 —
수학(Kabsch `calc_RT`)만 발췌해 재구현했고 패키지는 참고용으로만 남김.

---

## 2. 방법 선택과 근거

### 2.1 왜 plane-based(point-to-plane)인가 — RS-16 수직 FOV 제약
RS-16 수직 FOV는 **±15°(링 16개, 2° 간격)**. 거리 d에서 수직 커버리지 = 2d·tan15°.
- 1.1 m에서 **0.61 m**만 커버 ← 보드 세로 0.84 m보다 작음 → **보드 상·하단이
  물리적으로 관측 불가** (증거: `calib_debug/04_lidar_fov_clip.png`)
- 따라서 코너 4점 매칭(레거시 방식)은 근거리에서 불성립.
- **평면 밴드만 쓰는 point-to-plane**은 클리핑과 무관 → 채택.
- 보드 전체가 FOV에 들어오는 원거리 프레임에서만 코너를 보조 구속으로 사용
  (`corners_reliable` 플래그; 클리핑 여부는 inlier 고도각으로 자동 판정).

### 2.2 솔버 구조 (`solve.py`)
1. **회전**: 여러 자세의 보드 법선쌍 (n_lidar, n_cam)에 Kabsch (centroid 빼지 않음)
2. **병진**: point-to-plane 선형 최소제곱 `min_t Σ (n_c·(R·p+t)+d_c)²` — 정규방정식
   `A=Σ|P|·n nᵀ`, 고유값비 = **translation conditioning** (자세 다양성 진단)
3. **조인트 미세정렬**: scipy LM, 평면 잔차 + (신뢰 시) 코너 잔차(가중 5배)
4. 퇴화 처리: 법선이 전부 평행(rank<2)이면 신뢰 코너로 6DOF 폴백, 코너도 없으면 에러

### 2.3 카메라 쪽 — mono PnP → 스테레오 epipolar 삼각측량
- **mono**: ChArUco 내부 코너(최대 42개) → `estimatePoseCharucoBoard`. 실측 reproj 0.13–0.78 px.
- **스테레오**(최종): 좌/우 rectified에서 코너를 id 매칭 → row 일치로 epipolar 검증
  (실측 RMS 0.2–0.4 px) → disparity 삼각측량으로 **코너별 독립 메트릭 3D** →
  보드 레이아웃에 Kabsch 피팅 → pose·외곽코너·평면.
- 원거리 소형 마커 대응: **멀티스케일 검출**(2배 업스케일 재시도) — 2 m에서 매칭 11→36개.

### 2.4 발견: ZED 공장 rectification의 disparity 편향 **+1.75 px**
스테레오 깊이가 라이다·mono보다 거리비례 ~6% 짧게 나오는 것을 held-out에서 발견.
- cx_L=cx_R 확인 → 공식 문제 아님. **라이다(ToF)가 mono 편을 들어줌** → 스테레오 편향.
- 원인: rectification 잔여 toe-in (yaw ~0.19° ≙ 일정한 disparity 오프셋).
- **해법**: 보드가 크기를 아는 절대 자(ruler)임을 이용 — 삼각측량 그리드를 보드
  레이아웃에 rigid-fit할 때 **fit RMSE를 최소화하는 δd를 1D 최적화**로 자체 추정
  (mono 불신뢰 가정 불필요). 프레임별 추정 → **리그 전역 median = +1.75 px**.
- 효과: 스테레오 fit 21→**3.9 mm**, held-out 평면 오프셋 **190 mm → ~10 mm**.

---

## 3. 만든 것

### 3.1 신규 패키지 `src/charuco_lidar_calib/` (ament_python)

| 실행파일 | 역할 |
|---|---|
| `calibrate` | 캡처 폴더(모노/스테레오 자동 인식) → extrinsic yaml + static_tf 명령. 옵션: `--roi {interactive,auto}`, `--camera-info`, `--min-frames`, `--trust-corners`(테스트용), `--no-corners`, `--no-refine` |
| `verify` | 라이다→이미지 재투영 오버레이. `--color {depth,intensity}`, `--zoom-board`(보드 크롭 2배) |
| `colorize` | 이미지 RGB를 라이다 클라우드에 입힘 → XYZRGB pcd (+`--views` 렌더) |
| `tf_publisher` | extrinsic yaml → static TF 발행 (+launch 파일) |
| `dict_sniffer` | 보드 이미지/토픽에서 ArUco 딕셔너리 판별 |
| `grab_camera_info` | 라이브 K/D/baseline → yaml |

모듈: `board.py`(보드 모델·양세대 API 호환·멀티스케일 검출), `charuco_pose.py`(mono),
`stereo_pose.py`(삼각측량+δd), `lidar_board.py`(ROI→RANSAC 평면→배경 트리밍→
known-size 코너→정렬·FOV클리핑 판정), `solve.py`(솔버), `pcd_io.py`(PCD 파서),
`calibrate.py`(파이프라인), `verify.py`/`colorize.py`(검증 시각화).

### 3.2 기존 파일 수정
- **`sync_capture.py`**: 우측 이미지 구독 추가 → right 토픽이 살아있으면
  `NNNN_L/_R.png` 스테레오 저장(스탬프 매칭은 최근 60프레임 버퍼 — 최신 1장만으론
  싱크로나이저 지연 때문에 매칭 실패했던 버그 수정). 창에 `stereo(R): ON/OFF` 표시.
- **`launch.sh`**: ZED 실행에 `ros_params_override_path:=zed_lr_override.yaml`
  (`video.publish_left_right: true`) 추가 → 좌/우 rectified 토픽 활성.
- **`src/rslidar_sdk/CMakeLists.txt:8`**: `POINT_TYPE XYZI → XYZIRT`
  (per-point ring+timestamp; deskew용. 드라이버 CPU +무시가능(5.9%), 10.0 Hz 유지 확인)

### 3.3 라이다 보드 추출 세부 (`lidar_board.py`)
- 정면뷰(깊이 컬러) 투영 → 폴리곤 ROI(인터랙티브) 또는 카메라 pose 기반 자동 박스
- RANSAC 평면 → **배경 트리밍**: 보드 크기 반경의 median-중심 디스크로 인라이어
  한정 후 재피팅 (보드 뒤 벽이 준-동일평면일 때 extent 1.46→1.06 m로 개선)
- 코너: minAreaRect의 중심·방향만 취하고 **알려진 체커 크기(0.96×0.84)를 강제**
  → 엣지 번짐(빔 발산)·흰 여백이 대칭이면 코너 불편향
- 정렬: 평면에 투영한 (+Y=좌, +Z=상) 기준 C0(좌하)~C3(좌상) — 카메라 순서와 일치 검증됨

---

## 4. 캘리브레이션 이력

| 세션 | 방식 | 자세 | plane RMSE | normal RMS | conditioning | t (cm) |
|---|---|---|---|---|---|---|
| 192905 | mono, 핸드헬드 | 7 | 9.2 mm | 1.26° | 31.8 GOOD | (5.2, 7.2, −2.0) |
| 194426 | mono, 핸드헬드 | 8 | 9.0 mm | 0.90° | 52.7 WEAK | (4.8, 4.6, −1.7) |
| **200743** | **스테레오, pitch 포함** | **10** | **8.9 mm** | **0.667°** | **9.1 GOOD** | **(4.3, 4.3, −3.5)** |

교훈: 세워진 보드만 찍으면 법선이 수평에 몰려 **수직 병진이 미결정**
(mono 두 세션 간 수직축 25 mm 차이). **pitch(뒤로 젖힘) 자세**가 들어오자
conditioning 52.7→9.1, 수직축 재현 3 mm.

### 최종 extrinsic (lidar → camera optical, 2026-07-04 20:xx)
```
t = (0.0433, 0.0430, −0.0351) m
q = (0.4899, −0.5049, 0.5108, 0.4942) (xyzw)
ros2 run tf2_ros static_transform_publisher --x 0.043342 --y 0.042988 --z -0.035108 \
  --qx 0.489852 --qy -0.504934 --qz 0.510759 --qw 0.494178 \
  --frame-id zed_left_camera_frame_optical --child-frame-id rslidar
```
파일: `calib_debug/extrinsic_zed_rslidar.yaml` ·
이력: `calib_debug/history/extrinsic_20260704_*_10pose_stereo.yaml`

---

## 5. 검증 증거

| 검증 | 결과 |
|---|---|
| 솔버 합성 GT | 회전 0.0001°, 병진 0.00 mm 복원 (노이즈 5 mm, 6자세) |
| 스테레오 합성 | 42/42 매칭, epipolar RMS 0.09 px |
| 카메라 실측 | 42/42 코너, reproj 0.13–0.78 px |
| held-out 평면 (솔브 미사용 캡처) | 오프셋 **+12/+10 mm**, 각도 0.5–2.7° |
| **split-half 재현성** | 반쪽(4자세) 간 0.371°/31.5 mm → **풀 솔브 추정 ≈0.17°/14 mm → FSD 게이트(0.3°/2-3 cm) 통과 추정** |
| intensity 융합 | 흰/검 체커 패턴이 라이다 반사강도로 이미지 체커와 정렬 (`10_`,`11_`,`19_*_zoom.png`) |
| 신규 장면 투영 | 인물·패널 실루엣과 깊이 경계 픽셀단위 일치 (`20_`,`21_*.png`) |
| RGB 클라우드 | 파란 생수통 색이 정확한 3D 점에 입혀짐 (`13_colored_cloud_front.png`) |

시각 증거 전체: `calib_debug/01…21_*.png` (01 카메라 pose 검증, 04 FOV 클리핑 증명,
05 명목 마운트, 08/09 depth 융합, 10/11/19 intensity 융합, 12 재캘, 13–18 RGB 클라우드,
20/21 신규 장면).

## 6. 시간동기 실측 (주행 융합 대비)

- 클럭: 양쪽 host clock 통일 — 페어링 오프셋 **mean +3.1 ms** (스큐 없음)
- 페어링 산포: std 26 ms, worst ±59 ms (라이다 10 Hz 최근접 페어링의 구조적 한계)
- 주행 15 m/s 환산: 평균 4.7 cm, **worst 89 cm** → 정지 캘리브레이션엔 무해,
  **주행 융합에는 deskew 필수**
- 수신지연: 라이다 6.6 ms, 카메라 55 ms (ZED 처리시간; 스탬프는 grab 시점이라 무해)
- 대비책 완료: XYZIRT 전환으로 per-point timestamp 확보 → ZED IMU/odom 기반
  deskew 노드 구현 가능 상태

## 7. 운영 규칙 (FSG)

1. **이 yaml이 아니라 이 워크플로우가 자산** — 실차 장착 후 같은 절차로 재캘.
   실차에선 센서 간격이 커지므로 **`--roi interactive`** 사용 (auto는 명목 회전만
   가정, translation 무시 — 벤치 전용).
2. 캡처 규칙: 보드 **거치 고정·완전 정지 후 SPACE**, 거리 ~2 m
   (720p 검출 상한 ~2.5 m / FOV 하한 1.6 m), **6–10자세 + pitch 2–3장 필수**.
3. 리마운트마다 **2분 verify 게이트**: 2–3자세 → 저장된 yaml과 비교, 게이트 0.3°/2 cm.
   실패 시에만 풀 재캘. 결과는 항상 `calib_debug/history/`에 날짜 붙여 보존.
4. 품질 지표 읽기: `conditioning < 50` (자세 다양성), per-pose `plane_mm`,
   스테레오 `dd`(≈+1.75 px에서 크게 벗어나면 rectification 이상 의심),
   intensity 오버레이로 최종 눈 검증.

## 8. 남은 작업

- [ ] 실차 장착 후 재캘 + 같은 방식 2세션 재현성으로 게이트 최종 판정
- [ ] deskew(모션 보상) 노드 — XYZIRT timestamp + ZED IMU/odom (주행 융합 필수)
- [ ] 라이브 `/colored_points` 퍼블리셔 (콘 디버깅용, 선택)
- [ ] `--roi auto`에 직전 extrinsic 시드 옵션 (탈부착 워크플로우 편의, 선택)

## 9. 트러블슈팅 기록 (시행착오 요약)

| 증상 | 원인 | 해법 |
|---|---|---|
| 3 m에서 ChArUco 검출 실패 | 720p에서 마커 16 px | 거리 ~2 m + 멀티스케일 검출 |
| 라이다 보드 폭 1.46 m 과대 | 보드 뒤 벽이 준-동일평면 | median-디스크 트리밍 후 재피팅 |
| 솔브 발산 (t~20 cm, 59 mm RMSE) | 자세 2개가 법선 3°차뿐 | 자세 다양성 + conditioning 경고로 검출 |
| 스테레오 깊이 6% 짧음 | rectification toe-in (+1.75 px) | 보드-자 기반 δd 자체추정 (리그 median) |
| capture.sh가 mono만 저장 | 최신 1장 캐시 vs 싱크 지연 | 우측 60프레임 버퍼 스탬프 매칭 |
| 수직 병진 25 mm 흔들림 | 법선이 전부 수평 | pitch 자세 추가 (conditioning 9.1) |
| "지연" 체감 | rviz2+rqt CPU 점유(load 3.1) | 시각화 도구는 볼 때만; 센서는 10 Hz 정상 |
