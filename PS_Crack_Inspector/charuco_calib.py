"""
ChArUco 카메라 내부 파라미터 캘리브레이션 (OV9281, 1280x800)

보드: 11x7칸, 칸 6.0mm, 마커 4.5mm, DICT_4X4_50  (charuco_11x7_6mm_A4.pdf)

[촬영 모드]  python charuco_calib.py
  SPACE : 현재 프레임 저장 (검출 코너가 MIN_CORNERS 이상일 때만)
  P     : 스테이지 평면 이미지 저장 (보드를 스테이지 바닥에 평평하게 놓고 1장)
  [ / ] : 노출 -1 / +1
  C     : 저장된 이미지로 캘리브레이션 실행
  ESC   : 종료

[오프라인 모드]  python charuco_calib.py --offline
  calib_images/ 폴더의 이미지로 캘리브레이션만 다시 실행
  (오차 큰 이미지를 지운 뒤 다시 돌릴 때 사용)

결과: camera_intrinsics.npz / camera_intrinsics.yaml
      (P 이미지가 있으면 스테이지 평면 위치도 함께 저장)
"""
import os
import sys
import glob
import time
import cv2
import numpy as np

# ────────────────────────────────────────────────────────────
# [설정]
# ────────────────────────────────────────────────────────────
CAMERA_INDEX   = 1
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 800
CAM_EXPOSURE   = -7          # 보드가 잘 보이게 조절 ([ ] 키로 실시간 조절 가능)
CAM_GAIN       = 0

# 조명: 아두이노로 LED 전체 ON (안 쓰면 SERIAL_PORT = None, 실내등 사용)
SERIAL_PORT    = 'COM14'
BAUD_RATE      = 115200
ALL_ON_CMD     = b'9'
ALL_OFF_CMD    = b'0'

# 보드 사양 (인쇄 실측 기준)
SQUARES_X      = 11
SQUARES_Y      = 7
SQUARE_MM      = 6.0
MARKER_MM      = 4.5
BOARD_THICKNESS_MM = 1.3     # 평면 이미지(P): 보드 받침판+종이 두께. 스테이지 바닥 위치 보정용

MIN_CORNERS    = 15          # 저장 최소 코너 수 (전체 60개)
FIX_K3         = True        # 좁은 화각 렌즈면 True 권장
ZERO_TANGENT   = True        # True: 접선왜곡 p1=p2=0 고정 (--check 결과 보고 결정)
FIX_K2         = True        # True: k2=0 고정, k1만 사용 (--check 결과 보고 결정)

base_dir   = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)   # 실행 파일(exe)이면 exe 옆에 저장
            else os.path.dirname(os.path.abspath(__file__)))
img_dir    = os.path.join(base_dir, "calib_images")
plane_path = os.path.join(base_dir, "calib_plane.png")
out_npz    = os.path.join(base_dir, "camera_intrinsics.npz")
out_yaml   = os.path.join(base_dir, "camera_intrinsics.yaml")


# ════════════════════════════════════════════════════════════
# 보드 / 검출기 (OpenCV 4.6 이하, 4.7+ 모두 지원)
# ════════════════════════════════════════════════════════════

def make_board():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    try:
        board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, dictionary)
    except Exception:
        board = aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_MM, MARKER_MM, dictionary)

    if hasattr(aruco, "CharucoDetector"):            # 4.7+
        det = aruco.CharucoDetector(board)

        def detect(gray):
            cc, ci, _, _ = det.detectBoard(gray)
            return cc, ci
    else:                                             # 4.6 이하
        params = aruco.DetectorParameters_create()

        def detect(gray):
            mc, mi, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
            if mi is None or len(mi) == 0:
                return None, None
            _, cc, ci = aruco.interpolateCornersCharuco(mc, mi, gray, board)
            return cc, ci

    if hasattr(board, "getChessboardCorners"):
        chess = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    else:
        chess = np.asarray(board.chessboardCorners, dtype=np.float32)
    return board, detect, chess


BOARD, DETECT, CHESS_3D = make_board()


def detect_points(gray):
    """(이미지 코너 Nx2, 보드 3D 좌표 Nx3 [mm]) 또는 (None, None)"""
    cc, ci = DETECT(gray)
    if cc is None or ci is None or len(cc) == 0:
        return None, None
    img_pts = cc.reshape(-1, 2).astype(np.float32)
    obj_pts = CHESS_3D[ci.flatten()].reshape(-1, 3).astype(np.float32)
    return img_pts, obj_pts


# ════════════════════════════════════════════════════════════
# 캘리브레이션
# ════════════════════════════════════════════════════════════

def load_detections():
    files = sorted(glob.glob(os.path.join(img_dir, "img_*.png")))
    obj_list, img_list, used = [], [], []
    img_size = None
    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img_size = gray.shape[::-1]
        ip, op = detect_points(gray)
        if ip is None or len(ip) < 6:
            continue
        obj_list.append(op); img_list.append(ip); used.append(f)
    return obj_list, img_list, used, img_size


def stability_check(n_trials=20):
    """이미지 절반씩 무작위로 뽑아 반복 캘리브레이션 → 파라미터가 흔들리는 정도 확인"""
    obj_list, img_list, used, size = load_detections()
    n = len(used)
    if n < 10:
        print(f"!! 이미지 {n}장 — 안정성 검사는 10장 이상 필요")
        return
    rng = np.random.default_rng(0)
    zt = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    for label, flags in [("기본 (k3 고정)", cv2.CALIB_FIX_K3),
                         ("접선왜곡 0 고정 (p1=p2=0)", zt),
                         ("k1만 사용 (k2=k3=0, p=0)", zt | cv2.CALIB_FIX_K2)]:
        rms_all, K_all, d_all, *_ = cv2.calibrateCamera(obj_list, img_list, size, None, None, flags=flags)
        rows = []
        for _ in range(n_trials):
            idx = rng.choice(n, n // 2, replace=False)
            _, K, d, _, _ = cv2.calibrateCamera([obj_list[i] for i in idx],
                                                [img_list[i] for i in idx], size, None, None, flags=flags)
            rows.append([K[0, 0], K[0, 2], K[1, 2], d.ravel()[0], d.ravel()[1]])
        rows = np.array(rows)
        print(f"\n[{label}]  전체 {n}장 RMS={rms_all:.3f}px")
        print(f"   전체 : fx={K_all[0,0]:.1f} cx={K_all[0,2]:.1f} cy={K_all[1,2]:.1f} "
              f"k1={d_all.ravel()[0]:+.3f} k2={d_all.ravel()[1]:+.3f} "
              f"p1={d_all.ravel()[2]:+.4f} p2={d_all.ravel()[3]:+.4f}")
        print(f"   절반 {n_trials}회 표준편차: fx ±{rows[:,0].std():.1f}  cx ±{rows[:,1].std():.1f}  "
              f"cy ±{rows[:,2].std():.1f}  k1 ±{rows[:,3].std():.3f}  k2 ±{rows[:,4].std():.3f}")
    print("\n  해석: cx/cy 표준편차가 ±10px 이내 → 안정.  ±30px 이상 → 데이터 부족(더 크게 기울인 사진 추가)")
    print("        아래 방식으로 갈수록 파라미터가 줄어듦. RMS가 0.05px 이상 늘지 않으면서")
    print("        표준편차가 더 작은 방식을 선택 (ZERO_TANGENT / FIX_K2 설정)")


def calibrate():
    files = sorted(glob.glob(os.path.join(img_dir, "img_*.png")))
    if len(files) < 8:
        print(f"!! 이미지가 {len(files)}장뿐 — 최소 8장, 권장 15~25장")
        return None

    obj_list, img_list, used = [], [], []
    img_size = None
    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img_size = gray.shape[::-1]
        ip, op = detect_points(gray)
        if ip is None or len(ip) < 6:
            print(f"  skip (코너 부족): {os.path.basename(f)}")
            continue
        obj_list.append(op)
        img_list.append(ip)
        used.append(f)

    print(f"\n[캘리브레이션] 사용 이미지 {len(used)}장, 해상도 {img_size[0]}x{img_size[1]}")
    # 화각이 좁은 렌즈는 k3가 과적합되기 쉬움 → 고정
    flags = cv2.CALIB_FIX_K3 if FIX_K3 else 0
    if ZERO_TANGENT:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if FIX_K2:
        flags |= cv2.CALIB_FIX_K2
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_list, img_list, img_size, None, None,
                                                     flags=flags)

    # 이미지별 재투영 오차
    per_img = []
    for op, ip, rv, tv in zip(obj_list, img_list, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        per_img.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip) ** 2, axis=1)))))
    med = float(np.median(per_img))

    print(f"\n  RMS 재투영 오차 : {rms:.3f} px   (0.5 이하 좋음, 1.0 이상이면 재촬영 권장)")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"  광축 중심 vs 화면 중심: dx={K[0,2] - img_size[0]/2:+.1f}  dy={K[1,2] - img_size[1]/2:+.1f} px")
    print(f"  왜곡계수 k1,k2,p1,p2,k3 = {np.round(dist.ravel()[:5], 4)}")
    print("\n  [이미지별 오차]  (중앙값의 2배 넘으면 ← 표시: 지우고 --offline 재실행 고려)")
    for f, e in zip(used, per_img):
        flag = "  ←" if e > 2 * med else ""
        print(f"    {os.path.basename(f)}: {e:.3f} px{flag}")

    result = dict(K=K, dist=dist, rms=rms, width=img_size[0], height=img_size[1])

    # 스테이지 평면 (P 이미지)
    if os.path.exists(plane_path):
        gray = cv2.imread(plane_path, cv2.IMREAD_GRAYSCALE)
        ip, op = detect_points(gray)
        if ip is not None and len(ip) >= 6:
            ok, rv, tv = cv2.solvePnP(op, ip, K, dist)
            R, _ = cv2.Rodrigues(rv)
            n = R[:, 2].copy()
            if np.dot(n, tv.ravel()) < 0:      # 카메라에서 멀어지는 방향으로 통일
                n = -n
            p0 = tv.ravel() + BOARD_THICKNESS_MM * n     # 받침판 두께만큼 아래로 → 스테이지 바닥
            tilt = np.degrees(np.arccos(abs(n[2])))
            dist_mm = float(np.dot(n, p0))
            print(f"\n  [스테이지 평면]  카메라→평면 거리 {dist_mm:.2f} mm, "
                  f"광축 대비 기울기 {tilt:.2f}°")
            print(f"    평면 위 한 점 p0 = {np.round(p0, 2)} mm,  법선 n = {np.round(n, 4)}")
            result.update(plane_R=R, plane_t=tv.ravel(), plane_p0=p0, plane_n=n,
                          plane_dist_mm=dist_mm)
        else:
            print("\n  !! calib_plane.png 에서 보드 검출 실패 — 평면 정보 생략")

    np.savez(out_npz, **result)
    fs = cv2.FileStorage(out_yaml, cv2.FILE_STORAGE_WRITE)
    for k, v in result.items():
        fs.write(k, np.asarray(v, dtype=np.float64) if not np.isscalar(v) else float(v))
    fs.release()
    print(f"\n✅ 저장: {os.path.basename(out_npz)}, {os.path.basename(out_yaml)}")
    return result


# ════════════════════════════════════════════════════════════
# 촬영
# ════════════════════════════════════════════════════════════

def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE)
    cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
    return cap


def light(ser, cmd):
    if ser is not None:
        ser.write(cmd)
        time.sleep(0.1)


def run_capture():
    os.makedirs(img_dir, exist_ok=True)

    ser = None
    if SERIAL_PORT:
        try:
            import serial
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
            time.sleep(2.0)          # 아두이노 리셋 대기
            light(ser, ALL_ON_CMD)
        except Exception as e:
            print(f"  (시리얼 연결 실패 — 실내등으로 진행: {e})")
            ser = None

    cap = open_camera()
    ok, frame = cap.read()
    if not ok:
        print("카메라 읽기 실패")
        return
    print(f"[카메라] {frame.shape[1]}x{frame.shape[0]}")

    # 기존 저장분 코너 → 커버리지 표시용
    coverage = []
    existing = sorted(glob.glob(os.path.join(img_dir, "img_*.png")))
    for f in existing:
        ip, _ = detect_points(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
        if ip is not None:
            coverage.append(ip)
    n_saved = len(existing)
    next_idx = 0 if not existing else int(os.path.basename(existing[-1])[4:7]) + 1

    win = "ChArUco calib"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 800)
    msg, msg_t = "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ip, op = detect_points(gray)
        n = 0 if ip is None else len(ip)

        disp = frame.copy()
        # 지금까지 저장된 코너 위치 (화면을 골고루 채우는 게 목표)
        for pts in coverage:
            for x, y in pts.astype(int):
                cv2.circle(disp, (x, y), 2, (0, 160, 0), -1)
        # 현재 검출 코너
        if ip is not None:
            for x, y in ip.astype(int):
                cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)

        good = n >= MIN_CORNERS
        lines = [
            f"corners: {n}/{(SQUARES_X-1)*(SQUARES_Y-1)}  {'SPACE: save' if good else '(too few)'}",
            f"saved: {n_saved}   exposure={cap.get(cv2.CAP_PROP_EXPOSURE):+.0f}   "
            f"plane: {'YES' if os.path.exists(plane_path) else 'no'}",
            "SPACE save | P plane | [ ] exposure | C calibrate | ESC quit",
        ]
        for k, s in enumerate(lines):
            cv2.putText(disp, s, (10, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if (k == 0 and good) else (0, 255, 255), 2)
        if msg and time.time() - msg_t < 1.5:
            cv2.putText(disp, msg, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):
            if good:
                path = os.path.join(img_dir, f"img_{next_idx:03d}.png")
                cv2.imwrite(path, gray)            # PNG 무손실
                coverage.append(ip)
                next_idx += 1; n_saved += 1
                msg, msg_t = f"saved {os.path.basename(path)} ({n} corners)", time.time()
                print(f"  {msg}")
            else:
                msg, msg_t = "코너 부족 - 저장 안 함", time.time()
        elif key in (ord('p'), ord('P')):
            if good:
                cv2.imwrite(plane_path, gray)
                msg, msg_t = "plane image saved", time.time()
                print("  평면 이미지 저장: calib_plane.png")
        elif key == ord('['):
            cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) - 1)
        elif key == ord(']'):
            cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) + 1)
        elif key in (ord('c'), ord('C')):
            calibrate()

    light(ser, ALL_OFF_CMD)
    if ser is not None:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--check" in sys.argv:
        stability_check()
    elif "--offline" in sys.argv:
        calibrate()
    else:
        run_capture()