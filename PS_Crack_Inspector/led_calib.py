"""
근거리 PS — LED 3D 위치 캘리브레이션 (크롬볼 다중 위치 삼각측량)

필요: camera_intrinsics.npz (charuco_calib.py 결과: K, dist, 스테이지 평면)

원리
  1. 볼 테두리 점들(클릭) → 각 점의 시선 광선 → 볼 방향(원뿔 축)
     볼 중심 깊이는 "스테이지 위에 놓인 볼" 조건으로 계산 (평면 + 반지름)
  2. LED별 하이라이트 픽셀 → 시선 광선 → 구와 교차 → 법선 → 반사 → LED를 향하는 3D 직선
  3. 볼을 여러 위치에 놓으면 LED마다 직선이 여러 개 → 가장 가깝게 모이는 점 = LED 위치

[촬영]  python led_calib.py
  볼을 스테이지에 놓는다 → 미리보기에서 볼 '테두리'를 6~10곳 클릭
    좌클릭: 점 추가 | 우클릭: 마지막 점 취소 | C: 전체 취소
    [ / ] : 미리보기 노출 조절 (테두리가 잘 보이게)
    - / = : 볼 받침 높이 -1 / +1 mm,  9 / 0 : -10 / +10 mm (스테이지에 바로 놓으면 0)
    ENTER : 이 위치에서 LED 8개 촬영 (점 5개 이상 필요)
    Q     : 촬영 끝내고 계산 | ESC: 계산 없이 종료
  → 볼을 다른 위치로 옮겨서 반복 (4~5곳 권장: 화면 네 귀퉁이 쪽 + 중앙)

[계산만]  python led_calib.py --solve
  led_calib/ 폴더의 저장된 촬영본으로 다시 계산

결과: led_positions.txt (카메라 좌표계, mm), led_positions.npz
"""
import os
import sys
import glob
import json
import time
import cv2
import numpy as np

# ────────────────────────────────────────────────────────────
# [설정]
# ────────────────────────────────────────────────────────────
SERIAL_PORT     = 'COM14'
BAUD_RATE       = 115200
ACK_TIMEOUT     = 2.0
ALL_ON_CMD      = b'9'
ALL_OFF_CMD     = b'0'
LED_COUNT       = 8

CAMERA_INDEX    = 1
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 800
EXPOSURE_VIEW   = -6      # 테두리 클릭용 미리보기 노출 (밝게)
EXPOSURE_HL     = -11     # 하이라이트 촬영 노출 (어둡게 → 하이라이트 작고 둥글게)
CAM_GAIN        = 0
FRAMES_AVG      = 4       # LED당 평균 프레임 수
DISCARD_FRAMES  = 2
FLUSH_SEC       = 0.12

BALL_DIAMETER_MM = 30.0
BALL_SUPPORT_MM  = 1.9    # 볼 받침(링 등)이 있으면 그 높이. 스테이지에 바로 놓으면 0

# 계산: 번들 조정 (scipy 필요)
REFINE_BA        = True   # LED + 볼 중심(가로) 동시 최적화
BA_FREE_DEPTH    = False  # False 권장: 볼 깊이는 스테이지 평면+받침으로 고정
                          # (하이라이트로는 깊이가 거의 결정되지 않고, 테두리 크기는 크롬볼 특성상 작게 보임)
BA_EDGE_WEIGHT   = 1.0    # 테두리 클릭 신뢰도 (방향 정보만 사용)

# 링 받침: 볼이 링 구멍으로 가라앉는 양 보정 (링을 안 쓰면 0)
RING_HEIGHT_MM         = 1.9
RING_INNER_DIAMETER_MM = 7.2   # ← 링 안지름(mm) 입력. 0이면 보정 안 함

base_dir   = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)   # 실행 파일(exe)이면 exe 옆에 저장
            else os.path.dirname(os.path.abspath(__file__)))
INTR_PATH  = os.path.join(base_dir, "camera_intrinsics.npz")
OUT_DIR    = os.path.join(base_dir, "led_calib")
RESULT_TXT = os.path.join(base_dir, "led_positions.txt")
RESULT_NPZ = os.path.join(base_dir, "led_positions.npz")


# ════════════════════════════════════════════════════════════
# 기하 계산
# ════════════════════════════════════════════════════════════

def load_intrinsics(path=INTR_PATH):
    d = np.load(path)
    if "plane_n" not in d.files:
        raise RuntimeError("camera_intrinsics.npz 에 스테이지 평면 정보가 없음 (charuco_calib.py 에서 P 촬영 필요)")
    return dict(K=d["K"], dist=d["dist"], plane_n=d["plane_n"].ravel(),
                plane_d=float(d["plane_dist_mm"]))


def pix_to_rays(pts, K, dist):
    """픽셀 좌표 Nx2 → 카메라 좌표계 단위 광선 Nx3 (왜곡 보정 포함)"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    rays = np.column_stack([und, np.ones(len(und))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def fit_cone(rays):
    """테두리 광선들이 이루는 원뿔: d_i · a = cos(α) → 축 a, cos(α)"""
    w, *_ = np.linalg.lstsq(rays, np.ones(len(rays)), rcond=None)
    nw = np.linalg.norm(w)
    return w / nw, 1.0 / nw


def sphere_center(edge_px, intr, R, support_mm=0.0):
    """볼 중심 (평면 조건 기반, 원뿔 깊이 기반) 둘 다 반환 [mm]
    support_mm: 볼 바닥이 스테이지보다 높은 양 (받침/블록 높이)"""
    rays = pix_to_rays(edge_px, intr["K"], intr["dist"])
    a, cos_a = fit_cone(rays)
    sin_a = np.sqrt(max(1.0 - cos_a ** 2, 1e-12))
    C_cone = (R / sin_a) * a
    n = intr["plane_n"]
    s = (intr["plane_d"] - R - support_mm) / float(n @ a)
    C_plane = s * a
    return C_plane, C_cone


def fit_circle(px):
    """2D 원 피팅 (대수적) → cx, cy, r  (하이라이트 탐색 영역용)"""
    px = np.asarray(px, dtype=np.float64)
    A = np.column_stack([2 * px[:, 0], 2 * px[:, 1], np.ones(len(px))])
    b = (px ** 2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return cx, cy, np.sqrt(c + cx ** 2 + cy ** 2)


def find_highlight(gray, circle, inner=0.92):
    """볼 영역 안 가장 밝은 blob의 가중 중심 (포화돼도 OK)"""
    cx, cy, r = circle
    h, w = gray.shape
    Y, X = np.ogrid[:h, :w]
    disk = (X - cx) ** 2 + (Y - cy) ** 2 <= (inner * r) ** 2
    k = max(3, int(r / 60)) | 1
    img = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float64)
    img[~disk] = 0
    mx = img.max()
    if mx < 30:
        return None
    m = (img >= 0.95 * mx).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    ys, xs = np.where(lab == big)
    wts = img[ys, xs]
    return float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())


def reflect_line(hl_px, C, R, intr):
    """하이라이트 픽셀 → 구 표면점 X, LED 방향 r (단위). 실패 시 None"""
    v = pix_to_rays([hl_px], intr["K"], intr["dist"])[0]
    vc = float(v @ C)
    disc = vc ** 2 - (C @ C - R ** 2)
    if disc < 0:
        return None
    s = vc - np.sqrt(disc)          # 카메라에 가까운 교점
    X = s * v
    n = (X - C) / R
    r = v - 2.0 * float(v @ n) * n
    return X, r / np.linalg.norm(r)


def highlight_point(C, L, R):
    """볼 중심 C, LED L → 카메라가 보는 하이라이트 3D 점 (반사 법칙 반복 풀이)"""
    nrm = -C / np.linalg.norm(C)
    for _ in range(60):
        X = C + R * nrm
        a = (L - X) / np.linalg.norm(L - X)
        b = -X / np.linalg.norm(X)
        nrm = (a + b) / np.linalg.norm(a + b)
    return C + R * nrm


def project_pts(P, intr):
    p, _ = cv2.projectPoints(np.asarray(P, float).reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                             intr["K"], intr["dist"])
    return p.reshape(-1, 2)


def edge_residual_px(C, edge_px, intr, R):
    """테두리 클릭 광선이 볼 중심 방향 기준으로 얼마나 '동심원'에서 벗어났는지 (픽셀).
    크롬볼 테두리는 일관되게 안쪽으로 보이므로 원의 크기는 쓰지 않고(평균 각도 제거)
    볼의 방향(가로 위치)만 제약한다."""
    rays = pix_to_rays(edge_px, intr["K"], intr["dist"])
    Cn = np.linalg.norm(C)
    ang = np.arccos(np.clip(rays @ (C / Cn), -1, 1))
    return (ang - ang.mean()) * intr["K"][0, 0]


def effective_support(sup_mm, R):
    """링 받침이면 볼이 링 구멍으로 가라앉는 만큼 보정한 '구 바닥 기준 높이'"""
    if RING_INNER_DIAMETER_MM > 0 and sup_mm > 0:
        r_in = RING_INNER_DIAMETER_MM / 2.0
        sink = R - np.sqrt(max(R ** 2 - r_in ** 2, 0.0))   # 링 윗면 아래로 내려가는 양
        # 링 높이 이상은 못 내려감(그 아래는 스테이지/블록 바닥)
        return sup_mm - min(sink, RING_HEIGHT_MM, sup_mm)
    return sup_mm


def bundle_adjust(obs, L0, intr, R, edge_weight=1.0, free_depth=True):
    """LED 위치 + 볼 중심(가로 2자유도) + 공통 깊이 보정 Δ 를 동시에 최적화.
    목적: 하이라이트 재투영 오차(px) + 테두리 클릭 오차(px)
    obs: [dict(C0, edge, hl={led: (x,y)})]"""
    from scipy.optimize import least_squares
    n = intr["plane_n"]
    u = np.cross(n, [1.0, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    nL, nP = len(L0), len(obs)

    def unpack(x):
        Ls = x[:3 * nL].reshape(nL, 3)
        delta = x[3 * nL] if free_depth else 0.0
        ab = x[3 * nL + (1 if free_depth else 0):].reshape(nP, 2)
        Cs = [o["C0"] + a[0] * u + a[1] * v + delta * n for o, a in zip(obs, ab)]
        return Ls, delta, ab, Cs

    # 관측 인덱스를 미리 펼쳐 둠 (벡터화)
    obs_k = np.array([k for k, o in enumerate(obs) for i in o["hl"]])
    obs_i = np.array([i for o in obs for i in o["hl"]])
    obs_px = np.array([o["hl"][i] for o in obs for i in o["hl"]], dtype=np.float64)
    edge_rays = [pix_to_rays(o["edge"], intr["K"], intr["dist"]) for o in obs]
    fpx = intr["K"][0, 0]

    def hl_points_vec(Cs_arr, Ls):
        C = Cs_arr[obs_k]; L = Ls[obs_i]
        nrm = -C / np.linalg.norm(C, axis=1, keepdims=True)
        for _ in range(40):
            X = C + R * nrm
            a = L - X; a /= np.linalg.norm(a, axis=1, keepdims=True)
            b = -X / np.linalg.norm(X, axis=1, keepdims=True)
            nrm = a + b; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        return C + R * nrm

    def resid(x):
        Ls, _, _, Cs = unpack(x)
        Cs_arr = np.array(Cs)
        r_hl = (project_pts(hl_points_vec(Cs_arr, Ls), intr) - obs_px).ravel()
        r_edge = []
        for C, rays in zip(Cs_arr, edge_rays):
            Cn = np.linalg.norm(C)
            ang = np.arccos(np.clip(rays @ (C / Cn), -1, 1))
            r_edge.append(edge_weight * (ang - ang.mean()) * fpx)   # 크기 무시, 방향만
        return np.concatenate([r_hl] + r_edge)

    x0 = np.concatenate([L0.ravel(), [0.0] if free_depth else [], np.zeros(2 * nP)])
    sol = least_squares(resid, x0, x_scale='jac')
    Ls, delta, ab, Cs = unpack(sol.x)

    hl_err = {}
    for k, (o, C) in enumerate(zip(obs, Cs)):
        for i, hl in o["hl"].items():
            hl_err[(i, k)] = float(np.linalg.norm(project_pts(highlight_point(C, Ls[i], R), intr)[0] - hl))
    edge_err = [np.abs(edge_residual_px(C, o["edge"], intr, R)) for o, C in zip(obs, Cs)]
    return dict(L=Ls, delta=float(delta), shifts=np.linalg.norm(ab, axis=1), C=Cs,
                hl_err=hl_err, edge_err=edge_err)


def triangulate(points, dirs):
    """직선들 (X_k + t r_k) 에 가장 가까운 점 P, 각 직선까지 거리"""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for X, r in zip(points, dirs):
        M = np.eye(3) - np.outer(r, r)
        A += M; b += M @ X
    P = np.linalg.solve(A, b)
    res = [float(np.linalg.norm((np.eye(3) - np.outer(r, r)) @ (P - X)))
           for X, r in zip(points, dirs)]
    # 조건수: 직선들 방향이 다양할수록 작음 (크면 위치 부정확)
    cond = float(np.linalg.cond(A))
    return P, np.array(res), cond


# ════════════════════════════════════════════════════════════
# 계산 (저장된 촬영본으로)
# ════════════════════════════════════════════════════════════

def solve(out_dir=OUT_DIR, intr=None, verbose=True):
    intr = intr or load_intrinsics()
    R = BALL_DIAMETER_MM / 2.0
    pos_dirs = sorted(glob.glob(os.path.join(out_dir, "pos_*")))
    if len(pos_dirs) < 2:
        print(f"!! 볼 위치가 {len(pos_dirs)}곳뿐 — 최소 2곳, 권장 4~5곳")
        return None

    lines = {i: [] for i in range(LED_COUNT)}     # led → [(X, r, pos_name)]
    obs, names = [], []
    print(f"\n[볼 위치 {len(pos_dirs)}곳]")
    for pd in pos_dirs:
        meta = json.load(open(os.path.join(pd, "meta.json")))
        edge = np.array(meta["edge_points"], dtype=np.float64)
        sup_in = float(meta.get("support_mm", BALL_SUPPORT_MM))
        sup = effective_support(sup_in, R)
        C, C_cone = sphere_center(edge, intr, R, sup)
        circle = fit_circle(edge)
        name = os.path.basename(pd)
        print(f"  {name}: 받침 {sup_in:.1f}mm (유효 {sup:.2f}), 중심 {np.round(C, 2)} mm   "
              f"(테두리 크기 기반 깊이와 차이 {np.linalg.norm(C_cone) - np.linalg.norm(C):+.2f} mm)")
        o = dict(C0=C, edge=edge, hl={})
        obs.append(o); names.append(name)
        for i in range(LED_COUNT):
            p = os.path.join(pd, f"led{i}.png")
            if not os.path.exists(p):
                continue
            hl = find_highlight(cv2.imread(p, cv2.IMREAD_GRAYSCALE), circle)
            if hl is None:
                print(f"     LED{i}: 하이라이트 없음")
                continue
            out = reflect_line(hl, C, R, intr)
            if out is None:
                print(f"     LED{i}: 하이라이트가 볼 밖 — 테두리 클릭 확인")
                continue
            lines[i].append((out[0], out[1], name))
            o["hl"][i] = np.array(hl)

    print("\n[1차: 직선 삼각측량]  (카메라 좌표계, mm. z = 카메라에서의 깊이)")
    print("   LED |        x        y        z | 직선 수 | 평균 잔차 | 최대 잔차 | 스테이지 위 높이")
    P_all = np.full((LED_COUNT, 3), np.nan)
    res_all = np.full(LED_COUNT, np.nan)
    n = intr["plane_n"]; d = intr["plane_d"]
    for i in range(LED_COUNT):
        L = lines[i]
        if len(L) < 2:
            print(f"   {i:3d} |  (직선 {len(L)}개 — 계산 불가)")
            continue
        P, res, cond = triangulate([l[0] for l in L], [l[1] for l in L])
        P_all[i] = P; res_all[i] = res.mean()
        height = d - float(n @ P)
        warn = "  ← 잔차 큼" if res.max() > 1.0 else ""
        print(f"   {i:3d} | {P[0]:8.2f} {P[1]:8.2f} {P[2]:8.2f} | {len(L):7d} | "
              f"{res.mean():7.3f}mm | {res.max():7.3f}mm | {height:8.2f} mm{warn}")

    print("\n  잔차 = 각 볼 위치에서 구한 직선이 LED 점에서 떨어진 거리 (mm)")

    # 위치별 잔차 표 → 특정 pos가 나쁜지 확인
    print("\n  [직선 잔차 mm]  행=LED, 열=pos")
    print("        " + " ".join(f"{nm[-2:]:>6}" for nm in names))
    for i in range(LED_COUNT):
        if np.isnan(P_all[i, 0]):
            continue
        row = []
        for nm in names:
            hit = [l for l in lines[i] if l[2] == nm]
            if not hit:
                row.append("     -"); continue
            X, r, _ = hit[0]
            row.append(f"{np.linalg.norm((np.eye(3) - np.outer(r, r)) @ (P_all[i] - X)):6.2f}")
        print(f"  LED{i}: " + " ".join(row))

    # 2차: 번들 조정 (볼 중심 보정 + 공통 깊이 보정 + LED 동시 최적화)
    valid = ~np.isnan(P_all[:, 0])
    if REFINE_BA and valid.all():
        print("\n[2차: 번들 조정 계산 중... (수 초 소요)]", flush=True)
        try:
            ba = bundle_adjust(obs, P_all, intr, R, edge_weight=BA_EDGE_WEIGHT,
                               free_depth=BA_FREE_DEPTH)
        except ImportError:
            print("\n  (scipy 없음 → 번들 조정 생략: pip install scipy)")
            ba = None
        if ba is not None:
            hl_before = []
            for k, o in enumerate(obs):
                for i, hl in o["hl"].items():
                    hl_before.append(np.linalg.norm(
                        project_pts(highlight_point(o["C0"], P_all[i], R), intr)[0] - hl))
            e_before = np.concatenate([np.abs(edge_residual_px(o["C0"], o["edge"], intr, R)) for o in obs])
            e_after = np.concatenate(ba["edge_err"])
            hl_after = np.array(list(ba["hl_err"].values()))
            print("\n[2차: 번들 조정]  LED 위치 + 볼 중심을 함께 미세 보정")
            print(f"  하이라이트 재투영 오차: 평균 {np.mean(hl_before):.2f}px → {hl_after.mean():.2f}px "
                  f"(최대 {hl_after.max():.2f})")
            print(f"  테두리 클릭 오차     : 평균 {e_before.mean():.2f}px → {e_after.mean():.2f}px "
                  f"(최대 {e_after.max():.2f})  ※ 원 크기 제외, 동심도만")
            if BA_FREE_DEPTH:
                print(f"  공통 깊이 보정 Δ = {ba['delta']:+.2f} mm")
            print("  볼 중심 가로 보정(mm): " +
                  ", ".join(f"{nm[-2:]}:{s:.2f}" for nm, s in zip(names, ba["shifts"])))
            bad = [nm for nm, s in zip(names, ba["shifts"]) if s > 1.0]
            if bad:
                print(f"  ← {', '.join(bad)}: 1mm 넘게 보정됨 → 그 위치 테두리 클릭이 부정확했을 가능성")
            print("\n   LED |        x        y        z | 스테이지 위 높이 | 하이라이트 오차(평균px)")
            for i in range(LED_COUNT):
                Pi = ba["L"][i]
                e_i = np.mean([v for (li, _), v in ba["hl_err"].items() if li == i])
                print(f"   {i:3d} | {Pi[0]:8.2f} {Pi[1]:8.2f} {Pi[2]:8.2f} | {d - float(n @ Pi):8.2f} mm"
                      f"      | {e_i:.2f}")
            print("\n  하이라이트 오차 1~2px 이하면 양호.")
            P_all = ba["L"]

    with open(RESULT_TXT, "w") as f:
        f.write(f"{LED_COUNT}\n")
        for P in P_all:
            f.write(f"{P[0]:.4f} {P[1]:.4f} {P[2]:.4f}\n")
    np.savez(RESULT_NPZ, led_positions=P_all, mean_residual=res_all,
             K=intr["K"], dist=intr["dist"], plane_n=n, plane_d=d)
    print(f"\n✅ 저장: {os.path.basename(RESULT_TXT)}, {os.path.basename(RESULT_NPZ)}")
    return P_all, res_all


# ════════════════════════════════════════════════════════════
# 촬영
# ════════════════════════════════════════════════════════════

def wait_ack(ser, expected):
    deadline = time.time() + ACK_TIMEOUT
    while time.time() < deadline:
        if ser.in_waiting:
            if expected in ser.readline().decode(errors='ignore'):
                return True
    print(f"  !! ACK 타임아웃: '{expected}'")
    return False


def flush(cap, sec=FLUSH_SEC):
    t = time.time() + sec
    while time.time() < t:
        cap.read()


def set_exposure(cap, e):
    cap.set(cv2.CAP_PROP_EXPOSURE, e)
    flush(cap, 0.3)


def grab_avg(cap, n):
    acc = None
    for _ in range(DISCARD_FRAMES):
        cap.read()
    got = 0
    while got < n:
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
        acc = g if acc is None else acc + g
        got += 1
    return np.clip(np.round(acc / n), 0, 255).astype(np.uint8)


def capture_position(cap, ser, edge_pts, pos_dir, support_mm):
    os.makedirs(pos_dir, exist_ok=True)
    view = grab_avg(cap, 2)
    cv2.imwrite(os.path.join(pos_dir, "view.png"), view)

    ser.write(ALL_OFF_CMD); wait_ack(ser, "OFF")
    set_exposure(cap, EXPOSURE_HL)
    for i in range(1, LED_COUNT + 1):
        ser.write(str(i).encode())
        if not wait_ack(ser, "ON"):
            continue
        flush(cap)
        img = grab_avg(cap, FRAMES_AVG)
        cv2.imwrite(os.path.join(pos_dir, f"led{i-1}.png"), img)
        ser.write(ALL_OFF_CMD); wait_ack(ser, "OFF")
    set_exposure(cap, EXPOSURE_VIEW)
    ser.write(ALL_ON_CMD); wait_ack(ser, "ON")

    json.dump({"edge_points": [[float(x), float(y)] for x, y in edge_pts],
               "ball_diameter_mm": BALL_DIAMETER_MM,
               "support_mm": float(support_mm),
               "exposure_hl": EXPOSURE_HL},
              open(os.path.join(pos_dir, "meta.json"), "w"), indent=1)

    # 확인용 합성: 8개 하이라이트 위치 표시
    circle = fit_circle(edge_pts)
    comp = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    cv2.circle(comp, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 0, 0), 1)
    found = 0
    for i in range(LED_COUNT):
        p = os.path.join(pos_dir, f"led{i}.png")
        if not os.path.exists(p):
            continue
        hl = find_highlight(cv2.imread(p, cv2.IMREAD_GRAYSCALE), circle)
        if hl:
            found += 1
            cv2.circle(comp, (int(hl[0]), int(hl[1])), 6, (0, 0, 255), 2)
            cv2.putText(comp, str(i), (int(hl[0]) + 8, int(hl[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(pos_dir, "check.png"), comp)
    return comp, found


def run_capture():
    import serial
    intr = load_intrinsics()
    os.makedirs(OUT_DIR, exist_ok=True)

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
    t_end = time.time() + 5.0
    while time.time() < t_end:
        if ser.in_waiting and "READY" in ser.readline().decode(errors='ignore'):
            break
    ser.reset_input_buffer()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
    set_exposure(cap, EXPOSURE_VIEW)
    ser.write(ALL_ON_CMD); wait_ack(ser, "ON")

    existing = sorted(glob.glob(os.path.join(OUT_DIR, "pos_*")))
    pos_idx = 0 if not existing else int(os.path.basename(existing[-1])[4:]) + 1
    print(f"\n기존 저장 위치 {len(existing)}곳. 다음: pos_{pos_idx:02d}")

    pts, mouse = [], [0, 0]
    support = [BALL_SUPPORT_MM]
    msg = ["", 0.0]

    def on_mouse(ev, x, y, flags, p):
        mouse[0], mouse[1] = x, y
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    win = "LED calib - click ball edge"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 800)
    cv2.setMouseCallback(win, on_mouse)
    do_solve = False

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()
        H, W = disp.shape[:2]

        for (x, y) in pts:
            cv2.circle(disp, (x, y), 3, (0, 0, 255), -1)
        if len(pts) >= 3:
            cx, cy, r = fit_circle(pts)
            cv2.circle(disp, (int(cx), int(cy)), int(r), (0, 255, 0), 1)

        # 돋보기 (마우스 주변 4배 확대) — 테두리 정밀 클릭용
        mx, my = mouse
        hw = 30
        x0, y0 = int(np.clip(mx - hw, 0, W - 2 * hw)), int(np.clip(my - hw, 0, H - 2 * hw))
        zoom = cv2.resize(frame[y0:y0 + 2 * hw, x0:x0 + 2 * hw], (240, 240),
                          interpolation=cv2.INTER_NEAREST)
        cv2.line(zoom, (120, 0), (120, 240), (0, 255, 255), 1)
        cv2.line(zoom, (0, 120), (240, 120), (0, 255, 255), 1)
        zx = W - 250 if mx < W / 2 else 10
        disp[H - 250:H - 10, zx:zx + 240] = zoom

        info = [f"pos_{pos_idx:02d}   edge points: {len(pts)}  (need >= 5)",
                f"support height = {support[0]:.1f} mm   (- / = : -1/+1mm,  9 / 0 : -10/+10mm)",
                f"exposure(view)={cap.get(cv2.CAP_PROP_EXPOSURE):+.0f}  [ ]",
                "L-click add | R-click undo | C clear | ENTER capture | Q solve | ESC quit"]
        for k, s in enumerate(info):
            col = (0, 255, 0) if (k == 0 and len(pts) >= 5) else (0, 255, 255)
            cv2.putText(disp, s, (10, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
        if msg[0] and time.time() - msg[1] < 2.0:
            cv2.putText(disp, msg[0], (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('q'), ord('Q')):
            do_solve = True
            break
        elif key in (ord('c'), ord('C')):
            pts.clear()
        elif key == ord('['):
            set_exposure(cap, cap.get(cv2.CAP_PROP_EXPOSURE) - 1)
        elif key == ord(']'):
            set_exposure(cap, cap.get(cv2.CAP_PROP_EXPOSURE) + 1)
        elif key == ord('-'):
            support[0] = max(0.0, support[0] - 1)
        elif key == ord('='):
            support[0] += 1
        elif key == ord('9'):
            support[0] = max(0.0, support[0] - 10)
        elif key == ord('0'):
            support[0] += 10
        elif key in (13, 10) and len(pts) < 5:
            msg[0], msg[1] = f"edge points {len(pts)}/5 - click more on the ball edge", time.time()
        elif key in (13, 10):
            pos_dir = os.path.join(OUT_DIR, f"pos_{pos_idx:02d}")
            sup = support[0]
            cv2.putText(disp, "CAPTURING...", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow(win, disp); cv2.waitKey(1)
            print(f"  {os.path.basename(pos_dir)} 촬영 중... (받침 {sup:.1f}mm)")
            comp, found = capture_position(cap, ser, pts, pos_dir, sup)
            C, C_cone = sphere_center(np.array(pts, float), intr, BALL_DIAMETER_MM / 2, sup)
            print(f"    하이라이트 {found}/{LED_COUNT} 검출, 볼 중심 {np.round(C, 1)} mm, "
                  f"테두리 기반 깊이 차이 {np.linalg.norm(C_cone) - np.linalg.norm(C):+.2f} mm")
            cv2.imshow(win, comp)
            cv2.putText(comp, "check highlights - any key to continue", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, comp)
            cv2.waitKey(0)
            pts.clear()
            pos_idx += 1
            print("  → 볼을 다음 위치로 옮기고 테두리를 다시 클릭하세요.")

    ser.write(ALL_OFF_CMD)
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
    if do_solve:
        solve(intr=intr)


if __name__ == "__main__":
    if "--solve" in sys.argv:
        solve()
    else:
        run_capture()