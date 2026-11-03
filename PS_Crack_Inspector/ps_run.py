"""
근거리 PS — 촬영 + 복원 (흰 종이 기준 영상으로 LED 밝기/빔 모양 보정)

필요: camera_intrinsics.npz (1단계), led_positions.npz (2단계)

[1] 기준 촬영 (한 번만):  python ps_run.py ref
    흰 무광 종이를 화면 전체에 평평하게 깔고 실행.
    1~8: LED 선택해 보기 | [ ]: 노출 | ENTER: 촬영
    → 가장 밝은 LED도 포화(빨강) 없게 노출을 맞출 것. 물체가 종이보다 밝으면 노출을 더 낮게.

[2] 물체 촬영 + 균열 판정: python ps_run.py obj 이름
    종이를 치우고 물체를 놓고 실행. 노출은 기준과 동일하게 고정됨.
    ENTER: 촬영 → 균열 판정 (약 4초) → 판정 그림 창 ('균열이 있습니다' / '무늬입니다')
    → ps_out/이름/fast/verdict.png 저장 (판정 방법: camtest_and_solver.py 참고)

[다시 판정만]            python ps_run.py judge 이름
[높이 복원 (예전 방식)]  python ps_run.py solve 이름        (느림, 원본 해상도: solve 이름 full)
  결과: normal / albedo / curvature / concavity / height → ps_out/이름/

※ 기준 촬영 이후 카메라/렌즈/LED/노출/게인을 바꾸면 기준을 다시 찍어야 함
※ 첫 테스트는 '평평한 물체'(두꺼운 종이, 책 표지 등)로 → 평평하게 나오는지 확인
"""
import os
import sys
import warnings
import json
import time
import cv2
import numpy as np

# ────────────────────────────────────────────────────────────
# [설정]
# ────────────────────────────────────────────────────────────
SERIAL_PORT   = 'COM14'
BAUD_RATE     = 115200
ACK_TIMEOUT   = 2.0
LED_COUNT     = 8

CAMERA_INDEX  = 1
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 800
CAM_EXPOSURE  = -2        # 기준 촬영 시작값 ([ ] 로 조절, 물체 촬영은 기준값으로 고정)
CAM_GAIN      = 100
CAM_GAMMA     = 100       # 100 = 선형
CAM_BRIGHTNESS = 16       # 블랙 레벨: 0이면 어두운 값이 0으로 잘림 (camtest_and_solver.py lin 으로 확인)
CAM_CONTRAST  = 1         # 32 등 올리면 톤 커브가 급해져서 어두운/밝은 쪽이 다 잘림
FRAMES_AVG    = 8
DISCARD       = 3
FLUSH_SEC     = 0.15
SAT_LEVEL     = 245

PAPER_THICKNESS_MM = 0.1  # 기준 종이 두께 (스테이지 위 높이)

# 복원
DOWNSAMPLE    = 2         # 1=원본(느림), 2=640x400
MIN_REF       = 8.0       # 기준(종이) 신호가 이보다 작으면 그 LED는 빔 밖 → 제외
MIN_LEDS      = 3         # 픽셀당 최소 유효 LED 수
DROP_OUTLIER  = True      # 그림자 제거 후에도 유효 LED ≥5 이면 잔차 최대 1개 제외 (반사광)
SHADOW_FRAC   = 0.35      # 같은 픽셀 다른 LED 중앙값의 이 비율보다 어두우면 그림자로 제외
SHADOW_PRED_FRAC = 0.6    # 1차 풀이의 예측값보다 이 비율 이하로 어두우면 그림자로 제외
N_ITER        = 4         # 높이 ↔ 빛 방향 반복 횟수
BG_FLATTEN    = True      # 배경(기준과 같은 종이)이 평평하다는 조건으로 저주파 휨 제거
BG_POLY_DEG   = 3         # 휨 보정 다항식 차수
BG_MARGIN_PX  = 15        # 물체 경계에서 이만큼(다운샘플 px) 떨어진 곳만 배경으로 사용
CONCAVE_SIGMA = 1.0       # 오목도 계산 전 기울기 평활 (px). 노이즈 많으면 1.5~2

base_dir  = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)   # 실행 파일(exe)이면 exe 옆에 저장
            else os.path.dirname(os.path.abspath(__file__)))
INTR_PATH = os.path.join(base_dir, "camera_intrinsics.npz")
LED_PATH  = os.path.join(base_dir, "led_positions.npz")
REF_DIR   = os.path.join(base_dir, "ps_ref")
RESP_PATH = os.path.join(base_dir, "camera_response.npz")   # camtest_and_solver.py resp 가 만듦
OBJ_ROOT  = os.path.join(base_dir, "ps_obj")
OUT_ROOT  = os.path.join(base_dir, "ps_out")


# ════════════════════════════════════════════════════════════
# 복원
# ════════════════════════════════════════════════════════════

def load_geometry():
    ci = np.load(INTR_PATH)
    lp = np.load(LED_PATH)
    return dict(K=ci["K"], dist=ci["dist"], n=ci["plane_n"].ravel(),
                d=float(ci["plane_dist_mm"]), L=lp["led_positions"])


def response_for(folder):
    """그 촬영과 같은 카메라 설정(brightness/contrast)으로 추정한 응답 곡선이 있으면 반환"""
    if not os.path.exists(RESP_PATH):
        return None
    meta_p = os.path.join(folder, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    r = np.load(RESP_PATH)
    if meta.get("brightness") != float(r["brightness"]) or meta.get("contrast") != float(r["contrast"]):
        return None
    return r["knots"], r["g"]


def sat_level(folder):
    """포화 기준을 load_stack 결과와 같은 단위로"""
    resp = response_for(folder)
    return SAT_LEVEL if resp is None else float(np.interp(SAT_LEVEL, *resp))


def load_stack(folder):
    dark = np.load(os.path.join(folder, "dark.npy")).astype(np.float64)
    imgs = [np.load(os.path.join(folder, f"led{i}.npy")).astype(np.float64) for i in range(LED_COUNT)]
    resp = response_for(folder)
    if resp is not None:                    # 원본 DN → 선형값 (빼기 전에 변환해야 함)
        dark = np.interp(dark, *resp)
        imgs = [np.interp(im, *resp) for im in imgs]
    print(f"  {os.path.basename(folder)}: 응답 곡선 보정 {'적용' if resp is not None else '없음 (설정 불일치 또는 미보정)'}")
    return dark, imgs


def down(img, f):
    if f == 1:
        return img
    h, w = img.shape
    return img[:h - h % f, :w - w % f].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def frankot_chellappa(gx, gy):
    """기울기장 → 높이 (FFT 적분, 경계 반사 확장)"""
    H, W = gx.shape
    gx2 = np.block([[gx, -gx[:, ::-1]], [gx[::-1, :], -gx[::-1, ::-1]]])
    gy2 = np.block([[gy, gy[:, ::-1]], [-gy[::-1, :], -gy[::-1, ::-1]]])
    H2, W2 = gx2.shape
    wx = np.fft.fftfreq(W2) * 2 * np.pi
    wy = np.fft.fftfreq(H2) * 2 * np.pi
    WX, WY = np.meshgrid(wx, wy)
    denom = WX ** 2 + WY ** 2
    denom[0, 0] = 1.0
    Z = (-1j * WX * np.fft.fft2(gx2) - 1j * WY * np.fft.fft2(gy2)) / denom
    Z[0, 0] = 0
    return np.real(np.fft.ifft2(Z))[:H, :W]


def reconstruct(obj_dir, ref_dir=REF_DIR, ds=DOWNSAMPLE, verbose=True):
    geo = load_geometry()
    K, dist, n, d, Lpos = geo["K"], geo["dist"], geo["n"], geo["d"], geo["L"]
    ref_meta = json.load(open(os.path.join(ref_dir, "meta.json")))
    d_paper = d - ref_meta.get("paper_thickness_mm", PAPER_THICKNESS_MM)

    Rd, Rimg = load_stack(ref_dir)
    Od, Oimg = load_stack(obj_dir)
    sat_O = [down(im, ds) >= sat_level(obj_dir) for im in Oimg]
    R = np.array([down(im - Rd, ds) for im in Rimg])       # (nL,H,W)
    O = np.array([down(im - Od, ds) for im in Oimg])
    nL, H, W = R.shape

    # 픽셀 광선 (원본 해상도 좌표의 블록 중심)
    ys, xs = np.mgrid[0:H, 0:W]
    px = (xs + 0.5) * ds - 0.5
    py = (ys + 0.5) * ds - 0.5
    pts = np.column_stack([px.ravel(), py.ravel()]).reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    V = np.column_stack([und, np.ones(len(und))])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    V = V.reshape(H, W, 3)
    vn = V @ n                                              # 광선·평면법선

    # 기준 종이 위 점 & 그 점에서의 기하
    tP = d_paper / vn
    XP = V * tP[..., None]
    ref_geo = []
    for i in range(nL):
        D = Lpos[i] - XP
        r = np.linalg.norm(D, axis=-1)
        cosp = (D / r[..., None]) @ (-n)
        ref_geo.append((r, cosp))

    # 평면 좌표계 (높이 적분용)
    e1 = np.cross(n, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    U1, U2 = XP @ e1, XP @ e2
    du_dx, du_dy = np.gradient(U1, axis=1), np.gradient(U1, axis=0)
    dv_dx, dv_dy = np.gradient(U2, axis=1), np.gradient(U2, axis=0)

    ref_ok = R > MIN_REF
    obj_ok = ~np.array(sat_O) & (O > -5)
    valid_i = ref_ok & obj_ok                              # (nL,H,W)

    def ref_along_led_ray(i, X):
        """LED i → 물체점 X 직선을 연장해 종이와 만나는 점의 기준 밝기 (빔 방향 일치)"""
        Li = Lpos[i]
        Dv = X - Li
        s = (d_paper - Li @ n) / np.maximum(Dv @ n, 1e-9)
        XPp = Li + Dv * s[..., None]
        uv, _ = cv2.projectPoints(XPp.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
        uv = uv.reshape(H, W, 2)
        mx = ((uv[..., 0] + 0.5) / ds - 0.5).astype(np.float32)
        my = ((uv[..., 1] + 0.5) / ds - 0.5).astype(np.float32)
        Ri = cv2.remap(R[i].astype(np.float32), mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float64)
        Dp = Li - XPp
        rp = np.linalg.norm(Dp, axis=-1)
        cosp = (Dp / rp[..., None]) @ (-n)
        return Ri, rp, cosp

    # 배경 마스크: 물체 영상이 기준(종이)과 거의 같은 픽셀 = 같은 종이가 그대로 보이는 곳
    with np.errstate(divide='ignore', invalid='ignore'):
        rat = np.where(ref_ok, O / np.maximum(R, 1e-6), np.nan)
    med = np.nanmedian(rat, axis=0)
    spread = np.nanmax(rat, axis=0) - np.nanmin(rat, axis=0)
    bg = (np.abs(med - 1) < 0.12) & (spread < 0.3) & (ref_ok.sum(axis=0) >= MIN_LEDS)
    not_bg = (~bg).astype(np.uint8)
    k = 2 * BG_MARGIN_PX + 1
    not_bg = cv2.dilate(cv2.morphologyEx(not_bg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = bg & (not_bg == 0)

    def poly_terms(xn, yn, deg):
        return np.column_stack([xn ** i * yn ** j for i in range(deg + 1) for j in range(deg + 1 - i)])
    yy, xx = np.mgrid[0:H, 0:W]
    xn, yn = (xx / W - 0.5) * 2, (yy / H - 0.5) * 2
    bg_frac = bg.mean()

    h = np.zeros((H, W))                                   # 종이면 위 높이 (mm)
    for it in range(N_ITER):
        t = (d_paper - h) / vn
        X = V * t[..., None]
        Ls, M, Rs = [], [], []
        for i in range(nL):
            D = Lpos[i] - X
            r = np.linalg.norm(D, axis=-1)
            l = D / r[..., None]
            Ri, rp, cosp = ref_along_led_ray(i, X)
            ok = obj_ok[i] & (Ri > MIN_REF)
            # 물체/기준 비율 × 기준의 (n·l) × 거리 보정 → ρ'(n·l_i)
            m = np.where(ok, O[i] / np.maximum(Ri, 1e-6) * cosp * (r / rp) ** 2, 0.0)
            Ls.append(l); M.append(m); Rs.append(np.where(ok, Ri, 0.0))
        Ls = np.array(Ls)                                  # (nL,H,W,3)
        M = np.array(M)
        Rs = np.array(Rs)
        valid_i = Rs > 0
        Wt = valid_i * np.sqrt(Rs)                         # 기준 신호 강한 LED에 가중

        def solve_b(Wt):
            A = np.einsum('khw,khwi,khwj->hwij', Wt, Ls, Ls)
            c = np.einsum('khw,khw,khwi->hwi', Wt, M, Ls)
            A = A + np.eye(3) * 1e-9
            return np.linalg.solve(A, c[..., None])[..., 0]

        # 그림자 제거 1차: 같은 픽셀의 다른 LED 대비 너무 어두운 측정 (드리운 그림자)
        Mv = np.where(valid_i, M, np.nan)
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # 유효 LED가 하나도 없는 픽셀 (결과에 영향 없음)
            med_m = np.nanmedian(Mv, axis=0)
            shadow = valid_i & (M < SHADOW_FRAC * med_m[None])
        Wt = np.where(shadow, 0.0, Wt)
        b = solve_b(Wt)
        # 그림자 제거 2차: 예측보다 크게 어두운 측정 → 제외 후 재계산
        pred = np.einsum('khwi,hwi->khw', Ls, b)
        shadow2 = (Wt > 0) & (M < SHADOW_PRED_FRAC * pred)
        ok_after = ((Wt > 0) & ~shadow2).sum(axis=0) >= MIN_LEDS
        Wt = np.where(shadow2 & ok_after[None], 0.0, Wt)
        b = solve_b(Wt)
        if DROP_OUTLIER:                                   # 남은 이상값(반사광 등) 1개 제외
            cnt = (Wt > 0).sum(axis=0)
            res = np.abs(np.einsum('khwi,hwi->khw', Ls, b) - M) * (Wt > 0)
            worst = np.argmax(res, axis=0)
            drop = (cnt >= 5)
            kk = np.arange(nL)[:, None, None]
            Wt = np.where((kk == worst[None]) & drop[None], 0.0, Wt)
            b = solve_b(Wt)

        nvalid = (Wt > 0).sum(axis=0)
        rho = np.linalg.norm(b, axis=-1)
        N = b / np.maximum(rho[..., None], 1e-12)
        flip = np.einsum('hwi,hwi->hw', N, V) > 0          # 카메라 쪽을 향하게
        N[flip] *= -1
        good = (nvalid >= MIN_LEDS) & (rho > 0)

        # 평면 좌표계 기울기 → 픽셀 기울기 → 적분
        nz = N @ (-n)
        nz = np.where(np.abs(nz) < 0.05, 0.05, nz)
        p = -(N @ e1) / nz
        q = -(N @ e2) / nz
        p[~good] = 0; q[~good] = 0
        gx = p * du_dx + q * dv_dx
        gy = p * du_dy + q * dv_dy
        h = frankot_chellappa(gx, gy)
        if BG_FLATTEN and bg_frac > 0.05:
            m = bg & good
            A = poly_terms(xn[m], yn[m], BG_POLY_DEG)
            coef, *_ = np.linalg.lstsq(A, h[m], rcond=None)
            h = h - (poly_terms(xn.ravel(), yn.ravel(), BG_POLY_DEG) @ coef).reshape(H, W)
        else:
            h -= np.percentile(h[good], 5) if good.any() else 0
        if verbose:
            print(f"  반복 {it+1}/{N_ITER}: 높이 범위 {np.percentile(h,1):.2f} ~ {np.percentile(h,99):.2f} mm")

    mm_per_px = float(np.median(np.hypot(du_dx, dv_dx)))
    if verbose:
        print(f"  배경 영역 {bg_frac*100:.0f}% "
              + ("→ 배경 평탄화 적용" if BG_FLATTEN and bg_frac > 0.05 else "→ 평탄화 안 함"))
        if BG_FLATTEN and bg_frac > 0.05:
            print(f"  배경 평탄도: RMS {np.std(h[bg & good]):.3f} mm")
    return dict(N=N, rho=rho, h=h, good=good, nvalid=nvalid, mm_per_px=mm_per_px, ds=ds, bg=bg)


def flatness_report(res):
    """평면 피팅 → 평평한 물체 테스트용"""
    h, g = res["h"], res["good"]
    H, W = h.shape
    ys, xs = np.mgrid[0:H, 0:W]
    m = g & (xs > W * 0.05) & (xs < W * 0.95) & (ys > H * 0.05) & (ys < H * 0.95)
    A = np.column_stack([xs[m], ys[m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(A, h[m], rcond=None)
    resid = h[m] - A @ coef
    tilt = np.degrees(np.arctan(np.hypot(coef[0], coef[1]) / res["mm_per_px"]))
    return float(resid.std()), float(np.percentile(resid, 99) - np.percentile(resid, 1)), tilt


def save_results(res, out_dir, show=True):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any('Malgun' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    os.makedirs(out_dir, exist_ok=True)
    N, rho, h, g = res["N"], res["rho"], res["h"], res["good"]

    # 노멀 변화량 (곡률 유사) — 균열·홈·접힌 선 강조
    gxN = np.gradient(N, axis=1); gyN = np.gradient(N, axis=0)
    curv = np.sqrt((gxN ** 2 + gyN ** 2).sum(-1))
    curv[~g] = 0
    np.save(os.path.join(out_dir, "curvature.npy"), curv.astype(np.float32))
    well = g & (res["nvalid"] >= 5)      # 조명 충분한 영역 기준으로 색 범위 결정 (가장자리 노이즈 무시)
    cmax = np.percentile(curv[well], 99.7) if well.sum() > 100 else (np.percentile(curv[g], 99.5) if g.any() else 1.0)
    cimg = cv2.applyColorMap((np.clip(curv / max(cmax, 1e-6), 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(out_dir, "curvature.png"), cimg)

    # 오목도 (부호 있는 곡률) — 홈·균열은 가운데가 오목(음수)해서 강하게 나오고,
    # 색(무늬) 경계·그린 선의 가짜 신호는 +/− 쌍이라 거의 안 나옴
    nzc = np.where(-N[..., 2] < 0.05, 0.05, -N[..., 2])
    pc = cv2.GaussianBlur((N[..., 0] / nzc).astype(np.float32), (0, 0), CONCAVE_SIGMA)
    qc = cv2.GaussianBlur((N[..., 1] / nzc).astype(np.float32), (0, 0), CONCAVE_SIGMA)
    div = np.gradient(pc, axis=1) + np.gradient(qc, axis=0)
    concave = np.where(g, np.maximum(-div, 0), 0).astype(np.float32)
    np.save(os.path.join(out_dir, "concavity.npy"), concave)
    np.save(os.path.join(out_dir, "signed_curvature.npy"), np.where(g, div, 0).astype(np.float32))
    kmax = np.percentile(concave[well], 99.7) if well.sum() > 100 else (np.percentile(concave[g], 99.5) if g.any() else 1.0)
    kimg = cv2.applyColorMap((np.clip(concave / max(kmax, 1e-9), 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(out_dir, "concavity.png"), kimg)

    nm = ((N * np.array([1, -1, -1]) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)  # 보기용 (x→R, y↑→G, 카메라쪽→B)
    nm[~g] = 0
    cv2.imwrite(os.path.join(out_dir, "normal.png"), cv2.cvtColor(nm, cv2.COLOR_RGB2BGR))
    alb = rho / np.percentile(rho[g], 99) if g.any() else rho
    cv2.imwrite(os.path.join(out_dir, "albedo.png"), (alb.clip(0, 1) * 255).astype(np.uint8))
    np.save(os.path.join(out_dir, "height_mm.npy"), h)
    np.save(os.path.join(out_dir, "normal.npy"), N.astype(np.float32))

    bgm = res.get("bg", np.zeros_like(g)) & g
    if bgm.mean() > 0.05:
        obj = g & ~bgm
        txt = (f"배경(종이) 평탄도 RMS {np.std(h[bgm]):.3f} mm  |  "
               f"물체 높이 상위 5% {np.percentile(h[obj], 95) if obj.any() else 0:.2f} mm\n"
               f"픽셀 크기 약 {res['mm_per_px']:.3f} mm,  유효 LED 수 평균 {res['nvalid'][g].mean():.1f}")
    else:
        sd, pv, tilt = flatness_report(res)
        txt = (f"평면 피팅 잔차: RMS {sd:.3f} mm, P-V(1~99%) {pv:.3f} mm, 기울기 {tilt:.2f}°\n"
               f"픽셀 크기 약 {res['mm_per_px']:.3f} mm,  유효 LED 수 평균 {res['nvalid'][g].mean():.1f}")
    print("\n" + txt)

    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_subplot(2, 4, 1); ax.imshow(nm); ax.set_title("노멀 맵"); ax.axis('off')
    ax = fig.add_subplot(2, 4, 2); ax.imshow(alb, cmap='gray', vmin=0, vmax=1); ax.set_title("반사율(albedo)"); ax.axis('off')
    ax = fig.add_subplot(2, 4, 4)
    ax.imshow(np.clip(curv / max(cmax, 1e-6), 0, 1), cmap='inferno'); ax.set_title("노멀 변화량 (균열·홈)"); ax.axis('off')
    ax = fig.add_subplot(2, 4, 3)
    im = ax.imshow(np.where(g, h, np.nan), cmap='viridis'); ax.set_title("높이 (mm)"); ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax = fig.add_subplot(2, 4, 5)
    im = ax.imshow(res["nvalid"], cmap='magma', vmin=0, vmax=LED_COUNT); ax.set_title("유효 LED 수 (윤곽: 배경으로 쓴 영역)"); ax.axis('off')
    if "bg" in res:
        ax.contour(res["bg"].astype(float), levels=[0.5], colors='cyan', linewidths=0.8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax = fig.add_subplot(2, 4, 6)
    H, W = h.shape
    ax.plot(np.arange(W) * res["mm_per_px"], h[H // 2], label="가로 중앙선")
    ax.plot(np.arange(H) * res["mm_per_px"], h[:, W // 2], label="세로 중앙선")
    ax.set_xlabel("mm"); ax.set_ylabel("높이 mm"); ax.set_title("단면"); ax.legend(); ax.grid(alpha=.3)
    ax = fig.add_subplot(2, 4, 8)
    ax.imshow(np.clip(concave / max(kmax, 1e-9), 0, 1), cmap='inferno')
    ax.set_title("오목도 (홈·균열만, 무늬 경계 억제)"); ax.axis('off')
    ax = fig.add_subplot(2, 4, 7, projection='3d')
    s = max(1, max(H, W) // 120)
    Y, X = np.mgrid[0:H:s, 0:W:s] * res["mm_per_px"]
    ax.plot_surface(X, Y, h[::s, ::s], cmap='viridis', linewidth=0, antialiased=False)
    ax.set_title("3D"); ax.set_box_aspect((W, H, max(np.ptp(h) / res["mm_per_px"], 1)))
    fig.suptitle(txt, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "result.png"), dpi=120)
    print(f"\n✅ 저장: {out_dir}")
    if show:
        plt.show()


# ════════════════════════════════════════════════════════════
# 촬영
# ════════════════════════════════════════════════════════════

def wait_ack(ser, expected):
    t = time.time() + ACK_TIMEOUT
    while time.time() < t:
        if ser.in_waiting and expected in ser.readline().decode(errors='ignore'):
            return True
    print(f"  !! ACK 타임아웃 '{expected}'")
    return False


def led(ser, k):
    ser.write(str(k).encode())
    wait_ack(ser, "OFF" if k == 0 else "ON")


def flush(cap, sec=FLUSH_SEC):
    t = time.time() + sec
    while time.time() < t:
        cap.read()


def grab_avg(cap, n=FRAMES_AVG):
    for _ in range(DISCARD):
        cap.read()
    acc, got = None, 0
    while got < n:
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
        acc = g if acc is None else acc + g
        got += 1
    return (acc / n).astype(np.float32)


def open_devices(exposure):
    import serial
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
    t = time.time() + 5.0
    while time.time() < t:
        if ser.in_waiting and "READY" in ser.readline().decode(errors='ignore'):
            break
    ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_GAIN, CAM_GAIN)
    if CAM_GAMMA is not None:
        cap.set(cv2.CAP_PROP_GAMMA, CAM_GAMMA)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, CAM_BRIGHTNESS)   # 드라이버가 마지막 값을 기억하므로 항상 명시
    cap.set(cv2.CAP_PROP_CONTRAST, CAM_CONTRAST)
    return ser, cap


def capture(out_dir, exposure, allow_exposure_change):
    os.makedirs(out_dir, exist_ok=True)
    ser, cap = open_devices(exposure)
    cur = 1
    led(ser, cur)
    win = "PS capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 800)
    go = False
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        disp = f.copy()
        disp[g >= SAT_LEVEL] = (0, 0, 255)
        mx = int(cv2.GaussianBlur(g, (9, 9), 0).max())
        lines = [f"LED{cur-1} ON  max={mx} {'SATURATED' if mx >= SAT_LEVEL else ''}",
                 f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):+.0f} "
                 f"{'([ ] adjust)' if allow_exposure_change else '(locked to reference)'}  "
                 f"gain={cap.get(cv2.CAP_PROP_GAIN):.0f} gamma={cap.get(cv2.CAP_PROP_GAMMA):.0f}",
                 "1~8 LED | ENTER capture | ESC quit"]
        for k, s in enumerate(lines):
            cv2.putText(disp, s, (10, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if (k == 0 and mx >= SAT_LEVEL) else (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if ord('1') <= key <= ord('8'):
            cur = key - ord('0'); led(ser, 0); led(ser, cur)
        elif allow_exposure_change and key == ord('['):
            cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) - 1)
        elif allow_exposure_change and key == ord(']'):
            cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) + 1)
        elif key in (13, 10):
            go = True
            break
    cv2.destroyAllWindows()
    exp_used = cap.get(cv2.CAP_PROP_EXPOSURE)
    if go:
        print(f"\n촬영 (노출 {exp_used:+.0f}, {FRAMES_AVG}장 평균)")
        led(ser, 0); flush(cap, 0.3)
        dk = grab_avg(cap)
        np.save(os.path.join(out_dir, "dark.npy"), dk)
        # 평균엔 블랙 레벨(brightness)이 포함되므로, 주변광은 화면 내 불균일로 판단
        print(f"  암전 평균 {dk.mean():.1f} (블랙 레벨 포함), 불균일 {dk.std():.1f}"
              + ("  ← 환경광 큼 (조명 끄기 권장)" if dk.std() > 3 else ""))
        for i in range(1, LED_COUNT + 1):
            led(ser, i); flush(cap)
            im = grab_avg(cap)
            np.save(os.path.join(out_dir, f"led{i-1}.npy"), im)
            cv2.imwrite(os.path.join(out_dir, f"led{i-1}.png"), np.clip(im, 0, 255).astype(np.uint8))
            sat = float(np.mean(im >= SAT_LEVEL)) * 100
            led(ser, 0)
            print(f"  LED{i-1} 완료 (최대 {np.percentile(im, 99.9):.0f}{f', 포화 {sat:.1f}%' if sat > 0.1 else ''})")
        json.dump({"exposure": exp_used, "gain": cap.get(cv2.CAP_PROP_GAIN),
                   "gamma": cap.get(cv2.CAP_PROP_GAMMA),
                   "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
                   "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
                   "paper_thickness_mm": PAPER_THICKNESS_MM,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    led(ser, 0)
    ser.close()
    cap.release()
    return go


if __name__ == "__main__":
    # cp949 콘솔에서 ✅ 등 출력 시 죽지 않게, 파이프로 받을 땐 UTF-8
    sys.stdout.reconfigure(encoding=None if sys.stdout.isatty() else 'utf-8', errors='replace')
    sys.stderr.reconfigure(encoding=None if sys.stderr.isatty() else 'utf-8', errors='replace')
    args = sys.argv[1:]
    if "full" in args:                 # 원본 해상도 복원 (균열 등 미세 구조용, 느림)
        DOWNSAMPLE = 1
        args = [a for a in args if a != "full"]
    mode = args[0] if args else ""
    if mode == "ref":
        if capture(REF_DIR, CAM_EXPOSURE, allow_exposure_change=True):
            print("\n기준 촬영 완료. 이제: python ps_run.py obj 이름")
        else:
            print("\n촬영 취소 (ENTER 없이 종료) — 기존 기준 영상 그대로")
    elif mode in ("obj", "judge", "solve"):
        name = args[1] if len(args) > 1 else "test"
        if not os.path.exists(os.path.join(REF_DIR, "meta.json")):
            sys.exit("기준 촬영이 없음 → 먼저 python ps_run.py ref")
        ref_exp = json.load(open(os.path.join(REF_DIR, "meta.json")))["exposure"]
        obj_dir = os.path.join(OBJ_ROOT, name)
        if mode == "obj" and not capture(obj_dir, ref_exp, allow_exposure_change=False):
            sys.exit("촬영 취소")
        if mode in ("obj", "judge"):           # 균열 판정 (빠른 노멀 솔버 + 선 단위 판정) → 판정 그림
            import camtest_and_solver
            camtest_and_solver.run_solve(name, show="--noshow" not in args)
            sys.exit(0)
        print(f"\n[복원] (해상도 1/{DOWNSAMPLE})")
        res = reconstruct(obj_dir, ds=DOWNSAMPLE)
        save_results(res, os.path.join(OUT_ROOT, name))
    else:
        print(__doc__)