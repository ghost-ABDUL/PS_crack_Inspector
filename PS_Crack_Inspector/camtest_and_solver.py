"""
카메라 진단 + 빠른 노멀 전용 솔버 (균열 판별용)

[1] 포맷 확인 (LED 불필요)
    python camtest_and_solver.py fmt
    → MJPG / YUY2 / NV12 각각 실제 해상도·FPS 출력. YUY2가 1280x800 되는지 확인.

[2] 선형성 테스트 (흰 무광 종이 깔고, LED 하나 켜서 노출만 바꿔 촬영)
    python camtest_and_solver.py lin                       (기본: YUY2, LED1)
    python camtest_and_solver.py lin --fourcc MJPG         (비교용)
    python camtest_and_solver.py lin --brightness 16       (블랙 레벨 올려보기)
    python camtest_and_solver.py lin --brightness 16 --contrast 32 --gamma 100   (톤 커브 조절)
    python camtest_and_solver.py lin --led 3 --frames 8
    python camtest_and_solver.py lin --top -2               (가장 긴 노출 지정: 밝은 쪽까지 곡선 측정)
    → 노출을 한 단계 올렸을 때 밝기 배율이 '밝기 구간과 무관하게' 같아야 선형.
      어두운 구간 배율만 다르면 = 어두운 값이 눌림 (그린 무늬가 노멀로 새는 원인)

[2.5] 응답 곡선 보정 (lin 을 brightness 16 으로 한 번 돌린 뒤, 카메라 없이)
    python camtest_and_solver.py resp
    → camera_response.npz 저장. 이후 ps_run.py / solve 가 원본 DN 을 자동으로 선형값으로 변환.
      ※ 카메라 설정(brightness 등)을 바꾸면 lin → resp 다시.

[2.7] 정답 표시 (특징 비교용)
    python camtest_and_solver.py label 이름
    → 왼쪽 드래그=진짜 균열(빨강), 오른쪽 드래그=스티커(파랑), 가운데=지우개, 휠=확대, +/-=붓 크기, s=저장
      ps_out/이름/labels.npy 저장

[3] 빠른 솔버 (기존 ps_ref / ps_obj 데이터 사용, 높이 적분 없음)
    python camtest_and_solver.py solve 이름          (원본 해상도)
    python camtest_and_solver.py solve 이름 ds2      (1/2 해상도)
    → ps_out/이름/fast/ 에 저장
      geom_dev   : 주변 평균 노멀과의 각도 차 (기하 변화 = 진짜 홈·균열)
      overlay    : 빨강=균열, 파랑=균열 아닌 선(스티커·그림·무늬), 회색=신호 부족
      판정 규칙   : 후보 선 = '반사율만 다름' 검정 위반 또는 노멀이 크게 꺾인 픽셀의 중심선
                    선 따라 LINE_AGG_R px 묶어 (노멀 변화/노이즈 >= LINE_Z_THRESH) & (양옆 기울기 차 >= LINE_SIDE_DEG) → 균열
                    (진짜 균열은 조각이 어긋나 노멀이 꺾이고, 스티커·그림은 평평한 면 위 반사율 변화)
      ※ 기준값은 pat_real / pat_sticker (같은 무늬 타일) 두 장으로만 정한 것. 새 샘플로 재확인 필요.
      albedo 누설 지수 : '그린 무늬' 샘플에서 1 근처여야 정상. 크면 카메라가 아직 비선형.

※ 포맷(MJPG→YUY2)이나 brightness를 바꾸면 ps_ref 를 다시 찍어야 함.
"""
import os
import sys
import json
import time
import cv2
import numpy as np

import ps_run
from ps_run import (LED_COUNT, SAT_LEVEL, MIN_REF, MIN_LEDS, SHADOW_FRAC, SHADOW_PRED_FRAC,
                    REF_DIR, OBJ_ROOT, OUT_ROOT, load_geometry, load_stack, down)

# ────────────────────────────────────────────────────────────
# [설정]
# ────────────────────────────────────────────────────────────
LIN_FRAMES     = 8        # 노출당 평균 장수
LIN_STEPS      = 6        # 기준 노출에서 아래로 몇 단계 내려가며 찍을지
LIN_BINS       = [(2, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 240)]
LIN_REF_BIN    = 3        # 기준 구간 (32-64)
LIN_TOL        = 0.10     # 배율이 기준 구간 대비 이 비율 넘게 다르면 비선형
LIN_MIN_EXP    = -6       # 이 카메라는 -7, -8 노출을 제대로 적용 안 함 (측정으로 확인)

SIG_RELIABLE   = 20.0     # 픽셀의 LED 밝기 중앙값(DN)이 이보다 작으면 판정 불가 (노멀 안 지어냄)
GEOM_SIGMA     = 8.0      # 주변 평균 노멀 계산 범위 (px, 원본 해상도 기준)
GEOM_THRESH    = 3.0      # 그림 색 범위용 (도)
PRE_SIGMA      = 1.5      # 노멀 사전 평활 (px): 픽셀 노이즈 감소, 균열 폭(수 px)은 유지
Z_THRESH       = 5.0      # 균열 후보: 노멀 변화가 '같은 신호 세기의 노이즈'의 몇 배 이상
GEOM_MIN       = 2.0      # 균열 후보 최소 각도 차 (도)
MIN_AREA       = 40       # 이보다 작은 씨앗 덩어리(px)는 노이즈로 버림 (원본 해상도 기준)
CHI_THRESH     = 10.0     # 연장 조건: '반사율만 다르다' 검정 위반이 배경 중앙값의 몇 배 이상
CHI_NB_SIGMA   = 8.0      # 검정의 주변 패턴 범위 (px)
GROW_CLOSE     = 5        # 연장 전 끊긴 틈 메움 (px)
EDGE_MARGIN    = 15       # 물체 가장자리 이만큼(px)은 균열 판정 제외 (가장자리 단차가 걸림)

# 선 단위 판정 (후보 선 → 선을 따라 특징을 모아 진짜 균열 / 스티커·무늬 구분)
LINE_MIN_LEN   = 20       # 후보 중심선 최소 길이 (px)
LINE_AGG_R     = 40       # 선을 따라 이 반경(px) 안의 점들을 묶어 판정
LINE_Z_THRESH  = 10.0     # 묶음 중앙값: 노멀 변화/노이즈 최대가 이 이상이면 균열 (6장 검토: 8~12 에서 판정 안 바뀜)
LINE_SIDE_DEG  = 0.0      # 양옆 기울기 차 조건 (0=끔). 조각이 안 어긋난 가는 균열(new_real1)을 놓쳐서 끔
LINE_SIDE_OFF  = 14       # 양옆 표면을 볼 거리 (px)
LINE_MIN_CRACK = 50       # 균열 판정된 중심선 조각이 이보다 짧으면(px) 버림
ALB_EDGE_THRESH = 0.3     # 무늬 경계 판정: log(albedo) 기울기
CONCAVE_SIGMA  = 1.0


# ════════════════════════════════════════════════════════════
# 카메라
# ════════════════════════════════════════════════════════════

def fourcc_str(v):
    v = int(v)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def open_cam(fourcc, exposure, brightness=None, gamma=None, contrast=None):
    cap = cv2.VideoCapture(ps_run.CAMERA_INDEX, cv2.CAP_DSHOW)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ps_run.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ps_run.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_GAIN, ps_run.CAM_GAIN)
    gamma = ps_run.CAM_GAMMA if gamma is None else gamma
    if gamma is not None:
        cap.set(cv2.CAP_PROP_GAMMA, gamma)
    # 드라이버가 마지막 값을 기억하므로 지정 안 하면 ps_run 값으로 항상 되돌림
    cap.set(cv2.CAP_PROP_BRIGHTNESS, ps_run.CAM_BRIGHTNESS if brightness is None else brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, ps_run.CAM_CONTRAST if contrast is None else contrast)
    return cap


def describe(cap):
    ok, f = cap.read()
    h, w = f.shape[:2] if ok else (0, 0)
    return (f"{w}x{h}  FOURCC={fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))}  "
            f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):+.0f}  gain={cap.get(cv2.CAP_PROP_GAIN):.0f}  "
            f"gamma={cap.get(cv2.CAP_PROP_GAMMA):.0f}  brightness={cap.get(cv2.CAP_PROP_BRIGHTNESS):.0f}  "
            f"contrast={cap.get(cv2.CAP_PROP_CONTRAST):.0f}")


def run_fmt():
    for fcc in ["MJPG", "YUY2", "NV12"]:
        cap = open_cam(fcc, -6)
        for _ in range(5):
            cap.read()
        t0, ok = time.time(), 0
        for _ in range(20):
            ok += cap.read()[0]
        fps = ok / (time.time() - t0)
        print(f"[{fcc} 요청] {describe(cap)}  측정 FPS {fps:.1f}")
        cap.release()
    print(f"\n→ 요청 해상도 {ps_run.FRAME_WIDTH}x{ps_run.FRAME_HEIGHT} 그대로 나오는 무압축 포맷(YUY2/NV12)을 쓰세요.")
    print("  해상도가 바뀌면 camera_intrinsics / led_positions 캘리브레이션도 다시 해야 함.")


def open_serial():
    import serial
    ser = serial.Serial(ps_run.SERIAL_PORT, ps_run.BAUD_RATE, timeout=1.0)
    t = time.time() + 5.0
    while time.time() < t:
        if ser.in_waiting and "READY" in ser.readline().decode(errors='ignore'):
            break
    ser.reset_input_buffer()
    return ser


def run_lin(fourcc="YUY2", led_no=1, brightness=None, frames=LIN_FRAMES, gamma=None, contrast=None, top=None):
    ref_meta = os.path.join(REF_DIR, "meta.json")
    if top is None:
        top = int(json.load(open(ref_meta))["exposure"]) if os.path.exists(ref_meta) else ps_run.CAM_EXPOSURE
    exps = list(range(max(top - LIN_STEPS, LIN_MIN_EXP), top + 1))
    ser = open_serial()
    cap = open_cam(fourcc, exps[0], brightness, gamma, contrast)
    print(f"카메라: {describe(cap)}")
    print(f"LED{led_no - 1}, 노출 {exps}, {frames}장 평균\n")

    imgs, darks = {}, {}
    for e in exps:
        cap.set(cv2.CAP_PROP_EXPOSURE, e)
        ps_run.led(ser, 0)
        for _ in range(4):
            cap.read()
        darks[e] = ps_run.grab_avg(cap, frames).astype(np.float64)
        ps_run.led(ser, led_no)
        for _ in range(4):
            cap.read()
        imgs[e] = ps_run.grab_avg(cap, frames).astype(np.float64)
        ps_run.led(ser, 0)
        dk = darks[e]
        print(f"  노출 {e:+d}: 암전 평균 {dk.mean():.2f} (p99 {np.percentile(dk, 99):.1f})  "
              f"LED 영상 p50 {np.percentile(imgs[e], 50):.1f} p99.9 {np.percentile(imgs[e], 99.9):.1f}")
    ps_run.led(ser, 0)
    ser.close()
    cam_set = dict(brightness=cap.get(cv2.CAP_PROP_BRIGHTNESS), contrast=cap.get(cv2.CAP_PROP_CONTRAST))
    cap.release()

    dk_floor = np.mean([darks[e].mean() for e in exps])
    zero_frac = (darks[exps[0]] <= 0.5).mean() * 100
    print(f"\n[블랙 레벨] 가장 짧은 노출 암전: 평균 {darks[exps[0]].mean():.2f} DN, 0인 픽셀 {zero_frac:.0f}%"
          + ("  ← 0에 붙어 있음 = 블랙 클리핑. --brightness 올려서 다시" if zero_frac > 5 else "  (노이즈 바닥 있음)"))
    amb = darks[exps[-1]].mean() - darks[exps[0]].mean()
    if amb > 3:
        print(f"[주변광] 노출이 길수록 암전이 밝아짐 (+{amb:.1f} DN) → 방 불/창문 빛 들어옴. 끄고 다시 측정 권장")
    p50 = [np.percentile(imgs[e] - darks[e], 50) for e in exps]
    if any(b <= a for a, b in zip(p50[:-1], p50[1:])):
        print(f"[노출] 노출을 늘렸는데 어두워진 단계 있음 {np.round(p50, 1).tolist()} → 그 노출값은 카메라가 무시함, 쓰지 말 것")

    out = os.path.join(OUT_ROOT, "_camtest")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "lin_raw_latest.npz"), exps=np.array(exps),
             imgs=np.array([imgs[e] for e in exps], np.float32),
             darks=np.array([darks[e] for e in exps], np.float32), **cam_set)

    rows, bad = ratio_table(imgs, darks, exps)
    json.dump(dict(fourcc=fourcc, led=led_no, brightness=brightness, gamma=gamma, contrast=contrast,
                   dark_floor=dk_floor,
                   bins=LIN_BINS, rows=rows, nonlinear=[(e, list(b), r) for e, b, r in bad]),
              open(os.path.join(out, f"lin_{fourcc}_b{brightness}_g{gamma}_c{contrast}.json"), "w"), indent=1)
    print(f"\n원본 데이터 저장: {out}\\lin_raw_latest.npz  →  응답 곡선 보정: python camtest_and_solver.py resp")


def ratio_table(imgs, darks, exps, lin=None):
    """노출 한 단계 올렸을 때 밝기 배율을 '낮은 노출 밝기(원본 DN)' 구간별로 출력.
    구간은 평활한 영상으로 나눔 (노이즈로 어둡게 뽑힌 픽셀이 배율을 부풀리는 편향 방지).
    lin: 원본 DN → 선형값 변환 (응답 곡선 보정 확인용)"""
    lin = lin or (lambda x: x)
    print(f"\n[노출 +1 배율, 밝기 구간별]  선형이면 한 줄 안에서 값이 모두 같아야 함 (기준: {LIN_BINS[LIN_REF_BIN]} 구간)")
    hdr = "  노출쌍     " + "".join(f"{f'{a}-{b}':>10}" for a, b in LIN_BINS)
    print(hdr)
    bad = []
    rows = []
    for e0, e1 in zip(exps[:-1], exps[1:]):
        lo_raw = imgs[e0] - darks[e0]
        lo = lin(imgs[e0]) - lin(darks[e0])
        hi = lin(imgs[e1]) - lin(darks[e1])
        ok = imgs[e1] < SAT_LEVEL
        lo_s = cv2.GaussianBlur(lo_raw.astype(np.float32), (0, 0), 3)
        ratios = []
        for a, b in LIN_BINS:
            m = ok & (lo_s >= a) & (lo_s < b)
            ratios.append(hi[m].mean() / lo[m].mean() if m.sum() > 200 else np.nan)
        ratios = np.array(ratios)
        fin = np.where(np.isfinite(ratios))[0]
        if len(fin) == 0:
            continue
        refk = fin[np.argmin(np.abs(fin - LIN_REF_BIN))]   # 중간 밝기 구간 기준: 아래=토(눌림), 위=숄더(압축)
        rel = ratios / ratios[refk]
        rows.append(dict(exp=[e0, e1], ratio=ratios.tolist()))
        print(f"  {e0:+d}→{e1:+d}  " + "".join(f"{r:10.2f}" if np.isfinite(r) else f"{'-':>10}" for r in ratios)
              + f"   (기준 배율 {ratios[refk]:.2f})")
        for k in fin:
            if abs(rel[k] - 1) > LIN_TOL:
                bad.append((e0, LIN_BINS[k], rel[k]))

    print()
    if bad:
        print("❌ 비선형: 아래 구간에서 배율이 밝은 구간과 다름")
        for e0, (a, b), r in bad:
            kind = "어두운 쪽 눌림" if b <= LIN_BINS[LIN_REF_BIN][0] else "밝은 쪽 압축"
            print(f"   노출 {e0:+d}, 밝기 {a}~{b} DN: 상대 배율 {r:.2f}  ({kind})")
        print("   → 어두운 쪽 눌림: --brightness 올려서 다시. 밝은 쪽 압축: 촬영 시 최대 밝기를 그 구간 아래로.")
    else:
        print("✅ 선형: 밝기 구간과 무관하게 배율 일정.")
    return rows, bad


# ════════════════════════════════════════════════════════════
# 응답 곡선 추정 (노출 단계 데이터 → 원본 DN 을 선형값으로 펴는 곡선)
# ════════════════════════════════════════════════════════════

RESP_KNOTS  = np.arange(0, 257, 16, dtype=np.float64)   # 0,16,...,256 에서 곡선값 추정
RESP_SMOOTH = 3.0      # 곡선 매끄러움 정규화
RESP_IDENT  = 0.05     # 직선(무보정)에 가깝게 끄는 약한 정규화 (지수 모호성 고정)


def resp_curve(u):
    """증가량 log → 단조 증가 곡선 (0에서 0, 128에서 128로 고정)"""
    g = np.concatenate([[0.0], np.cumsum(np.exp(u))])
    g128 = np.interp(128.0, RESP_KNOTS, g)
    return g * 128.0 / g128


def run_resp(path=None):
    from scipy.optimize import least_squares
    path = path or os.path.join(OUT_ROOT, "_camtest", "lin_raw_latest.npz")
    z = np.load(path)
    exps, I, Dk = [int(e) for e in z["exps"]], z["imgs"].astype(np.float64), z["darks"].astype(np.float64)
    if "brightness" not in z:
        sys.exit("이 데이터엔 카메라 설정 기록이 없음 → lin 을 다시 한 번 돌린 뒤 resp")
    print(f"데이터: {path}  노출 {exps}  brightness={float(z['brightness']):.0f} contrast={float(z['contrast']):.0f}")

    # 표본: 공간 평활 (종이는 매끄러움) 후 8px 격자
    Is = np.array([cv2.GaussianBlur(im.astype(np.float32), (0, 0), 2) for im in I]).astype(np.float64)
    Ds = np.array([cv2.GaussianBlur(im.astype(np.float32), (0, 0), 2) for im in Dk]).astype(np.float64)
    Is, Ds = Is[:, ::8, ::8].reshape(len(exps), -1), Ds[:, ::8, ::8].reshape(len(exps), -1)
    pairs = []
    for k in range(len(exps) - 1):
        m = (Is[k + 1] < SAT_LEVEL) & (Is[k] - Ds[k] > 3) & (Is[k + 1] - Ds[k + 1] > 3)
        pairs.append((k, np.where(m)[0]))
        print(f"  {exps[k]:+d}→{exps[k+1]:+d}: 표본 {m.sum()}")
    nk = len(RESP_KNOTS) - 1

    def resid(x):
        g = resp_curve(x[:nk])
        lr = x[nk:]
        f = lambda v: np.interp(v, RESP_KNOTS, g)
        out = []
        for (k, idx), l in zip(pairs, lr):
            a = f(Is[k, idx]) - f(Ds[k, idx])
            b = f(Is[k + 1, idx]) - f(Ds[k + 1, idx])
            out.append((b - np.exp(l) * a) / np.maximum(b + np.exp(l) * a, 1.0) * 2)   # 상대 오차
        step = np.diff(g)
        out.append(RESP_SMOOTH * np.diff(np.log(step)) / np.sqrt(nk))
        out.append(RESP_IDENT * (g[1:-1] - RESP_KNOTS[1:-1]) / 128.0)
        return np.concatenate(out)

    x0 = np.concatenate([np.full(nk, np.log(16.0)), np.log(np.full(len(pairs), 1.7))])
    sol = least_squares(resid, x0, method='trf', x_scale='jac')
    g = resp_curve(sol.x[:nk])
    r = np.exp(sol.x[nk:])
    rms_before = np.sqrt(np.mean(resid(np.concatenate([np.full(nk, np.log(16.0)), sol.x[nk:]]))[:sum(len(i) for _, i in pairs)] ** 2))
    rms_after = np.sqrt(np.mean(sol.fun[:sum(len(i) for _, i in pairs)] ** 2))
    print(f"\n노출 배율 추정: {np.round(r, 3).tolist()}")
    print("  ※ 노출 배율을 정확히 모르면 곡선은 '정답의 거듭제곱'까지만 정해짐 (합성 데이터로 확인).\n"
          "    반사율은 모든 LED에 같은 지수로 걸려 비율에서 지워지므로, 무늬→가짜 노멀 누설은 이것으로 제거됨.\n"
          "    남는 건 노멀 각도의 매끄러운 왜곡뿐 (무늬/균열 구분엔 영향 없음, 정량 형상엔 약간)")
    print(f"상대 오차 RMS: 보정 전 {rms_before*100:.2f}%  →  보정 후 {rms_after*100:.2f}%")
    print("응답 곡선 (원본 DN → 선형값):")
    print("  " + "  ".join(f"{int(k)}→{v:.0f}" for k, v in zip(RESP_KNOTS, g)))

    out_path = os.path.join(ps_run.base_dir, "camera_response.npz")
    np.savez(out_path, knots=RESP_KNOTS, g=g, exps=np.array(exps), source=path,
             brightness=float(z["brightness"]), contrast=float(z["contrast"]))
    print(f"\n✅ 저장: {out_path}  (이후 ps_run / solve 가 자동으로 적용)")

    lin = lambda v: np.interp(v, RESP_KNOTS, g)
    print("\n[확인] 보정 후 배율표 — 한 줄이 평평해졌으면 성공")
    ratio_table({e: I[k] for k, e in enumerate(exps)}, {e: Dk[k] for k, e in enumerate(exps)}, exps, lin)

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any('Malgun' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Malgun Gothic'
    xs = np.linspace(0, 255, 256)
    plt.figure(figsize=(5, 5))
    plt.plot(xs, xs, 'k--', lw=0.8, label='무보정 (직선)')
    plt.plot(xs, lin(xs), lw=2, label='추정 응답 곡선')
    plt.xlabel('카메라 출력 DN'); plt.ylabel('선형값'); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(os.path.join(OUT_ROOT, "_camtest", "camera_response.png"), dpi=110)


# ════════════════════════════════════════════════════════════
# 빠른 솔버 (노멀만, 높이 적분·반복 없음)
# ════════════════════════════════════════════════════════════

def gblur_masked(x, m, s):
    m = m.astype(np.float32)
    num = cv2.GaussianBlur((x * m).astype(np.float32), (0, 0), s)
    den = cv2.GaussianBlur(m, (0, 0), s)
    return num / np.maximum(den, 1e-6)


def solve_normals(obj_dir, ds=1):
    t0 = time.time()
    geo = load_geometry()
    K, dist, n, d, Lpos = geo["K"], geo["dist"], geo["n"], geo["d"], geo["L"]
    ref_meta = json.load(open(os.path.join(REF_DIR, "meta.json")))
    d_paper = d - ref_meta.get("paper_thickness_mm", ps_run.PAPER_THICKNESS_MM)

    Rd, Rimg = load_stack(REF_DIR)
    Od, Oimg = load_stack(obj_dir)
    sat = np.array([down(im, ds) >= ps_run.sat_level(obj_dir) for im in Oimg])
    R = np.array([down(im - Rd, ds) for im in Rimg]).astype(np.float32)
    O = np.array([down(im - Od, ds) for im in Oimg]).astype(np.float32)
    nL, H, W = R.shape

    # 픽셀 광선 → 종이면 위 점 (물체 두께는 무시: 저주파 노멀 편향만 생기고 국소 균열 판별엔 영향 적음)
    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.column_stack([((xs + 0.5) * ds - 0.5).ravel(),
                           ((ys + 0.5) * ds - 0.5).ravel()]).reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    V = np.column_stack([und, np.ones(len(und))])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    V = V.reshape(H, W, 3)
    X = V * (d_paper / (V @ n))[..., None]

    Ls = np.empty((nL, H, W, 3), np.float32)
    M = np.zeros((nL, H, W), np.float32)
    inbeam = (R > MIN_REF) & ~sat
    for i in range(nL):
        D = Lpos[i] - X
        l = D / np.linalg.norm(D, axis=-1, keepdims=True)
        Ls[i] = l
        cosp = l @ (-n)
        M[i] = np.where(inbeam[i], O[i] / np.maximum(R[i], 1e-6) * cosp, 0.0)

    # 신호 부족 픽셀: LED 밝기 중앙값이 낮으면 비선형·양자화 영역 → 노멀을 지어내지 않음
    with np.errstate(invalid='ignore'):
        med_sig = np.nanmedian(np.where(inbeam, O, np.nan), axis=0)
    reliable = np.nan_to_num(med_sig, nan=0.0) >= SIG_RELIABLE

    # '반사율만 다르다' 검정: 픽셀의 LED별 밝기가 주변 패턴의 단순 배율로 설명되는지.
    # 노멀이 판정 불가한 어두운 픽셀(그림자 진 균열)에서도 계산 가능 → 균열 연장에 사용
    Nb = np.array([cv2.GaussianBlur(O[i], (0, 0), CHI_NB_SIGMA / ds) for i in range(nL)])
    wv = inbeam / (1.0 + 0.15 * np.maximum(O, 0))          # 판독 + 산탄 노이즈 근사
    a = (wv * O * Nb).sum(0) / np.maximum((wv * Nb * Nb).sum(0), 1e-9)
    chi = (wv * (O - a * Nb) ** 2).sum(0) / np.maximum(inbeam.sum(0) - 1, 1)
    chi_ok = (inbeam.sum(0) >= 4) & (np.where(inbeam, O, 0).max(0) >= 15)
    base = chi_ok & reliable                                 # 기준: 신호 충분한 물체 픽셀 (배경 노이즈 제외)
    zchi = np.where(chi_ok, chi / max(np.median(chi[base]), 1e-9) if base.any() else 0, 0)

    Wt = inbeam * np.sqrt(np.maximum(R, 0))

    def solve_b(Wt):
        A = np.einsum('khw,khwi,khwj->hwij', Wt, Ls, Ls).astype(np.float64) + np.eye(3) * 1e-9
        c = np.einsum('khw,khw,khwi->hwi', Wt, M, Ls).astype(np.float64)
        return np.linalg.solve(A, c[..., None])[..., 0]

    with np.errstate(invalid='ignore'):
        med_m = np.nanmedian(np.where(Wt > 0, M, np.nan), axis=0)
        Wt = np.where((Wt > 0) & (M < SHADOW_FRAC * med_m[None]), 0.0, Wt)   # 드리운 그림자
    b = solve_b(Wt)
    pred = np.einsum('khwi,hwi->khw', Ls, b)
    shadow2 = (Wt > 0) & (M < SHADOW_PRED_FRAC * pred)
    keep = ((Wt > 0) & ~shadow2).sum(axis=0) >= MIN_LEDS
    Wt = np.where(shadow2 & keep[None], 0.0, Wt)
    b = solve_b(Wt)

    nvalid = (Wt > 0).sum(axis=0)
    rho = np.linalg.norm(b, axis=-1)
    N = b / np.maximum(rho[..., None], 1e-12)
    N[np.einsum('hwi,hwi->hw', N, V) > 0] *= -1
    good = reliable & (nvalid >= MIN_LEDS) & (rho > 0)
    N[~good] = 0
    print(f"  노멀 계산 {time.time() - t0:.1f}s  ({W}x{H}, 판정 가능 {good.mean()*100:.0f}%)")
    return dict(N=N.astype(np.float32), rho=rho.astype(np.float32), good=good,
                reliable=reliable, inbeam_any=inbeam.any(axis=0), nvalid=nvalid, ds=ds,
                sig=np.nan_to_num(med_sig, nan=0.0).astype(np.float32),
                zchi=zchi.astype(np.float32), chi_ok=chi_ok,
                mm_per_px=float(np.median(np.linalg.norm(X[:, 1:] - X[:, :-1], axis=-1))))


def crack_maps(res):
    N, rho, g, ds = res["N"], res["rho"], res["good"], res["ds"]
    s = GEOM_SIGMA / ds

    # 기하 변화: 주변 평균 노멀과의 각도 차 (반사율과 무관 — 카메라가 선형일 때)
    Np = np.stack([gblur_masked(N[..., k], g, PRE_SIGMA / ds) for k in range(3)], -1)
    Np /= np.maximum(np.linalg.norm(Np, axis=-1, keepdims=True), 1e-9)
    Nsm = np.stack([gblur_masked(Np[..., k], g, s) for k in range(3)], -1)
    Nsm /= np.maximum(np.linalg.norm(Nsm, axis=-1, keepdims=True), 1e-9)
    dev = np.degrees(np.arccos(np.clip((Np * Nsm).sum(-1), -1, 1)))
    dev[~g] = 0

    # 노이즈 모델: 신호 세기 구간별 dev 중앙값 (균열은 소수라 중앙값은 노이즈 수준).
    # 어두운 무늬 픽셀은 신호가 약해 노멀이 더 흔들림 → 같은 세기의 노이즈 대비 몇 배인지로 판정
    sig = res["sig"]
    edges = np.geomspace(max(SIG_RELIABLE, 1.0), max(sig[g].max(), SIG_RELIABLE * 2) if g.any() else 256, 12)
    cen, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = g & (sig >= a) & (sig < b)
        if k.sum() > 200:
            cen.append(np.sqrt(a * b)); med.append(np.median(dev[k]))
    if cen:
        med = np.minimum.accumulate(np.array(med))          # 신호가 셀수록 노이즈는 작거나 같음
        noise = np.interp(np.log(np.maximum(sig, 1e-3)), np.log(cen), med)
    else:
        noise = np.full_like(dev, np.median(dev[g]) if g.any() else 1.0)
    z = np.where(g, dev / np.maximum(noise, 1e-3), 0)

    # 오목도 (ps_run 과 같은 정의)
    nzc = np.where(-N[..., 2] < 0.05, 0.05, -N[..., 2])
    pc = cv2.GaussianBlur((N[..., 0] / nzc).astype(np.float32), (0, 0), CONCAVE_SIGMA)
    qc = cv2.GaussianBlur((N[..., 1] / nzc).astype(np.float32), (0, 0), CONCAVE_SIGMA)
    concave = np.where(g, np.maximum(-(np.gradient(pc, axis=1) + np.gradient(qc, axis=0)), 0), 0)

    # 반사율 경계 (무늬)
    ref_rho = np.percentile(rho[g], 99) if g.any() else 1.0
    alb = np.clip(rho / ref_rho, 1e-3, None)
    la = cv2.GaussianBlur(np.log(alb).astype(np.float32), (0, 0), 1.0)
    aedge = np.hypot(np.gradient(la, axis=1), np.gradient(la, axis=0)) * ds
    aedge[~res["inbeam_any"]] = 0

    # 물체 안쪽만 (가장자리 단차 제외)
    obj = cv2.morphologyEx(res["chi_ok"].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    k = 2 * int(EDGE_MARGIN / ds) + 1
    interior = cv2.erode(obj, np.ones((k, k), np.uint8)) > 0

    # 씨앗: 노멀이 확실히 판정되는 픽셀에서 노이즈보다 크게 꺾인 곳 (그린 무늬는 여기서 걸러짐)
    seed = g & (z > Z_THRESH) & (dev > GEOM_MIN) & interior
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < MIN_AREA / ds ** 2
    small[0] = False
    seed[small[lab]] = False
    # 연장: 씨앗과 이어진 '반사율만 다르다' 검정 위반 픽셀 (그림자 져서 노멀 판정 불가한 균열 구간)
    weak = (res["zchi"] > CHI_THRESH) & interior
    if GROW_CLOSE:
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROW_CLOSE, GROW_CLOSE))
        weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, el) > 0
    n_lab, lab = cv2.connectedComponents((seed | weak).astype(np.uint8), connectivity=8)
    keep = np.zeros(n_lab, bool)
    keep[np.unique(lab[seed])] = True
    keep[0] = False
    crack = keep[lab] & (seed | weak)
    pattern = (aedge > ALB_EDGE_THRESH) & ~cv2.dilate(crack.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    unknown = res["inbeam_any"] & ~g & ~crack

    # 무늬 픽셀(주변보다 어두운 곳) 중 균열로 판정된 비율: 그린 균열 샘플에선 0 근처여야 함
    loc = gblur_masked(alb, g, 4 * s)
    dark = g & (alb < 0.6 * loc)
    dark_crack = crack[dark].mean() * 100 if dark.sum() > 50 else np.nan
    return dict(dev=dev, z=z, noise=noise, seed=seed, interior=interior, concave=concave, alb=np.clip(alb, 0, 1), aedge=aedge,
                crack=crack, pattern=pattern, unknown=unknown, dark_crack=dark_crack,
                n_dark=int(dark.sum()), noise_curve=(cen, list(med)))


def classify_lines(res, mp):
    """후보 선을 찾고, 선을 따라 묶은 특징으로 진짜 균열인지 판정.
    진짜 균열: 조각이 어긋나 선을 따라 노멀이 꺾이고 양옆 표면 기울기가 다름.
    스티커/그린 선/무늬 경계: 평평한 면 위의 반사율 변화라 둘 다 작음."""
    from skimage.morphology import skeletonize
    from scipy.spatial import cKDTree
    g, ds, N, z = res["good"], res["ds"], res["N"], mp["z"]
    interior = mp["interior"]
    H, W = g.shape

    # 1) 후보: '반사율만 다름' 위반 또는 노멀이 크게 꺾인 픽셀
    cand = interior & ((res["zchi"] > CHI_THRESH) | (g & (z > Z_THRESH) & (mp["dev"] > GEOM_MIN)))
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROW_CLOSE, GROW_CLOSE))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, el) > 0
    sk = skeletonize(cand)
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    small = st[:, cv2.CC_STAT_AREA] < LINE_MIN_LEN / ds
    small[0] = True
    sk &= ~small[lab]
    ys, xs = np.nonzero(sk)
    comp = lab[ys, xs]
    if len(ys) == 0:
        return dict(cand=cand, crack=np.zeros_like(g), line_other=np.zeros_like(g), pts=None)

    # 2) 점별 특징
    r = max(int(round(5 / ds)), 1)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    zmax = cv2.dilate(np.where(g, z, 0).astype(np.float32), disk)[ys, xs]
    # 선 방향: 후보 영역의 구조 텐서 → 법선 방향
    m = cv2.GaussianBlur(cand.astype(np.float32), (0, 0), 3 / ds)
    Ix, Iy = cv2.Sobel(m, cv2.CV_32F, 1, 0), cv2.Sobel(m, cv2.CV_32F, 0, 1)
    Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), 5 / ds)
    Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), 5 / ds)
    Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), 5 / ds)
    th = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)[ys, xs]
    nx, ny = np.cos(th), np.sin(th)
    # 양옆 표면 평균 노멀 (판정 가능 픽셀만)
    wgt = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 4 / ds)
    Nsm = np.stack([cv2.GaussianBlur((N[..., k] * g).astype(np.float32), (0, 0), 4 / ds) for k in range(3)], -1)
    off = LINE_SIDE_OFF / ds
    side = np.full(len(ys), np.nan)
    pa = (np.clip(np.round(ys - off * ny), 0, H - 1).astype(int), np.clip(np.round(xs - off * nx), 0, W - 1).astype(int))
    pb = (np.clip(np.round(ys + off * ny), 0, H - 1).astype(int), np.clip(np.round(xs + off * nx), 0, W - 1).astype(int))
    ok = (wgt[pa] > 0.3) & (wgt[pb] > 0.3)
    a = Nsm[pa] / np.maximum(np.linalg.norm(Nsm[pa], axis=-1, keepdims=True), 1e-9)
    b = Nsm[pb] / np.maximum(np.linalg.norm(Nsm[pb], axis=-1, keepdims=True), 1e-9)
    side[ok] = np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))[ok]

    # 3) 같은 선 위 반경 안의 점들 묶어 중앙값
    tree = cKDTree(np.column_stack([ys, xs]))
    nbrs = tree.query_ball_point(np.column_stack([ys, xs]), LINE_AGG_R / ds)
    zagg = np.empty(len(ys)); sagg = np.empty(len(ys))
    for i, nb in enumerate(nbrs):
        nb = np.asarray(nb)
        nb = nb[comp[nb] == comp[i]]
        zagg[i] = np.median(zmax[nb])
        sv = side[nb]; sv = sv[np.isfinite(sv)]
        sagg[i] = np.median(sv) if len(sv) >= 3 else np.nan
    is_crack = (zagg >= LINE_Z_THRESH) & ~(np.isfinite(sagg) & (sagg < LINE_SIDE_DEG))
    # 균열로 판정된 중심선 조각이 짧으면 버림 (진짜 균열은 길게 이어짐)
    ck = np.zeros((H, W), np.uint8)
    ck[ys[is_crack], xs[is_crack]] = 1
    n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(ck, connectivity=8)
    short = st_c[:, cv2.CC_STAT_AREA] < LINE_MIN_CRACK / ds
    short[0] = True
    is_crack &= ~short[lab_c[ys, xs]]

    # 4) 후보 픽셀을 가장 가까운 중심선 점의 판정으로 칠함
    idx_img = np.full((H, W), -1, np.int64)
    idx_img[ys, xs] = np.arange(len(ys))
    _, nearest = cv2.distanceTransformWithLabels((~sk).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    # DIST_LABEL_PIXEL 라벨 → 해당 0픽셀(중심선 점) 좌표
    lab_ys = np.zeros(nearest.max() + 1, np.int64); lab_xs = np.zeros_like(lab_ys)
    zl = nearest[sk]; lab_ys[zl] = ys; lab_xs[zl] = xs
    pt = idx_img[lab_ys[nearest], lab_xs[nearest]]
    crack = cand & (pt >= 0) & is_crack[np.maximum(pt, 0)]
    other = cand & ~crack
    return dict(cand=cand, crack=crack, line_other=other,
                pts=dict(y=ys, x=xs, zmax=zmax, side=side, zagg=zagg, sagg=sagg, is_crack=is_crack))


def save_fast(res, mp, lines, out_dir, title, show=True):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any('Malgun' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    os.makedirs(out_dir, exist_ok=True)
    N, g = res["N"], res["good"]
    np.save(os.path.join(out_dir, "normal.npy"), N)
    np.save(os.path.join(out_dir, "geom_dev_deg.npy"), mp["dev"].astype(np.float32))
    np.save(os.path.join(out_dir, "geom_z.npy"), mp["z"].astype(np.float32))
    np.save(os.path.join(out_dir, "concavity.npy"), mp["concave"].astype(np.float32))
    np.save(os.path.join(out_dir, "crack_mask.npy"), lines["crack"])

    nm = ((N * np.array([1, -1, -1]) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    nm[~g] = 0
    cv2.imwrite(os.path.join(out_dir, "normal.png"), cv2.cvtColor(nm, cv2.COLOR_RGB2BGR))

    base = (mp["alb"] * 255).astype(np.uint8)
    ov = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    ov[mp["unknown"] & ~lines["cand"]] = (110, 110, 110)
    ov[lines["line_other"]] = (255, 120, 0)
    ov[lines["crack"]] = (0, 0, 255)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), ov)

    pts = lines["pts"]
    n_crack = int(pts["is_crack"].sum()) if pts is not None else 0
    n_line = len(pts["y"]) if pts is not None else 0
    leak_txt = (f"후보 선 {n_line * res['ds']} px 중 균열 판정 {n_crack * res['ds']} px  "
                f"(선 따라 {LINE_AGG_R}px 묶음: 노멀 변화 {LINE_Z_THRESH:g}배 이상 & 양옆 기울기 차 {LINE_SIDE_DEG:g}° 이상)")
    cen, med = mp["noise_curve"]
    print("  노이즈 수준 (신호 DN: 각도 중앙값°): " + "  ".join(f"{c:.0f}:{m:.2f}" for c, m in zip(cen, med)))
    txt = (f"{title}  |  균열 {'있음' if n_crack else '없음'}  |  판정 불가 {mp['unknown'].mean()*100:.1f}%\n"
           + leak_txt)
    print("\n" + txt)

    dmax = max(np.percentile(mp["dev"][g], 99.5), GEOM_THRESH * 2) if g.any() else 10
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    ax[0, 0].imshow(nm); ax[0, 0].set_title("노멀 맵 (검정 = 판정 불가)")
    ax[0, 1].imshow(mp["alb"], cmap='gray', vmin=0, vmax=1); ax[0, 1].set_title("반사율")
    ax[0, 2].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
    ax[0, 2].set_title("빨강=균열  파랑=균열 아닌 선(스티커·그림·무늬)  회색=신호 부족")
    im = ax[1, 0].imshow(mp["z"], cmap='inferno', vmin=0, vmax=Z_THRESH * 2)
    ax[1, 0].set_title(f"노멀 변화 / 노이즈 (기준 {Z_THRESH:g}배)"); fig.colorbar(im, ax=ax[1, 0], fraction=0.03)
    ax[1, 1].imshow(mp["alb"], cmap='gray', vmin=0, vmax=1)
    if pts is not None and n_line:
        sc = ax[1, 1].scatter(pts["x"], pts["y"], c=np.minimum(pts["zagg"] / LINE_Z_THRESH, 2), s=1, cmap='coolwarm', vmin=0, vmax=2)
        fig.colorbar(sc, ax=ax[1, 1], fraction=0.03)
    ax[1, 1].set_title("후보 선의 균열 점수 (1 이상 = 노멀 기준 통과)")
    im = ax[1, 2].imshow(res["nvalid"], cmap='magma', vmin=0, vmax=LED_COUNT)
    ax[1, 2].set_title("유효 LED 수"); fig.colorbar(im, ax=ax[1, 2], fraction=0.03)
    for a in ax.ravel():
        a.axis('off')
    fig.suptitle(txt, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "result_fast.png"), dpi=120)
    print(f"✅ 저장: {out_dir}")
    if show:
        plt.show()


def show_verdict(res, lines, obj_dir, out_dir, name, show=True):
    """최종 판정 그림: 촬영 사진 위에 균열 판정 부분 표시 + '균열이 있습니다 / 무늬입니다'"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    if any('Malgun' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    ds = res["ds"]
    photo = np.mean([np.load(os.path.join(obj_dir, f"led{i}.npy")) for i in range(LED_COUNT)], axis=0)
    photo = down(photo, ds)
    photo = np.clip(photo / np.percentile(photo, 99.5), 0, 1)
    rgb = np.dstack([photo] * 3)

    crack = lines["crack"]
    pts = lines["pts"]
    mmpp = res["mm_per_px"]
    regions = []
    if crack.any():
        thick = cv2.dilate(crack.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        rgb[thick] = rgb[thick] * 0.3 + np.array([1.0, 0.1, 0.1]) * 0.7
        # 가까운 균열 조각끼리 묶어서 영역 단위로 표시
        k = int(30 / ds) * 2 + 1
        grp = cv2.dilate(crack.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(grp, connectivity=8)
        sk_crack = np.zeros_like(crack)
        sk_crack[pts["y"][pts["is_crack"]], pts["x"][pts["is_crack"]]] = True
        for i in range(1, n):
            length = sk_crack[lab == i].sum() * mmpp
            if length <= 0:
                continue
            x, y, w, h = st[i, :4]
            regions.append((x, y, w, h, length))
        regions.sort(key=lambda r: -r[4])

    total = sum(r[4] for r in regions)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.imshow(rgb)
    if regions:
        for j, (x, y, w, h, length) in enumerate(regions, 1):
            pad = 12 / ds
            ax.add_patch(Rectangle((x - pad, y - pad), w + 2 * pad, h + 2 * pad,
                                   fill=False, ec='yellow', lw=2))
            ax.text(x - pad, y - pad - 6 / ds, f"균열 {j}  ({length:.0f} mm)", color='yellow',
                    fontsize=11, fontweight='bold', va='bottom',
                    bbox=dict(facecolor='black', alpha=0.6, pad=2, ec='none'))
        title, color = f"균열이 있습니다  ({len(regions)}곳, 총 길이 약 {total:.0f} mm)", "#d62728"
    else:
        title, color = "균열 없음 — 무늬입니다", "#2ca02c"
    ax.set_title(f"{name}:  {title}", fontsize=16, fontweight='bold', color=color)
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(out_dir, "verdict.png")
    plt.savefig(path, dpi=110)
    print(f"\n판정: {title}")
    print(f"판정 그림: {path}")
    if show:
        plt.show()
    plt.close(fig)


def run_solve(name, ds=1, show=True):
    obj_dir = os.path.join(OBJ_ROOT, name)
    if not os.path.exists(os.path.join(obj_dir, "led0.npy")):
        sys.exit(f"촬영 데이터 없음: {obj_dir}")
    t0 = time.time()
    print(f"[빠른 복원] {name} (해상도 1/{ds})")
    res = solve_normals(obj_dir, ds)
    mp = crack_maps(res)
    lines = classify_lines(res, mp)
    print(f"  전체 {time.time() - t0:.1f}s")
    out_dir = os.path.join(OUT_ROOT, name, "fast")
    save_fast(res, mp, lines, out_dir, name, show=False)
    show_verdict(res, lines, obj_dir, out_dir, name, show)


# ════════════════════════════════════════════════════════════
# 정답 표시 (특징 비교용)
# ════════════════════════════════════════════════════════════

def run_label(name):
    """촬영 영상 위에 마우스로 정답 표시 → ps_out/이름/labels.npy (0=없음, 1=진짜 균열, 2=스티커)"""
    obj_dir = os.path.join(OBJ_ROOT, name)
    if not os.path.exists(os.path.join(obj_dir, "led0.npy")):
        names = sorted(d for d in os.listdir(OBJ_ROOT) if os.path.exists(os.path.join(OBJ_ROOT, d, "led0.npy")))
        sys.exit(f"촬영 데이터 없음: {name}\n있는 이름: {', '.join(names)}")
    imgs = [np.load(os.path.join(obj_dir, f"led{i}.npy")) for i in range(LED_COUNT)]
    base = np.mean(imgs, axis=0)
    base = np.clip(base / np.percentile(base, 99.5) * 255, 0, 255).astype(np.uint8)
    H, W = base.shape
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    lab_path = os.path.join(out_dir, "labels.npy")
    lab = np.load(lab_path) if os.path.exists(lab_path) else np.zeros((H, W), np.uint8)
    st = dict(r=4, down=0, zoom=1.0, cx=W / 2, cy=H / 2)

    def view_to_img(x, y):
        z = st["zoom"]
        return int(st["cx"] + (x - W / 2) / z), int(st["cy"] + (y - H / 2) / z)

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN: st["down"] = 1
        elif ev == cv2.EVENT_RBUTTONDOWN: st["down"] = 2
        elif ev == cv2.EVENT_MBUTTONDOWN: st["down"] = 3          # 지우개
        elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP): st["down"] = 0
        elif ev == cv2.EVENT_MOUSEWHEEL:
            ix, iy = view_to_img(x, y)
            st["zoom"] = float(np.clip(st["zoom"] * (1.25 if flags > 0 else 0.8), 1, 8))
            st["cx"], st["cy"] = ix, iy
        if st["down"] and ev in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            ix, iy = view_to_img(x, y)
            cv2.circle(lab, (ix, iy), st["r"], 0 if st["down"] == 3 else st["down"], -1)

    win = "label: L=real crack(red)  R=sticker(blue)  M=erase  wheel=zoom  +/-=brush  s=save  ESC=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        disp[lab == 1] = (disp[lab == 1] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        disp[lab == 2] = (disp[lab == 2] * 0.4 + np.array([255, 120, 0]) * 0.6).astype(np.uint8)
        z = st["zoom"]
        M = np.float32([[z, 0, W / 2 - z * st["cx"]], [0, z, H / 2 - z * st["cy"]]])
        disp = cv2.warpAffine(disp, M, (W, H), flags=cv2.INTER_NEAREST)
        cv2.putText(disp, f"brush {st['r']}px  zoom x{z:.1f}  real {int((lab == 1).sum())}px  sticker {int((lab == 2).sum())}px",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(15) & 0xFF
        if k in (ord('+'), ord('=')): st["r"] = min(st["r"] + 1, 30)
        elif k in (ord('-'), ord('_')): st["r"] = max(st["r"] - 1, 1)
        elif k == ord('s'):
            np.save(lab_path, lab)
            print(f"저장: {lab_path}  (진짜 균열 {(lab == 1).sum()} px, 스티커 {(lab == 2).sum()} px)")
            break
        elif k == 27:
            print("저장 안 하고 종료")
            break
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════════

def opt(args, key, default, cast=str):
    if key in args:
        i = args.index(key)
        return cast(args[i + 1])
    return default


if __name__ == "__main__":
    # cp949 콘솔에서 ✅ 등 출력 시 죽지 않게, 파이프로 받을 땐 UTF-8
    sys.stdout.reconfigure(encoding=None if sys.stdout.isatty() else 'utf-8', errors='replace')
    args = sys.argv[1:]
    mode = args[0] if args else ""
    if mode == "fmt":
        run_fmt()
    elif mode == "label" and len(args) > 1:
        run_label(args[1])
    elif mode == "resp":
        run_resp(args[1] if len(args) > 1 else None)
    elif mode == "lin":
        run_lin(fourcc=opt(args, "--fourcc", "YUY2"),
                led_no=opt(args, "--led", 1, int),
                brightness=opt(args, "--brightness", None, float),
                frames=opt(args, "--frames", LIN_FRAMES, int),
                gamma=opt(args, "--gamma", None, float),
                contrast=opt(args, "--contrast", None, float),
                top=opt(args, "--top", None, int))
    elif mode == "solve" and len(args) > 1:
        ds = 2 if "ds2" in args else 1
        run_solve(args[1], ds, show="--noshow" not in args)
    else:
        print(__doc__)
