"""
PS 균열 검사기 — 장치 연결부터 균열 판정까지 한 화면에서

실행:  python ps_app.py      (또는 run_inspector.bat 더블클릭)

단계
  0. 장치 연결          : 아두이노(LED) 포트와 카메라를 자동으로 찾아 연결
  1. 카메라 캘리브레이션 : ChArUco 보드 사진으로 렌즈·스테이지 평면 측정
  2. LED 위치 캘리브레이션: 크롬볼 하이라이트로 LED 8개의 3D 위치 측정
  3. 기준 촬영          : 흰 종이로 LED 밝기·빔 모양 기록
  4. 균열 검사          : 타일 촬영 → '균열이 있습니다' / '무늬입니다'

계산은 기존 모듈을 그대로 사용 (charuco_calib, led_calib, ps_run, camtest_and_solver).
"""
import os
import sys
import io
import json
import time
import glob
import shutil
import queue
import threading
import contextlib
import traceback

import matplotlib
matplotlib.use("Agg")          # 계산 스레드에서 그림 저장 (창 없음)
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

BASE = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)   # 실행 파일(exe)이면 exe 옆 폴더
        else os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import ps_run
import charuco_calib
import led_calib
import camtest_and_solver as judge

CONFIG_PATH = os.path.join(BASE, "ps_config.json")
LOG_PATH = os.path.join(BASE, "ps_app_log.txt")        # 화면 기록을 파일로도 남김 (문의 시 첨부)
W, H = ps_run.FRAME_WIDTH, ps_run.FRAME_HEIGHT
LED_COUNT = ps_run.LED_COUNT
PREVIEW_W, PREVIEW_H = 880, 550
FONT = "Malgun Gothic"

COL_OK, COL_STALE, COL_TODO = "#1a7f37", "#c46a00", "#808080"


# ════════════════════════════════════════════════════════════
# 하드웨어: 카메라 1대 + 아두이노 1대를 앱이 독점
# ════════════════════════════════════════════════════════════

class Hardware:
    """카메라는 전용 스레드가 계속 읽고, 촬영 함수는 '새 프레임'을 기다려 평균낸다."""

    def __init__(self):
        self.ser = None
        self.cap = None
        self.port = None
        self.cam_index = None
        self._cap_lock = threading.Lock()
        self._ser_lock = threading.Lock()
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._run = False
        self._thread = None

    @property
    def connected(self):
        return self.ser is not None and self.cap is not None

    # ── 시리얼 ──────────────────────────────────────────────
    @staticmethod
    def open_serial(port, wait_ready=3.0):
        import serial
        s = serial.Serial(port, ps_run.BAUD_RATE, timeout=0.2)
        t = time.time() + wait_ready
        ready = False
        while time.time() < t:                 # 포트를 열면 아두이노가 리셋되며 READY 를 보냄
            if "READY" in s.readline().decode(errors="ignore"):
                ready = True
                break
        s.reset_input_buffer()
        return s, ready

    @staticmethod
    def wait_ack(s, expected, timeout=2.0):
        t = time.time() + timeout
        while time.time() < t:
            if expected in s.readline().decode(errors="ignore"):
                return True
        return False

    def led(self, k):
        """0=끄기, 1~8=LED 하나 켜기, 9=전체 켜기"""
        if self.ser is None:
            return False
        with self._ser_lock:
            self.ser.reset_input_buffer()
            self.ser.write(str(k).encode())
            return self.wait_ack(self.ser, "OFF" if k == 0 else "ON")

    # ── 카메라 ──────────────────────────────────────────────
    @staticmethod
    def open_camera(index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        return cap

    def apply_settings(self, exposure, gain):
        """노출·게인 + 측정용 고정 설정 (드라이버가 마지막 값을 기억하므로 항상 명시)"""
        with self._cap_lock:
            c = self.cap
            c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            c.set(cv2.CAP_PROP_EXPOSURE, exposure)
            c.set(cv2.CAP_PROP_GAIN, gain)
            if ps_run.CAM_GAMMA is not None:
                c.set(cv2.CAP_PROP_GAMMA, ps_run.CAM_GAMMA)
            c.set(cv2.CAP_PROP_BRIGHTNESS, ps_run.CAM_BRIGHTNESS)
            c.set(cv2.CAP_PROP_CONTRAST, ps_run.CAM_CONTRAST)
        self.wait_frames(3)

    def get(self, prop):
        with self._cap_lock:
            return self.cap.get(prop)

    def _reader(self):
        while self._run:
            with self._cap_lock:
                ok, f = self.cap.read() if self.cap is not None else (False, None)
            if not ok:
                time.sleep(0.02)
                continue
            with self._cond:
                self._frame = f
                self._seq += 1
                self._cond.notify_all()
            time.sleep(0.002)                   # 설정 변경 스레드가 잠금을 잡을 틈

    def latest(self):
        with self._cond:
            return self._frame

    def wait_frames(self, n, timeout=15.0):
        with self._cond:
            target = self._seq + n
            t_end = time.time() + timeout
            while self._seq < target:
                rem = t_end - time.time()
                if rem <= 0:
                    raise TimeoutError("카메라 영상이 들어오지 않습니다 (USB 연결 확인)")
                self._cond.wait(rem)
            return self._frame

    def grab_avg(self, n, discard=3):
        """설정·LED 변경 직후 프레임 discard 장 버리고 n 장 평균 (흑백, float32)"""
        self.wait_frames(discard)
        acc = None
        for _ in range(n):
            g = cv2.cvtColor(self.wait_frames(1), cv2.COLOR_BGR2GRAY).astype(np.float64)
            acc = g if acc is None else acc + g
        return (acc / n).astype(np.float32)

    # ── 연결 / 해제 ─────────────────────────────────────────
    def connect(self, port, cam_index, ser=None):
        self.disconnect()
        if ser is None:
            ser, _ = self.open_serial(port)
        cap = self.open_camera(cam_index)
        if cap is None:
            ser.close()
            raise RuntimeError(f"카메라 #{cam_index} 를 열 수 없습니다 (다른 프로그램이 쓰는 중인지 확인)")
        ok, f = cap.read()
        if not ok:
            cap.release(); ser.close()
            raise RuntimeError(f"카메라 #{cam_index} 영상을 읽을 수 없습니다")
        if f.shape[1] != W or f.shape[0] != H:
            cap.release(); ser.close()
            raise RuntimeError(f"카메라 #{cam_index} 해상도가 {f.shape[1]}x{f.shape[0]} 입니다 "
                               f"({W}x{H} 필요). 다른 카메라일 수 있습니다.")
        self.ser, self.cap, self.port, self.cam_index = ser, cap, port, cam_index
        self._run = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self.wait_frames(2)

    def disconnect(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self.ser is not None:
            try:
                self.ser.write(b"0")
                self.ser.close()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
        self.ser = self.cap = None


def detect_arduino(log):
    """USB 시리얼 포트를 하나씩 열어 READY / OFF 응답으로 LED 컨트롤러를 찾음"""
    from serial.tools import list_ports
    ports = [p for p in list_ports.comports() if p.vid is not None]    # 블루투스 등 가상 포트 제외
    ports.sort(key=lambda p: 0 if "arduino" in (p.description or "").lower() else 1)
    if not ports:
        log("  USB 시리얼 장치가 없습니다. 아두이노 USB 케이블을 확인하세요.")
        return None, None
    for p in ports:
        log(f"  {p.device} ({p.description}) 확인 중...")
        try:
            s, ready = Hardware.open_serial(p.device)
            ok = ready
            if not ok:
                s.write(b"0")
                ok = Hardware.wait_ack(s, "OFF", 1.5)
            if ok:
                log(f"  → LED 컨트롤러 발견: {p.device}")
                return p.device, s
            s.close()
            log("    응답 없음 (LED 컨트롤러가 아님)")
        except Exception as e:
            log(f"    열 수 없음: {e}")
    return None, None


def detect_camera(ser, log, max_index=6):
    """카메라 번호를 하나씩 열어, LED를 켰다 껐을 때 밝기가 바뀌는 카메라를 고름"""
    cands = []
    for i in range(max_index):
        cap = Hardware.open_camera(i)
        if cap is None:
            continue
        ok, f = cap.read()
        if not ok:
            cap.release()
            continue
        h, w = f.shape[:2]
        diff = None
        if ser is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, -4)
            means = []
            for cmd, ack in ((b"0", "OFF"), (b"9", "ON")):
                ser.reset_input_buffer(); ser.write(cmd); Hardware.wait_ack(ser, ack)
                for _ in range(5):
                    cap.read()
                vals = []
                for _ in range(3):
                    ok, f = cap.read()
                    if ok:
                        vals.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())
                means.append(np.mean(vals) if vals else 0.0)
            ser.reset_input_buffer(); ser.write(b"0"); Hardware.wait_ack(ser, "OFF")
            diff = means[1] - means[0]
        cap.release()
        log(f"  카메라 #{i}: {w}x{h}" + (f", LED 켰을 때 밝기 변화 {diff:+.1f}" if diff is not None else ""))
        cands.append((i, w, h, diff))
    if not cands:
        log("  카메라를 찾지 못했습니다. 카메라 USB 케이블을 확인하세요.")
        return None
    right_res = [c for c in cands if (c[1], c[2]) == (W, H)]
    if ser is not None:
        lit = [c for c in cands if c[3] is not None and c[3] > 5]
        if lit:
            best = max(lit, key=lambda c: c[3])
            log(f"  → 검사용 카메라: #{best[0]} (LED에 반응함)")
            return best[0]
    if len(right_res) == 1:
        log(f"  → 검사용 카메라: #{right_res[0][0]} ({W}x{H} 지원)")
        return right_res[0][0]
    log("  LED에 반응하는 카메라를 확정하지 못했습니다. 아래에서 번호를 직접 고르세요.")
    return None


# ════════════════════════════════════════════════════════════
# 진행 상태 (파일로 판단)
# ════════════════════════════════════════════════════════════

def _mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else None


def step_status():
    """각 단계: 'ok' | 'stale'(앞 단계를 다시 해서 다시 필요) | 'todo'"""
    st = {}
    intr = _mtime(ps_run.INTR_PATH)
    has_plane = False
    if intr:
        try:
            has_plane = "plane_n" in np.load(ps_run.INTR_PATH).files
        except Exception:
            pass
    st["camcal"] = "ok" if intr and has_plane else "todo"
    led = _mtime(ps_run.LED_PATH)
    st["ledcal"] = "todo" if not led else ("ok" if intr and led >= intr else "stale")
    ref_meta = os.path.join(ps_run.REF_DIR, "meta.json")
    ref = _mtime(ref_meta)
    if not ref:
        st["ref"] = "todo"
    else:
        try:
            m = json.load(open(ref_meta))
        except Exception:
            m = {}
        same_cam = m.get("brightness") == float(ps_run.CAM_BRIGHTNESS) and m.get("contrast") == float(ps_run.CAM_CONTRAST)
        st["ref"] = "ok" if same_cam and led and ref >= led else "stale"
    return st


# ════════════════════════════════════════════════════════════
# 화면
# ════════════════════════════════════════════════════════════

class QueueWriter(io.TextIOBase):
    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(s)
        return len(s)


class Page:
    key = ""
    title = ""
    wants_preview = True

    def __init__(self, app):
        self.app = app
        self.hw = app.hw

    def instructions(self):
        return ""

    def next_hint(self):
        return ""

    def build(self, frame):
        pass

    def on_enter(self):
        pass

    def on_leave(self):
        pass

    def overlay(self, frame):
        return frame

    def on_click(self, x, y, button):
        pass

    def on_motion(self, x, y):
        pass

    # 공통 도우미
    def btn(self, parent, text, cmd, need_hw=True, **kw):
        b = ttk.Button(parent, text=text, command=lambda: self.app.guard(cmd, need_hw))
        b.pack(side=kw.get("side", "left"), padx=3, pady=3)
        return b


# ── 0. 장치 연결 ────────────────────────────────────────────
class ConnectPage(Page):
    key, title = "connect", "0. 장치 연결"
    wants_preview = True

    def instructions(self):
        return ("검사 장치의 카메라와 LED 컨트롤러(아두이노)를 PC에 USB로 연결하세요.\n\n"
                "[자동으로 찾아 연결]을 누르면 USB 포트를 하나씩 확인해 LED 컨트롤러를 찾고, "
                "LED를 켰다 껐다 하면서 밝기가 바뀌는 카메라를 검사용 카메라로 고릅니다. "
                "이때 LED가 몇 번 깜빡이는 것은 정상입니다.\n\n"
                "한 번 연결에 성공하면 설정이 저장되어 다음 실행부터는 자동으로 연결됩니다.")

    def next_hint(self):
        if self.hw.connected:
            return "연결 완료. 왼쪽의 다음 단계로 진행하세요 (이미 완료된 단계는 건너뛰어도 됩니다)."
        return "[자동으로 찾아 연결]을 누르세요."

    def build(self, f):
        row = ttk.Frame(f); row.pack(fill="x")
        self.btn(row, "자동으로 찾아 연결", self.auto)
        self.btn(row, "연결 해제", self.disconnect)
        self.btn(row, "LED 테스트 (1→8 차례로)", self.led_test)
        row2 = ttk.Frame(f); row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="직접 지정 — LED 포트:").pack(side="left")
        from serial.tools import list_ports
        ports = [p.device for p in list_ports.comports()]
        self.port_var = tk.StringVar(value=self.hw.port or (ports[0] if ports else ""))
        ttk.Combobox(row2, textvariable=self.port_var, values=ports, width=8).pack(side="left", padx=3)
        ttk.Label(row2, text="카메라 번호:").pack(side="left")
        self.cam_var = tk.IntVar(value=self.hw.cam_index if self.hw.cam_index is not None else 1)
        ttk.Spinbox(row2, from_=0, to=9, textvariable=self.cam_var, width=4).pack(side="left", padx=3)
        self.btn(row2, "이 설정으로 연결", self.manual)
        self.status = ttk.Label(f, text="", font=(FONT, 11, "bold"))
        self.status.pack(anchor="w", pady=6)
        self.refresh()

    def refresh(self):
        if self.hw.connected:
            self.status.config(text=f"● 연결됨 — 카메라 #{self.hw.cam_index} ({W}x{H}), LED 컨트롤러 {self.hw.port}",
                               foreground=COL_OK)
        else:
            self.status.config(text="● 연결 안 됨", foreground=COL_TODO)

    def _finish_connect(self, port, cam, ser=None):
        self.hw.connect(port, cam, ser)
        json.dump({"serial_port": port, "camera_index": cam}, open(CONFIG_PATH, "w"), indent=1)
        # 카메라 점검: 블랙 레벨 (0으로 잘리면 어두운 무늬가 가짜 균열로 보임)
        self.hw.led(0)
        self.hw.apply_settings(-6, ps_run.CAM_GAIN)
        dk = self.hw.grab_avg(4)
        zero = float((dk <= 0.5).mean() * 100)
        print(f"카메라 점검: 암전 평균 {dk.mean():.1f}, 0으로 잘린 픽셀 {zero:.0f}%")
        if zero > 5:
            print("  !! 블랙 레벨이 잘립니다. 카메라 설정(brightness)이 적용되지 않았을 수 있습니다.")
        print(f"✅ 연결 완료: LED {port}, 카메라 #{cam}")

    def auto(self):
        def work():
            self.hw.disconnect()                 # 기존 연결이 포트·카메라를 잡고 있으면 탐지가 안 됨
            print("[LED 컨트롤러 찾는 중]")
            port, ser = detect_arduino(print)
            if port is None:
                raise RuntimeError("LED 컨트롤러(아두이노)를 찾지 못했습니다. USB 케이블과 전원을 확인하세요.")
            print("[카메라 찾는 중]")
            cam = detect_camera(ser, print)
            if cam is None:
                ser.close()
                raise RuntimeError("검사용 카메라를 확정하지 못했습니다. 아래 '직접 지정'에서 카메라 번호를 골라 연결하세요.")
            self._finish_connect(port, cam, ser)
            self.app.post(lambda: (self.port_var.set(port), self.cam_var.set(cam)))
        self.app.run_task("장치 찾는 중...", work, self.refresh)

    def manual(self):
        port, cam = self.port_var.get(), int(self.cam_var.get())
        self.app.run_task("연결 중...", lambda: self._finish_connect(port, cam), self.refresh)

    def disconnect(self):
        self.hw.disconnect()
        self.refresh()
        self.app.update_nav()

    def led_test(self):
        if not self.hw.connected:
            return self.app.warn("먼저 장치를 연결하세요.")

        def work():
            self.hw.apply_settings(-6, ps_run.CAM_GAIN)
            for i in range(1, LED_COUNT + 1):
                print(f"  LED{i - 1} 켜기: {'OK' if self.hw.led(i) else '응답 없음'}")
                time.sleep(0.4)
            self.hw.led(0)
        self.app.run_task("LED 테스트 중...", work)

    def on_enter(self):
        if self.hw.connected:
            self.app.run_task("", lambda: (self.hw.apply_settings(-6, ps_run.CAM_GAIN), self.hw.led(9)))


# ── 1. 카메라 캘리브레이션 ──────────────────────────────────
class CamCalPage(Page):
    key, title = "camcal", "1. 카메라 캘리브레이션"

    def __init__(self, app):
        super().__init__(app)
        self.exp = charuco_calib.CAM_EXPOSURE
        self.coverage = []
        self.n_corners = 0
        self.last_ip = None

    def counts(self):
        n = len(glob.glob(os.path.join(charuco_calib.img_dir, "img_*.png")))
        return n, os.path.exists(charuco_calib.plane_path)

    def instructions(self):
        c = charuco_calib
        return ("렌즈의 왜곡과 스테이지 바닥의 위치를 측정합니다. 장치를 처음 조립했거나 카메라·렌즈·초점을 바꿨을 때만 합니다.\n\n"
                f"준비물: ChArUco 보드 ({c.SQUARES_X}x{c.SQUARES_Y}칸, 칸 {c.SQUARE_MM}mm). [보드 PDF 열기]로 열어 "
                "'실제 크기(100%)'로 인쇄하고, 휘지 않는 판에 평평하게 붙이세요. 인쇄된 칸 크기를 자로 재서 "
                f"{c.SQUARE_MM}mm 가 맞는지 확인하세요.\n\n"
                "① 보드를 손으로 들고 여러 각도(앞뒤·좌우로 20~40° 기울임)와 위치에서 [사진 저장]을 누르세요. "
                "미리보기의 빨간 점이 검출된 코너, 초록 점이 지금까지 저장된 코너입니다. 초록 점이 화면 전체를 고르게 덮도록 "
                "15~25장 저장합니다.\n"
                "② 보드를 스테이지 바닥(타일 놓는 곳)에 평평하게 놓고 [바닥 평면 사진 저장]을 누르세요.\n"
                "③ [캘리브레이션 실행]을 누르세요. 재투영 오차가 0.5px 이하면 좋고, 1.0px 이상이면 사진을 다시 찍는 게 좋습니다.\n\n"
                "※ 이 단계를 다시 하면 2·3단계도 다시 해야 합니다.")

    def next_hint(self):
        n, plane = self.counts()
        st = step_status()["camcal"]
        if n < 15:
            return f"보드를 여러 각도로 기울여 [사진 저장] — 지금 {n}장 (15~25장 권장)"
        if not plane:
            return "보드를 스테이지 바닥에 평평하게 놓고 [바닥 평면 사진 저장]"
        if st != "ok":
            return "[캘리브레이션 실행]"
        return "완료. 2단계(LED 위치 캘리브레이션)로 진행하세요."

    def build(self, f):
        row = ttk.Frame(f); row.pack(fill="x")
        self.btn(row, "보드 PDF 열기", self.open_pdf, need_hw=False)
        self.btn(row, "사진 저장", self.save)
        self.btn(row, "바닥 평면 사진 저장", self.save_plane)
        self.btn(row, "캘리브레이션 실행", self.calibrate)
        row2 = ttk.Frame(f); row2.pack(fill="x")
        ttk.Label(row2, text="노출:").pack(side="left")
        self.btn(row2, "어둡게", lambda: self.set_exp(-1))
        self.btn(row2, "밝게", lambda: self.set_exp(+1))
        self.btn(row2, "저장 사진 모두 삭제", self.clear)
        self.info = ttk.Label(f, text="", font=(FONT, 10))
        self.info.pack(anchor="w", pady=4)
        self.refresh()

    def refresh(self):
        n, plane = self.counts()
        self.info.config(text=f"저장된 사진 {n}장 (권장 15~25)  |  바닥 평면 사진: {'있음' if plane else '없음'}  |  노출 {self.exp:+d}")
        self.app.update_nav()

    def on_enter(self):
        def work():
            self.hw.apply_settings(self.exp, charuco_calib.CAM_GAIN)
            self.hw.led(9)
            cov = []
            for p in sorted(glob.glob(os.path.join(charuco_calib.img_dir, "img_*.png"))):
                ip, _ = charuco_calib.detect_points(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
                if ip is not None:
                    cov.append(ip)
            self.coverage = cov
        self.app.run_task("카메라 준비 중...", work, self.refresh)

    def on_leave(self):
        if self.hw.connected:
            self.hw.led(0)

    def set_exp(self, d):
        self.exp += d
        self.app.run_task("", lambda: self.hw.apply_settings(self.exp, charuco_calib.CAM_GAIN), self.refresh)

    def overlay(self, frame):
        self._tick = getattr(self, "_tick", 0) + 1
        if self._tick % 3 == 1:                  # 보드 검출은 무거워서 3프레임마다
            ip, _ = charuco_calib.detect_points(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            self.n_corners = 0 if ip is None else len(ip)
            self.last_ip = ip
        ip = self.last_ip
        for pts in self.coverage:
            for x, y in pts.astype(int):
                cv2.circle(frame, (x, y), 3, (0, 170, 0), -1)
        if ip is not None:
            for x, y in ip.astype(int):
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        ok = self.n_corners >= charuco_calib.MIN_CORNERS
        cv2.putText(frame, f"corners {self.n_corners}  {'OK' if ok else 'too few'}", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0) if ok else (0, 200, 255), 3)
        return frame

    def _grab_board(self):
        fr = self.hw.latest()
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ip, _ = charuco_calib.detect_points(gray)
        if ip is None or len(ip) < charuco_calib.MIN_CORNERS:
            raise RuntimeError(f"보드 코너가 부족합니다 ({0 if ip is None else len(ip)}개, "
                               f"{charuco_calib.MIN_CORNERS}개 이상 필요). 보드가 화면에 잘 보이게 하세요.")
        return gray, ip

    def save(self):
        gray, ip = self._grab_board()
        os.makedirs(charuco_calib.img_dir, exist_ok=True)
        ex = sorted(glob.glob(os.path.join(charuco_calib.img_dir, "img_*.png")))
        idx = 0 if not ex else int(os.path.basename(ex[-1])[4:7]) + 1
        path = os.path.join(charuco_calib.img_dir, f"img_{idx:03d}.png")
        cv2.imwrite(path, gray)
        self.coverage.append(ip)
        self.app.log(f"사진 저장: {os.path.basename(path)} (코너 {len(ip)}개)\n")
        self.refresh()

    def save_plane(self):
        gray, ip = self._grab_board()
        cv2.imwrite(charuco_calib.plane_path, gray)
        self.app.log(f"바닥 평면 사진 저장 (코너 {len(ip)}개)\n")
        self.refresh()

    def clear(self):
        n, _ = self.counts()
        if not messagebox.askyesno("확인", f"저장된 보드 사진 {n}장과 바닥 평면 사진을 모두 삭제할까요?"):
            return
        for p in glob.glob(os.path.join(charuco_calib.img_dir, "img_*.png")):
            os.remove(p)
        if os.path.exists(charuco_calib.plane_path):
            os.remove(charuco_calib.plane_path)
        self.coverage = []
        self.refresh()

    def calibrate(self):
        n, plane = self.counts()
        if n < 8:
            return self.app.warn(f"사진이 {n}장뿐입니다. 최소 8장, 권장 15~25장을 저장하세요.")
        if not plane:
            return self.app.warn("바닥 평면 사진이 없습니다. 보드를 스테이지 바닥에 평평하게 놓고 저장하세요.")

        def work():
            r = charuco_calib.calibrate()
            if r is None:
                raise RuntimeError("캘리브레이션 실패 (위 기록 참고)")
            if "plane_n" not in r:
                raise RuntimeError("바닥 평면 사진에서 보드를 찾지 못했습니다. 다시 찍어주세요.")
            q = "좋음" if r["rms"] <= 0.5 else ("보통" if r["rms"] < 1.0 else "나쁨 — 사진을 다시 찍는 것을 권장")
            print(f"\n결과: 재투영 오차 {r['rms']:.3f}px ({q}), 카메라→바닥 {r['plane_dist_mm']:.1f}mm")
            print("→ 2단계(LED 위치 캘리브레이션)와 3단계(기준 촬영)를 다시 해야 합니다.")
        self.app.run_task("캘리브레이션 계산 중...", work, self.refresh)

    def open_pdf(self):
        pdf = os.path.join(BASE, "charuco_11x7_6mm_A4.pdf")
        if os.path.exists(pdf):
            os.startfile(pdf)
        else:
            self.app.warn("보드 PDF 파일이 없습니다: charuco_11x7_6mm_A4.pdf")


# ── 2. LED 위치 캘리브레이션 ────────────────────────────────
class LedCalPage(Page):
    key, title = "ledcal", "2. LED 위치 캘리브레이션"

    def __init__(self, app):
        super().__init__(app)
        self.pts = []
        self.mouse = (W // 2, H // 2)
        self.exp = led_calib.EXPOSURE_VIEW
        self.review = None          # 촬영 직후 확인 이미지

    def positions(self):
        return sorted(glob.glob(os.path.join(led_calib.OUT_DIR, "pos_*")))

    def instructions(self):
        return ("LED 8개의 3D 위치를 측정합니다. 1단계를 했거나 LED를 옮겼을 때 합니다.\n\n"
                f"준비물: 지름 {led_calib.BALL_DIAMETER_MM:.0f}mm 크롬(거울) 볼, 볼이 구르지 않게 받칠 링.\n\n"
                "① 볼을 스테이지에 놓으세요. 링을 쓰면 '받침 높이'에 링 높이(mm)를, 바닥에 바로 놓으면 0을 입력합니다.\n"
                "② 미리보기에서 볼의 '테두리'를 빙 둘러가며 6~10곳 클릭하세요 (왼쪽 클릭: 추가, 오른쪽 클릭: 취소). "
                "오른쪽 아래 돋보기로 테두리를 정확히 맞추세요. 초록 원이 볼 윤곽과 겹치면 됩니다.\n"
                "③ [이 위치 촬영]을 누르면 LED 8개를 하나씩 켜서 촬영합니다. 확인 화면에서 볼 위의 빨간 원(하이라이트)이 "
                "8개 모두 잡혔는지 확인하세요.\n"
                "④ 볼을 다른 위치로 옮겨 ②~③을 반복합니다. 화면 네 귀퉁이 쪽과 가운데, 총 4~5곳을 권장합니다.\n"
                "⑤ [계산]을 누르세요. 하이라이트 오차 1~2px 이하면 양호합니다.\n\n"
                "※ 이 단계를 다시 하면 3단계(기준 촬영)도 다시 해야 합니다.")

    def next_hint(self):
        if step_status()["camcal"] != "ok":
            return "먼저 1단계(카메라 캘리브레이션)를 완료하세요."
        if self.review is not None:
            return "하이라이트 확인 후 [계속]을 누르세요. 8개가 아니면 [마지막 위치 삭제] 후 다시 찍으세요."
        n = len(self.positions())
        if n < 4:
            return f"볼 테두리를 6~10곳 클릭 → [이 위치 촬영]  (지금 {n}곳 / 4~5곳 권장, 클릭 {len(self.pts)}개)"
        if step_status()["ledcal"] != "ok":
            return f"{n}곳 촬영됨. 더 찍거나 [계산]을 누르세요."
        return "완료. 3단계(기준 촬영)로 진행하세요."

    def build(self, f):
        row = ttk.Frame(f); row.pack(fill="x")
        self.btn(row, "이 위치 촬영", self.capture)
        self.btn(row, "계속", self.resume)
        self.btn(row, "클릭 모두 지우기", self.clear_pts)
        self.btn(row, "계산", self.solve)
        row2 = ttk.Frame(f); row2.pack(fill="x")
        ttk.Label(row2, text="받침 높이(mm):").pack(side="left")
        self.sup_var = tk.DoubleVar(value=led_calib.BALL_SUPPORT_MM)
        ttk.Spinbox(row2, from_=0, to=100, increment=0.1, textvariable=self.sup_var, width=6).pack(side="left", padx=3)
        ttk.Label(row2, text="  미리보기 노출:").pack(side="left")
        self.btn(row2, "어둡게", lambda: self.set_exp(-1))
        self.btn(row2, "밝게", lambda: self.set_exp(+1))
        self.btn(row2, "마지막 위치 삭제", self.del_last)
        self.btn(row2, "모든 위치 삭제", self.del_all)
        self.info = ttk.Label(f, text="", font=(FONT, 10))
        self.info.pack(anchor="w", pady=4)
        self.refresh()

    def refresh(self):
        self.info.config(text=f"촬영된 볼 위치 {len(self.positions())}곳 (권장 4~5)  |  테두리 클릭 {len(self.pts)}개 (5개 이상)")
        self.app.update_nav()

    def on_enter(self):
        if step_status()["camcal"] != "ok":
            return
        self.app.run_task("카메라 준비 중...", lambda: (self.hw.apply_settings(self.exp, led_calib.CAM_GAIN),
                                                    self.hw.led(9)))

    def on_leave(self):
        if self.hw.connected:
            self.hw.led(0)

    def set_exp(self, d):
        self.exp += d
        self.app.run_task("", lambda: self.hw.apply_settings(self.exp, led_calib.CAM_GAIN))

    @property
    def wants_preview(self):
        return self.review is None

    def static_image(self):
        return self.review

    def on_click(self, x, y, button):
        if self.review is not None:
            return
        if button == 1:
            self.pts.append((x, y))
        elif self.pts:
            self.pts.pop()
        self.refresh()
        self.app.update_hint()

    def on_motion(self, x, y):
        self.mouse = (x, y)

    def overlay(self, frame):
        raw = frame.copy()
        for (x, y) in self.pts:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
        if len(self.pts) >= 3:
            cx, cy, r = led_calib.fit_circle(self.pts)
            cv2.circle(frame, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        mx, my = self.mouse
        hw_ = 30
        x0 = int(np.clip(mx - hw_, 0, W - 2 * hw_)); y0 = int(np.clip(my - hw_, 0, H - 2 * hw_))
        zoom = cv2.resize(raw[y0:y0 + 2 * hw_, x0:x0 + 2 * hw_], (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.line(zoom, (140, 0), (140, 280), (0, 255, 255), 1)
        cv2.line(zoom, (0, 140), (280, 140), (0, 255, 255), 1)
        zx = W - 290 if mx < W / 2 else 10
        frame[H - 290:H - 10, zx:zx + 280] = zoom
        cv2.rectangle(frame, (zx, H - 290), (zx + 280, H - 10), (0, 255, 255), 2)
        return frame

    def clear_pts(self):
        self.pts = []
        self.refresh()

    def capture(self):
        if step_status()["camcal"] != "ok":
            return self.app.warn("먼저 1단계(카메라 캘리브레이션)를 완료하세요.")
        if len(self.pts) < 5:
            return self.app.warn(f"볼 테두리를 5곳 이상 클릭하세요 (지금 {len(self.pts)}곳).")
        pts = list(self.pts)
        sup = float(self.sup_var.get())
        ex = self.positions()
        idx = 0 if not ex else int(os.path.basename(ex[-1])[4:]) + 1
        pos_dir = os.path.join(led_calib.OUT_DIR, f"pos_{idx:02d}")

        def work():
            os.makedirs(pos_dir, exist_ok=True)
            view = np.clip(np.round(self.hw.grab_avg(2)), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(pos_dir, "view.png"), view)
            self.hw.led(0)
            self.hw.apply_settings(led_calib.EXPOSURE_HL, led_calib.CAM_GAIN)
            for i in range(1, LED_COUNT + 1):
                if not self.hw.led(i):
                    print(f"  LED{i - 1}: 응답 없음")
                    continue
                img = np.clip(np.round(self.hw.grab_avg(led_calib.FRAMES_AVG)), 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(pos_dir, f"led{i - 1}.png"), img)
                self.hw.led(0)
            self.hw.apply_settings(self.exp, led_calib.CAM_GAIN)
            self.hw.led(9)
            json.dump({"edge_points": [[float(x), float(y)] for x, y in pts],
                       "ball_diameter_mm": led_calib.BALL_DIAMETER_MM,
                       "support_mm": sup, "exposure_hl": led_calib.EXPOSURE_HL},
                      open(os.path.join(pos_dir, "meta.json"), "w"), indent=1)
            circle = led_calib.fit_circle(pts)
            comp = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
            cv2.circle(comp, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 0, 0), 2)
            found = 0
            for i in range(LED_COUNT):
                p = os.path.join(pos_dir, f"led{i}.png")
                if not os.path.exists(p):
                    continue
                hl = led_calib.find_highlight(cv2.imread(p, cv2.IMREAD_GRAYSCALE), circle)
                if hl:
                    found += 1
                    cv2.circle(comp, (int(hl[0]), int(hl[1])), 8, (0, 0, 255), 2)
                    cv2.putText(comp, str(i), (int(hl[0]) + 10, int(hl[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(comp, f"highlights {found}/{LED_COUNT}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 220, 0) if found == LED_COUNT else (0, 0, 255), 3)
            cv2.imwrite(os.path.join(pos_dir, "check.png"), comp)
            print(f"{os.path.basename(pos_dir)} 촬영 완료: 하이라이트 {found}/{LED_COUNT}")
            if found < LED_COUNT:
                print("  !! 하이라이트가 부족합니다. 테두리 클릭이 정확한지 확인하고, 필요하면 [마지막 위치 삭제] 후 다시 찍으세요.")
            self.review = comp

        def done():
            self.pts = []
            self.refresh()
        self.app.run_task("볼 촬영 중 (LED 8개)...", work, done)

    def resume(self):
        self.review = None
        self.app.update_hint()

    def del_last(self):
        ex = self.positions()
        if not ex:
            return
        if messagebox.askyesno("확인", f"{os.path.basename(ex[-1])} 를 삭제할까요?"):
            shutil.rmtree(ex[-1])
            self.review = None
            self.refresh()

    def del_all(self):
        ex = self.positions()
        if ex and messagebox.askyesno("확인", f"촬영된 볼 위치 {len(ex)}곳을 모두 삭제할까요?"):
            for d in ex:
                shutil.rmtree(d)
            self.review = None
            self.refresh()

    def solve(self):
        if len(self.positions()) < 2:
            return self.app.warn("볼 위치가 2곳 이상 필요합니다 (권장 4~5곳).")

        def work():
            out = led_calib.solve()
            if out is None:
                raise RuntimeError("계산 실패 (위 기록 참고)")
            P, res = out
            if np.isnan(P).any():
                raise RuntimeError("일부 LED 위치를 계산하지 못했습니다. 하이라이트가 안 잡힌 위치를 다시 찍으세요.")
            print("\n결과: LED 8개 위치 계산 완료. → 3단계(기준 촬영)를 다시 해야 합니다.")
        self.app.run_task("LED 위치 계산 중 (수십 초)...", work, self.refresh)


# ── 3. 기준 촬영 ────────────────────────────────────────────
class RefPage(Page):
    key, title = "ref", "3. 기준 촬영 (흰 종이)"

    def __init__(self, app):
        super().__init__(app)
        meta = os.path.join(ps_run.REF_DIR, "meta.json")
        self.exp = int(json.load(open(meta))["exposure"]) if os.path.exists(meta) else ps_run.CAM_EXPOSURE
        self.cur = 1
        self.maxv = 0

    def instructions(self):
        return ("LED마다 밝기와 빛이 퍼지는 모양을 흰 종이로 기록합니다. 2단계를 했거나, 노출을 바꾸거나, "
                "장치를 옮겼을 때 다시 합니다.\n\n"
                "① 방 조명을 끄고 창문 빛을 가리세요 (검사 때와 같은 조건).\n"
                "② 흰 무광 종이(일반 복사용지)를 화면 전체에 평평하게 까세요. 주름이 없어야 합니다.\n"
                "③ [LED 1]~[LED 8]을 눌러 하나씩 켜 보세요. 빨갛게 칠해지는 곳은 너무 밝아 포화된 곳입니다. "
                "가장 밝은 LED에서도 빨간 곳이 없도록 [어둡게]/[밝게]로 노출을 맞추세요.\n"
                "④ [기준 촬영]을 누르세요 (약 30초).\n\n"
                "※ 검사할 때 노출은 여기서 정한 값으로 자동 고정됩니다.")

    def next_hint(self):
        st = step_status()
        if st["ledcal"] != "ok":
            return "먼저 2단계(LED 위치 캘리브레이션)를 완료하세요."
        if st["ref"] != "ok":
            return "불 끄고 흰 종이를 깔고, 포화(빨강) 없게 노출을 맞춘 뒤 [기준 촬영]"
        return "완료. 4단계(균열 검사)로 진행하세요."

    def build(self, f):
        row = ttk.Frame(f); row.pack(fill="x")
        for i in range(1, LED_COUNT + 1):
            self.btn(row, f"LED {i}", lambda i=i: self.sel(i))
        row2 = ttk.Frame(f); row2.pack(fill="x")
        ttk.Label(row2, text="노출:").pack(side="left")
        self.btn(row2, "어둡게", lambda: self.set_exp(-1))
        self.btn(row2, "밝게", lambda: self.set_exp(+1))
        self.btn(row2, "기준 촬영", self.capture)
        self.info = ttk.Label(f, text="", font=(FONT, 10))
        self.info.pack(anchor="w", pady=4)
        self.refresh()

    def refresh(self):
        self.info.config(text=f"켜진 LED {self.cur}  |  노출 {self.exp:+d}  |  밝은 곳 최대값 {self.maxv:.0f} "
                              f"({ps_run.SAT_LEVEL} 이상이면 포화)")
        self.app.update_nav()

    def on_enter(self):
        self.app.run_task("카메라 준비 중...", lambda: (self.hw.apply_settings(self.exp, ps_run.CAM_GAIN),
                                                    self.hw.led(self.cur)))

    def on_leave(self):
        if self.hw.connected:
            self.hw.led(0)

    def sel(self, i):
        self.cur = i
        self.app.run_task("", lambda: self.hw.led(i), self.refresh)

    def set_exp(self, d):
        self.exp += d
        self.app.run_task("", lambda: self.hw.apply_settings(self.exp, ps_run.CAM_GAIN), self.refresh)

    def overlay(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.maxv = float(cv2.GaussianBlur(g, (9, 9), 0).max())
        frame[g >= ps_run.SAT_LEVEL] = (0, 0, 255)
        sat = self.maxv >= ps_run.SAT_LEVEL
        cv2.putText(frame, f"LED{self.cur}  max {self.maxv:.0f}  {'SATURATED - darker!' if sat else 'OK'}", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if sat else (0, 220, 0), 3)
        return frame

    def capture(self):
        if step_status()["ledcal"] != "ok":
            return self.app.warn("먼저 2단계(LED 위치 캘리브레이션)를 완료하세요.")
        if os.path.exists(os.path.join(ps_run.REF_DIR, "meta.json")):
            if not messagebox.askyesno("확인", "기존 기준 촬영을 새로 찍은 것으로 바꿀까요?\n(기존 것은 백업 폴더에 보관됩니다)"):
                return

        def work():
            tmp = ps_run.REF_DIR + "_new"
            shutil.rmtree(tmp, ignore_errors=True)
            capture_stack(self.hw, tmp, self.exp, "기준")
            if os.path.exists(ps_run.REF_DIR):
                bak = ps_run.REF_DIR + time.strftime("_backup_%Y%m%d_%H%M%S")
                shutil.move(ps_run.REF_DIR, bak)
                print(f"기존 기준 촬영 백업: {os.path.basename(bak)}")
            shutil.move(tmp, ps_run.REF_DIR)
            print("✅ 기준 촬영 완료")
            self.hw.led(self.cur)
        self.app.run_task("기준 촬영 중 (약 30초, 장치를 건드리지 마세요)...", work, self.refresh)


def capture_stack(hw, out_dir, exposure, label):
    """암전 + LED 8개 촬영 → out_dir (ps_run 과 같은 형식). 문제 있으면 경고 출력"""
    os.makedirs(out_dir, exist_ok=True)
    hw.apply_settings(exposure, ps_run.CAM_GAIN)
    hw.led(0)
    dk = hw.grab_avg(ps_run.FRAMES_AVG)
    np.save(os.path.join(out_dir, "dark.npy"), dk)
    warn = []
    if dk.std() > 3:
        warn.append("주변 빛이 들어왔습니다 (방 조명·창문). 결과가 부정확할 수 있으니 불을 끄고 다시 찍으세요.")
    print(f"  암전: 평균 {dk.mean():.1f}, 불균일 {dk.std():.1f}")
    for i in range(1, LED_COUNT + 1):
        if not hw.led(i):
            raise RuntimeError(f"LED{i - 1} 이 응답하지 않습니다. 연결을 확인하세요.")
        im = hw.grab_avg(ps_run.FRAMES_AVG)
        np.save(os.path.join(out_dir, f"led{i - 1}.npy"), im)
        cv2.imwrite(os.path.join(out_dir, f"led{i - 1}.png"), np.clip(im, 0, 255).astype(np.uint8))
        sat = float(np.mean(im >= ps_run.SAT_LEVEL)) * 100
        print(f"  {label} LED{i - 1} 완료 (최대 {np.percentile(im, 99.9):.0f}{f', 포화 {sat:.1f}%' if sat > 0.1 else ''})")
        if sat > 0.1:
            warn.append(f"LED{i - 1} 포화 {sat:.1f}% — 노출을 낮춰야 합니다.")
    hw.led(0)
    json.dump({"exposure": float(exposure), "gain": float(ps_run.CAM_GAIN),
               "gamma": hw.get(cv2.CAP_PROP_GAMMA),
               "brightness": hw.get(cv2.CAP_PROP_BRIGHTNESS),
               "contrast": hw.get(cv2.CAP_PROP_CONTRAST),
               "paper_thickness_mm": ps_run.PAPER_THICKNESS_MM,
               "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    for w in warn:
        print("  !! " + w)
    return warn


# ── 4. 균열 검사 ────────────────────────────────────────────
class InspectPage(Page):
    key, title = "inspect", "4. 균열 검사"

    def __init__(self, app):
        super().__init__(app)
        self.result_img = None
        self.verdict = None

    def instructions(self):
        return ("타일에 실제 균열이 있는지, 균열처럼 보이는 무늬·스티커인지 판정합니다.\n\n"
                "① 방 조명을 끄세요.\n"
                "② 타일을 스테이지에 놓으세요. 미리보기에서 타일이 화면 가운데에 오도록 맞춥니다.\n"
                "③ 이름을 확인하고 [촬영 + 판정]을 누르세요 (약 40초). 촬영 중에는 장치와 타일을 건드리지 마세요.\n"
                "④ 결과 화면에 '균열이 있습니다' 또는 '균열 없음 — 무늬입니다'가 표시됩니다. "
                "균열이면 해당 부분이 빨갛게, 노란 네모로 표시됩니다.\n\n"
                "※ 판정 원리: 실제 균열은 깨진 면 때문에 표면 방향이 꺾이고, 무늬·스티커는 평평한 면 위의 색 변화입니다. "
                "아주 가늘고 전혀 어긋나지 않은 균열은 놓칠 수 있습니다.")

    def next_hint(self):
        st = step_status()
        missing = [n for k, n in (("camcal", "1"), ("ledcal", "2"), ("ref", "3")) if st[k] != "ok"]
        if missing:
            return f"먼저 {', '.join(missing)}단계를 완료하세요."
        if self.result_img is not None:
            return "결과를 확인하세요. 다음 타일은 [새 검사]를 누른 뒤 타일을 바꿔 놓으세요."
        return "불 끄고 타일을 놓은 뒤 [촬영 + 판정]"

    @property
    def wants_preview(self):
        return self.result_img is None

    def static_image(self):
        return self.result_img

    def build(self, f):
        row = ttk.Frame(f); row.pack(fill="x")
        ttk.Label(row, text="이름:").pack(side="left")
        self.name_var = tk.StringVar(value=self.default_name())
        ttk.Entry(row, textvariable=self.name_var, width=24).pack(side="left", padx=3)
        self.btn(row, "촬영 + 판정", self.inspect)
        self.btn(row, "새 검사", self.new)
        self.btn(row, "자세한 분석 그림", self.open_detail, need_hw=False)
        self.btn(row, "결과 폴더 열기", self.open_folder, need_hw=False)
        self.verdict_lbl = tk.Label(f, text="", font=(FONT, 20, "bold"))
        self.verdict_lbl.pack(anchor="w", pady=4)
        hist = ttk.Frame(f); hist.pack(fill="x")
        ttk.Label(hist, text="지난 검사 (클릭하면 결과 보기):").pack(anchor="w")
        self.hist = tk.Listbox(hist, height=4, font=(FONT, 10))
        self.hist.pack(fill="x")
        self.hist.bind("<<ListboxSelect>>", self.show_history)
        self.load_history()

    @staticmethod
    def default_name():
        return time.strftime("tile_%Y%m%d_%H%M%S")

    def load_history(self):
        items = sorted((p for p in glob.glob(os.path.join(ps_run.OUT_ROOT, "*", "fast", "verdict.png"))
                        if not os.path.basename(os.path.dirname(os.path.dirname(p))).startswith("_")),   # 자가 진단 제외
                       key=os.path.getmtime, reverse=True)[:50]
        self.hist_items = items
        self.hist.delete(0, "end")
        for p in items:
            name = os.path.basename(os.path.dirname(os.path.dirname(p)))
            crack = self._has_crack(os.path.dirname(p))
            t = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
            self.hist.insert("end", f"{t}   {name}   →  {'균열' if crack else '무늬'}")

    @staticmethod
    def _has_crack(fast_dir):
        p = os.path.join(fast_dir, "crack_mask.npy")
        return bool(np.load(p).any()) if os.path.exists(p) else False

    def show_history(self, _e=None):
        sel = self.hist.curselection()
        if not sel:
            return
        p = self.hist_items[sel[0]]
        self.show_result(os.path.dirname(p))

    def show_result(self, fast_dir):
        img = cv2.imread(os.path.join(fast_dir, "verdict.png"))
        self.result_img = img
        self.last_dir = fast_dir
        crack = self._has_crack(fast_dir)
        self.verdict_lbl.config(text="균열이 있습니다" if crack else "균열 없음 — 무늬입니다",
                                fg="#d62728" if crack else COL_OK)
        self.app.update_hint()

    def new(self):
        self.result_img = None
        self.verdict_lbl.config(text="")
        self.name_var.set(self.default_name())
        self.app.update_hint()

    def on_enter(self):
        meta = os.path.join(ps_run.REF_DIR, "meta.json")
        if not os.path.exists(meta):
            return
        exp = json.load(open(meta))["exposure"]
        self.app.run_task("카메라 준비 중...", lambda: (self.hw.apply_settings(exp, ps_run.CAM_GAIN), self.hw.led(1)))

    def on_leave(self):
        if self.hw.connected:
            self.hw.led(0)

    def inspect(self):
        st = step_status()
        missing = [n for k, n in (("camcal", "1"), ("ledcal", "2"), ("ref", "3")) if st[k] != "ok"]
        if missing:
            return self.app.warn(f"먼저 {', '.join(missing)}단계를 완료하세요.")
        name = self.name_var.get().strip()
        if not name or any(c in name for c in '\\/:*?"<>|'):
            return self.app.warn("이름에 쓸 수 없는 글자가 있습니다.")
        obj_dir = os.path.join(ps_run.OBJ_ROOT, name)
        if os.path.exists(obj_dir) and not messagebox.askyesno("확인", f"'{name}' 이(가) 이미 있습니다. 덮어쓸까요?"):
            return
        exp = json.load(open(os.path.join(ps_run.REF_DIR, "meta.json")))["exposure"]

        def work():
            print(f"\n[검사: {name}]")
            warn = capture_stack(self.hw, obj_dir, exp, "검사")
            print("판정 계산 중...")
            judge.run_solve(name, show=False)
            self.hw.led(1)
            if warn:
                print("\n주의: " + " / ".join(warn))
            self.app.post(lambda: (self.show_result(os.path.join(ps_run.OUT_ROOT, name, "fast")),
                                   self.load_history()))
        self.app.run_task("촬영 + 판정 중 (약 40초, 건드리지 마세요)...", work)

    def open_detail(self):
        d = getattr(self, "last_dir", None)
        p = os.path.join(d, "result_fast.png") if d else None
        if p and os.path.exists(p):
            os.startfile(p)
        else:
            self.app.warn("먼저 검사를 하거나 지난 검사를 선택하세요.")

    def open_folder(self):
        d = getattr(self, "last_dir", None) or ps_run.OUT_ROOT
        os.makedirs(d, exist_ok=True)
        os.startfile(d)


# ════════════════════════════════════════════════════════════
# 메인 창
# ════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PS 균열 검사기")
        self.geometry("1400x940")
        self.minsize(1200, 820)
        style = ttk.Style(self)
        style.configure(".", font=(FONT, 10))
        style.configure("TButton", padding=(8, 4))
        self.hw = Hardware()
        self.logq = queue.Queue()
        self.uiq = queue.Queue()
        self.busy = False
        self.pages = [ConnectPage(self), CamCalPage(self), LedCalPage(self), RefPage(self), InspectPage(self)]
        self.page = None
        self._photo = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, self._poll)
        self.after(80, self._tick)
        self.after(200, self._startup)

    # ── 레이아웃 ──
    def _build(self):
        top = tk.Frame(self, bg="#1f2937")
        top.pack(fill="x")
        tk.Label(top, text="PS 균열 검사기", fg="white", bg="#1f2937", font=(FONT, 16, "bold")).pack(side="left", padx=14, pady=8)
        self.conn_lbl = tk.Label(top, text="", fg="white", bg="#1f2937", font=(FONT, 10))
        self.conn_lbl.pack(side="right", padx=14)

        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        nav = ttk.Frame(body, padding=10); nav.pack(side="left", fill="y")
        ttk.Label(nav, text="단계", font=(FONT, 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.nav_items = {}
        for p in self.pages:
            fr = tk.Frame(nav, cursor="hand2", bd=1, relief="solid", bg="white")
            fr.pack(fill="x", pady=3)
            t = tk.Label(fr, text=p.title, font=(FONT, 11, "bold"), bg="white", anchor="w", width=22)
            t.pack(fill="x", padx=8, pady=(6, 0))
            s = tk.Label(fr, text="", font=(FONT, 10), bg="white", anchor="w")
            s.pack(fill="x", padx=8, pady=(0, 6))
            for w in (fr, t, s):
                w.bind("<Button-1>", lambda e, p=p: self.show(p))
            self.nav_items[p.key] = (fr, t, s)
        ttk.Label(nav, text="\n처음 설치: 0 → 1 → 2 → 3 → 4\n평소 검사: 0 → 4",
                  font=(FONT, 10), foreground="#555").pack(anchor="w")

        main = ttk.Frame(body, padding=(4, 10, 10, 10)); main.pack(side="left", fill="both", expand=True)
        self.head = ttk.Label(main, text="", font=(FONT, 15, "bold"))
        self.head.pack(anchor="w")
        upper = ttk.Frame(main); upper.pack(fill="both", expand=True)
        left = ttk.Frame(upper); left.pack(side="left", fill="both", expand=False)
        self.canvas = tk.Canvas(left, width=PREVIEW_W, height=PREVIEW_H, bg="black", highlightthickness=0)
        self.canvas.pack(pady=(6, 4))
        self.canvas.bind("<Button-1>", lambda e: self._click(e, 1))
        self.canvas.bind("<Button-3>", lambda e: self._click(e, 3))
        self.canvas.bind("<Motion>", self._motion)
        self.controls = ttk.Frame(left); self.controls.pack(fill="x")

        right = ttk.Frame(upper, padding=(10, 6, 0, 0)); right.pack(side="left", fill="both", expand=True)
        hint_box = tk.Frame(right, bg="#fff7d6", bd=1, relief="solid")
        hint_box.pack(fill="x")
        tk.Label(hint_box, text="다음에 할 일", bg="#fff7d6", font=(FONT, 11, "bold")).pack(anchor="w", padx=8, pady=(6, 0))
        self.hint = tk.Label(hint_box, text="", bg="#fff7d6", font=(FONT, 11), wraplength=380, justify="left", anchor="w")
        self.hint.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(right, text="설명", font=(FONT, 11, "bold")).pack(anchor="w", pady=(10, 2))
        self.instr = tk.Text(right, wrap="word", font=(FONT, 10), height=18, relief="flat", bg="#f6f6f6")
        self.instr.pack(fill="both", expand=True)

        bottom = ttk.Frame(main); bottom.pack(fill="x")
        self.busy_lbl = ttk.Label(bottom, text="", font=(FONT, 11, "bold"), foreground="#1d4ed8")
        self.busy_lbl.pack(anchor="w")
        self.logbox = tk.Text(bottom, height=9, font=("Consolas", 9), bg="#111", fg="#ddd", wrap="word")
        self.logbox.pack(fill="x")

    # ── 페이지 전환 ──
    def show(self, page):
        if self.busy:
            return self.warn("작업이 끝날 때까지 기다려 주세요.")
        if self.page is not None and self.hw.connected:
            try:
                self.page.on_leave()
            except Exception:
                pass
        self.page = page
        for w in self.controls.winfo_children():
            w.destroy()
        self.head.config(text=page.title)
        self.instr.config(state="normal")
        self.instr.delete("1.0", "end")
        self.instr.insert("1.0", page.instructions())
        self.instr.config(state="disabled")
        page.build(self.controls)
        self.update_nav()
        if self.hw.connected:
            page.on_enter()
        elif page.key != "connect":
            self.log("장치가 연결되어 있지 않습니다. '0. 장치 연결'을 먼저 하세요.\n")

    def update_nav(self):
        st = step_status()
        for p in self.pages:
            fr, t, s = self.nav_items[p.key]
            sel = p is self.page
            bg = "#dbeafe" if sel else "white"
            for w in (fr, t, s):
                w.config(bg=bg)
            if p.key == "connect":
                ok = self.hw.connected
                s.config(text="● 연결됨" if ok else "● 연결 필요", fg=COL_OK if ok else COL_TODO)
            elif p.key == "inspect":
                ready = all(st[k] == "ok" for k in ("camcal", "ledcal", "ref"))
                s.config(text="● 검사 가능" if ready else "● 앞 단계 필요", fg=COL_OK if ready else COL_TODO)
            else:
                v = st[p.key]
                s.config(text={"ok": "● 완료", "stale": "● 다시 필요 (앞 단계 변경됨)", "todo": "● 필요"}[v],
                         fg={"ok": COL_OK, "stale": COL_STALE, "todo": COL_TODO}[v])
        self.conn_lbl.config(text=(f"카메라 #{self.hw.cam_index} · LED {self.hw.port}" if self.hw.connected
                                   else "장치 연결 안 됨"))
        self.update_hint()

    def update_hint(self):
        if self.page is not None:
            self.hint.config(text=self.page.next_hint())

    # ── 작업 실행 (스레드) ──
    def guard(self, fn, need_hw=True):
        if self.busy:
            return self.warn("작업이 끝날 때까지 기다려 주세요.")
        if need_hw and self.page is not None and self.page.key != "connect" and not self.hw.connected:
            return self.warn("장치가 연결되어 있지 않습니다. '0. 장치 연결'을 먼저 하세요.")
        try:
            fn()
        except Exception as e:
            self.warn(str(e))

    def run_task(self, label, fn, done=None):
        if not self.hw.connected and self.page is not None and self.page.key != "connect":
            return
        self.busy = True
        self.busy_lbl.config(text=("⏳ " + label) if label else "")
        writer = QueueWriter(self.logq)

        def runner():
            err = None
            try:
                with contextlib.redirect_stdout(writer):
                    fn()
            except Exception as e:
                err = e
                self.logq.put(traceback.format_exc() if not isinstance(e, (RuntimeError, TimeoutError)) else "")
            self.uiq.put(lambda: self._task_done(err, done))
        threading.Thread(target=runner, daemon=True).start()

    def _task_done(self, err, done):
        self.busy = False
        self.busy_lbl.config(text="")
        if err is not None:
            self.log(f"\n✖ {err}\n")
            self.warn(str(err))
        elif done:
            done()
        self.update_nav()

    def post(self, fn):
        self.uiq.put(fn)

    def log(self, s):
        self.logq.put(s)

    def warn(self, msg):
        messagebox.showwarning("안내", msg)

    def _poll(self):
        try:
            while True:
                s = self.logq.get_nowait()
                self.logbox.insert("end", s)
                self.logbox.see("end")
                try:
                    with open(LOG_PATH, "a", encoding="utf-8") as fh:
                        fh.write(s)
                except OSError:
                    pass
        except queue.Empty:
            pass
        try:
            while True:
                self.uiq.get_nowait()()
        except queue.Empty:
            pass
        self.after(50, self._poll)

    # ── 미리보기 ──
    def _tick(self):
        try:
            img = None
            p = self.page
            if p is not None:
                if p.wants_preview and self.hw.connected:
                    f = self.hw.latest()
                    if f is not None:
                        img = p.overlay(f.copy())
                elif hasattr(p, "static_image"):
                    img = p.static_image()
            if img is not None:
                h, w = img.shape[:2]
                s = min(PREVIEW_W / w, PREVIEW_H / h)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                self._scale = s
                self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
            elif p is not None and not self.hw.connected:
                self.canvas.delete("all")
                self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2, text="장치가 연결되지 않았습니다",
                                        fill="#aaa", font=(FONT, 16))
        except Exception as e:
            self.log(f"(미리보기 오류: {e})\n")
        self.after(80, self._tick)

    def _to_img(self, e):
        s = getattr(self, "_scale", PREVIEW_W / W)
        return e.x / s, e.y / s

    def _click(self, e, button):
        if self.page is not None and not self.busy:
            x, y = self._to_img(e)
            if 0 <= x < W and 0 <= y < H:
                self.page.on_click(x, y, button)

    def _motion(self, e):
        if self.page is not None:
            x, y = self._to_img(e)
            self.page.on_motion(x, y)

    # ── 시작 / 종료 ──
    def _startup(self):
        self.show(self.pages[0])
        self.log(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} PS 균열 검사기 시작 =====\n")
        if os.path.exists(CONFIG_PATH):
            cfg = json.load(open(CONFIG_PATH))
            port, cam = cfg.get("serial_port"), cfg.get("camera_index")
            self.log(f"저장된 설정으로 연결 시도: LED {port}, 카메라 #{cam}\n")
            pg = self.pages[0]

            def done():
                pg.refresh()
                st = step_status()
                if all(st[k] == "ok" for k in ("camcal", "ledcal", "ref")):
                    self.show(self.pages[4])
            self.run_task("저장된 설정으로 연결 중...", lambda: pg._finish_connect(port, cam), done)
        else:
            self.log("처음 실행입니다. [자동으로 찾아 연결]을 누르세요.\n")

    def on_close(self):
        if self.busy and not messagebox.askyesno("종료", "작업이 진행 중입니다. 그래도 종료할까요?"):
            return
        try:
            self.hw.disconnect()
        finally:
            self.destroy()


def selftest():
    """자가 진단 (화면 없이): 저장된 설정으로 연결 → 촬영 → 판정 → 결과를 ps_app_log.txt 에 기록"""
    with open(LOG_PATH, "a", encoding="utf-8") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 자가 진단 =====")
        hw = Hardware()
        try:
            cfg = json.load(open(CONFIG_PATH))
            hw.connect(cfg["serial_port"], cfg["camera_index"])
            print(f"연결: LED {hw.port}, 카메라 #{hw.cam_index}")
            st = step_status()
            print(f"캘리브레이션 상태: {st}")
            exp = json.load(open(os.path.join(ps_run.REF_DIR, "meta.json")))["exposure"]
            capture_stack(hw, os.path.join(ps_run.OBJ_ROOT, "_selftest"), exp, "진단")
            judge.run_solve("_selftest", show=False)
            print("자가 진단 완료: 정상")
        except Exception:
            print("자가 진단 실패:\n" + traceback.format_exc())
        finally:
            hw.disconnect()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        App().mainloop()
