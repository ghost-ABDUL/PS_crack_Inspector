"""
PS 균열 검사기 실행 파일 빌드

  python build_release.py

결과
  dist/PS_Inspector/            이 PC용 (현재 캘리브레이션·연결 설정 포함 → 바로 검사 가능)
  dist/PS_Inspector_release.zip 고객 배포용 (캘리브레이션 없음 → 고객 장치에서 0~3단계 진행)
"""
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(BASE, "dist")
APP = os.path.join(DIST, "PS_Inspector")

EXCLUDE = ["tensorflow", "keras", "tensorboard", "torch", "IPython", "jupyter_client", "jupyter_core",
           "notebook", "pandas", "PyQt5", "PyQt6", "PySide6"]
COMMON_FILES = ["charuco_11x7_6mm_A4.pdf", "사용설명서.txt"]               # 고객에게도 필요
CALIB_FILES = ["camera_intrinsics.npz", "camera_intrinsics.yaml", "led_positions.npz", "led_positions.txt",
               "ps_config.json", "calib_plane.png"]                        # 이 장치 전용
CALIB_DIRS = ["ps_ref", "calib_images", "led_calib"]


def copy_newer(src, dst):
    """dst가 없거나 src가 더 최근일 때만 복사 (exe에서 새로 한 캘리브레이션을 덮어쓰지 않도록)"""
    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
        shutil.copy2(src, dst)


def copy_into(dst, files, dirs=()):
    for f in files:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            copy_newer(src, os.path.join(dst, f))
    for d in dirs:
        src_root = os.path.join(BASE, d)
        for root, _, names in os.walk(src_root):
            out = os.path.join(dst, d, os.path.relpath(root, src_root))
            os.makedirs(out, exist_ok=True)
            for n in names:
                copy_newer(os.path.join(root, n), os.path.join(out, n))


PROGRAM = {"PS_Inspector.exe", "_internal"}   # PyInstaller가 다시 만드는 것 — 나머지는 사용자 데이터


def main():
    # PyInstaller가 dist/PS_Inspector를 통째로 지우므로 검사 결과·캘리브레이션을 잠시 옮겨 둔다
    keep = os.path.join(DIST, "_keep")
    if os.path.isdir(APP):
        shutil.rmtree(keep, ignore_errors=True)
        os.makedirs(keep)
        for n in os.listdir(APP):
            if n not in PROGRAM:
                shutil.move(os.path.join(APP, n), os.path.join(keep, n))

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--windowed", "--name", "PS_Inspector",
           "--distpath", DIST, "--workpath", os.path.join(BASE, "build", "pyi"),
           "--hidden-import", "serial.tools.list_ports", "--collect-submodules", "skimage.morphology"]
    for m in EXCLUDE:
        cmd += ["--exclude-module", m]
    cmd.append(os.path.join(BASE, "ps_app.py"))
    print("빌드 중... (1~2분)")
    try:
        subprocess.run(cmd, check=True, cwd=BASE)
    finally:
        if os.path.isdir(keep):
            os.makedirs(APP, exist_ok=True)
            for n in os.listdir(keep):
                if not os.path.exists(os.path.join(APP, n)):
                    shutil.move(os.path.join(keep, n), os.path.join(APP, n))
            shutil.rmtree(keep, ignore_errors=True)

    # 고객 배포용: 프로그램 + 공통 파일만
    rel = os.path.join(DIST, "release", "PS_Inspector")
    shutil.rmtree(os.path.join(DIST, "release"), ignore_errors=True)
    shutil.copytree(APP, rel)
    copy_into(rel, COMMON_FILES)
    zip_path = shutil.make_archive(os.path.join(DIST, "PS_Inspector_release"), "zip",
                                   os.path.join(DIST, "release"), "PS_Inspector")
    shutil.rmtree(os.path.join(DIST, "release"))

    # 이 PC용: 캘리브레이션·설정까지
    copy_into(APP, COMMON_FILES + CALIB_FILES, CALIB_DIRS)

    print(f"\n완료\n  이 PC용   : {APP}\\PS_Inspector.exe\n  고객 배포용: {zip_path}")


if __name__ == "__main__":
    main()
