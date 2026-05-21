"""
================================================================
2026 지능IoT해커톤 - 문제 2 : 위성 이미지 건물 탐지
SOLID + ETA + Cloud H100 단일 GPU 안전 실행 버전
================================================================
목표:
  - 클라우드 H100 1대(80GB) 환경에서 첫 실행에서도 오류 없이 끝까지 진행.
  - 패키지/시스템 라이브러리 누락 자동 복구 (libGL, opencv-headless 등).
  - CUDA 환경 사전 검사로 mismatch를 학습 전에 잡아냄.
  - VRAM은 peak ~50GB로 80GB 한도 안 안전 유지.
  - 얼리스탑(patience) + OOM 자동 강하 + 시간 예산 자동 종료.

기본 실행 (권장 - 클라우드 H100 단일 GPU, 5시간 예산):
  python Q2fix180_cloud.py --preset h100-cloud-5h --device cuda --force-cuda

데이터 폴더가 자동 탐색 안 되면:
  python Q2fix180_cloud.py --preset h100-cloud-5h --device cuda --force-cuda \
      --data-dir /workspace/data

더 보수적인 VRAM (다른 작업과 GPU 공유 시):
  python Q2fix180_cloud.py --preset h100-cloud-5h-safe --device cuda --force-cuda

빠른 환경 진단만 (실제 학습 없이):
  python Q2fix180_cloud.py --inspect

필수 디렉토리 구조 (자동 탐색됨):
  data/
    train_images/   (*.tif)
    train_labels/   (*.geojson)
    test_images/    (*.tif)
    building_detection_template.csv

자동 탐색 위치:
  - --data-dir 인자
  - 스크립트 폴더/data
  - 현재 작업 폴더 (CWD) / data
  - /workspace/data, /content/data, /root/data, /data, /mnt/data
  - 위 폴더의 building_detection_template.csv 재귀 검색

얼리스탑:
  - Ultralytics 내장 patience 기반 (기본 60).
  - val fitness가 60 epoch 동안 개선 X → 자동 종료 후 best.pt 저장.

OOM 안전 장치:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (HBM3 fragmentation 방지)
  - cache='disk' 강제 (RAM 부담 ↓)
  - plots=False (matplotlib 메모리 회피)
  - OOM 발생 시 batch /= 2 자동 재시도
  - 연속 OOM 2회 시 imgsz도 한 단계 강하
================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# ──────────────────────────────────────────────────────────────
# [클라우드 안전] torch import 이전에 메모리 할당자 옵션을 강제 설정.
#   - expandable_segments: H100 HBM3 fragmentation 방지 (OOM 빈도 ↓)
#   - max_split_size_mb: 큰 블록 분할 한계로 NMS 메모리 spike 완화
# ──────────────────────────────────────────────────────────────
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ.setdefault("OMP_NUM_THREADS", "4")
# DataLoader 워커 hang 방지 (특히 도커 컨테이너)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Ultralytics 자동 업데이트 끄기 (네트워크 의존성 ↓)
os.environ.setdefault("YOLO_OFFLINE", "False")
os.environ.setdefault("ULTRALYTICS_OFFLINE_MODE", "0")

# [클라우드 안전] Jupyter/REPL 환경에서 __file__ 미정의 대비
try:
    _SCRIPT_FILE = Path(__file__).resolve()
except NameError:
    _SCRIPT_FILE = Path.cwd() / "Q2fix180_cloud.py"
    print(f"[INIT] __file__ 미정의 환경 → CWD 기준으로 진행: {_SCRIPT_FILE.parent}")

import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 1. 패키지 설치 확인
# ──────────────────────────────────────────────────────────────
class PackageInstaller:
    """필수 패키지 설치 + 시스템 라이브러리 확인. 클라우드 환경에서 첫 실행 안정성 보강."""

    REQUIRED = {
        # 모듈명 : (pip 패키지명, 추가 import 검증 함수 또는 None)
        "ultralytics": ("ultralytics", None),
        "cv2": ("opencv-python-headless", None),
        "rasterio": ("rasterio", None),
        "shapely": ("shapely", None),
        "torch": ("torch", None),  # torch 누락 시 명확한 안내
    }

    PIP_RETRIES = 3

    @classmethod
    def install_if_missing(cls) -> None:
        print(f"[INIT] Python {platform.python_version()} | {platform.system()} {platform.release()}")
        print(f"[INIT] 작업 디렉터리: {Path.cwd()}")

        for module, (pkg, _) in cls.REQUIRED.items():
            if cls._try_import(module):
                continue

            # torch는 자동 설치하지 않음 (CUDA 빌드를 사용자가 골라야 함)
            if module == "torch":
                raise RuntimeError(
                    "PyTorch(torch)가 설치되어 있지 않습니다. CUDA 11/12에 맞춰 직접 설치하세요.\n"
                    "  CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
                    "  CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                )

            print(f"[설치] {pkg} 설치 시도...")
            ok = cls._pip_install(pkg)
            if not ok:
                raise RuntimeError(f"{pkg} 설치 실패. 수동 설치 후 다시 실행하세요: pip install {pkg}")

            # opencv 설치 후 libGL 누락 시 자동 복구
            if module == "cv2" and not cls._try_import("cv2"):
                cls._try_install_libgl()
                if not cls._try_import("cv2"):
                    raise RuntimeError(
                        "cv2 import 실패. opencv-python-headless를 설치했으나 시스템 라이브러리 누락 가능.\n"
                        "Ubuntu 기반: sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0"
                    )

        print("[패키지] 모든 패키지 확인 완료\n")

    @staticmethod
    def _try_import(module: str) -> bool:
        try:
            __import__(module)
            return True
        except Exception as e:
            # ImportError 외에 ModuleNotFoundError, OSError(libGL) 등도 처리
            print(f"  [import 실패] {module}: {type(e).__name__}: {e}")
            return False

    @classmethod
    def _pip_install(cls, pkg: str) -> bool:
        for attempt in range(1, cls.PIP_RETRIES + 1):
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "-q", "--disable-pip-version-check"],
                    timeout=600,
                )
                return True
            except subprocess.CalledProcessError as e:
                print(f"  [pip 재시도 {attempt}/{cls.PIP_RETRIES}] {pkg} 실패 (code={e.returncode})")
            except subprocess.TimeoutExpired:
                print(f"  [pip 재시도 {attempt}/{cls.PIP_RETRIES}] {pkg} 타임아웃")
            except Exception as e:
                print(f"  [pip 재시도 {attempt}/{cls.PIP_RETRIES}] {pkg} 예외: {e}")
            time.sleep(2)
        return False

    @staticmethod
    def _try_install_libgl() -> None:
        """Ubuntu/Debian 기반 클라우드에서 libGL 자동 설치 (sudo 권한 있을 때만)."""
        if platform.system() != "Linux":
            return
        # apt-get이 있는 환경에서만 시도
        if shutil.which("apt-get") is None:
            return
        print("  [시스템] libGL 누락 추정 → apt-get으로 libgl1, libglib2.0-0 설치 시도...")
        # root 권한 확인
        is_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
        prefix = [] if is_root else (["sudo", "-n"] if shutil.which("sudo") else [])
        cmds = [
            prefix + ["apt-get", "update", "-qq"],
            prefix + ["apt-get", "install", "-y", "-qq", "libgl1", "libglib2.0-0"],
        ]
        for cmd in cmds:
            try:
                subprocess.check_call(cmd, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except Exception as e:
                print(f"  [시스템] 자동 설치 실패 ({' '.join(cmd[:3])}...): {e}")
                print("  [시스템] 수동으로 다음을 실행하세요: sudo apt-get install -y libgl1 libglib2.0-0")
                return
        print("  [시스템] libGL 설치 완료.")


PackageInstaller.install_if_missing()

import cv2  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import ColorInterp  # noqa: E402


# ──────────────────────────────────────────────────────────────
# 2. 설정
# ──────────────────────────────────────────────────────────────
class CFG:
    """기본값 모음. CLI 인자가 들어오면 PipelineRunner에서 필요한 값만 대체한다."""

    # 경로 (Jupyter/REPL 환경에서도 안전한 _SCRIPT_FILE 사용)
    PROJECT_ROOT = _SCRIPT_FILE.parent
    BASE_DIR = PROJECT_ROOT / "data"
    TRAIN_IMG_DIR = BASE_DIR / "train_images"
    TRAIN_LBL_DIR = BASE_DIR / "train_labels"
    TEST_IMG_DIR = BASE_DIR / "test_images"
    TEMPLATE_CSV = BASE_DIR / "building_detection_template.csv"

    YOLO_ROOT = PROJECT_ROOT / "yolo_dataset_quality"
    RUNS_DIR = PROJECT_ROOT / "runs"
    RUN_NAME_BASE = "building_detector_quality"
    SUBMISSION_CSV = PROJECT_ROOT / "submission.csv"

    # 결과 중심 기본값
    IMG_SIZE = 1024
    VAL_RATIO = 0.15
    SEED = 42
    ENSEMBLE_SEEDS = "42,77"

    # 학습 (클라우드 H100 단일 GPU 안전치)
    MODEL = "yolo11x.pt"
    EPOCHS = 200
    BATCH = 8                  # H100 80GB에서 peak ~50GB로 안전
    PATIENCE = 50
    WORKERS = 4                 # 클라우드 컨테이너 DataLoader 안정성
    DEVICE = "auto"
    FORCE_CUDA = False
    AMP = True
    MAX_DET = 10000             # [수정] 30000→15000: val NMS 메모리 안전
    TIME_BUDGET_MIN = 0.0
    RESERVE_SUBMIT_MIN = 20.0

    # 추론/보정
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.50
    CALIB_CONFS = tuple(np.round(np.arange(0.03, 0.551, 0.02), 2))
    CALIB_IOUS = tuple(np.round(np.arange(0.25, 0.751, 0.05), 2))
    FAST_CALIBRATION = True

    # 작은 객체/위성 이미지 보강
    MIN_BOX_PX = 5.0
    SPLIT_MODE = "group-stratified"
    INFER_MODE = "tile"
    TILE_SIZE = 768
    TILE_OVERLAP = 0.20
    TILE_NMS_IOU = 0.50
    TILE_CALIB_CONFS = tuple(np.round(np.concatenate([np.arange(0.02, 0.121, 0.01), np.arange(0.14, 0.301, 0.04)]), 2))
    TILE_CALIB_IOUS = tuple(np.round(np.arange(0.25, 0.751, 0.05), 2))
    # [버그수정] degrees=45는 axis-aligned bbox를 √2배 부풀려 라벨을 손상시킴.
    #            90/180/270° 회전은 TTA에서 정확히 처리 → 학습 회전은 0이 안전.
    AUG_DEGREES = 0.0
    AUG_MOSAIC = 0.55
    AUG_SCALE = 0.40
    AUG_TRANSLATE = 0.08
    AUG_MIXUP = 0.05            # 소규모 데이터셋 regularization
    AUG_COPY_PASTE = 0.10
    CLOSE_MOSAIC = 30

    @classmethod
    def set_base_dir(cls, base_dir: Path) -> None:
        cls.BASE_DIR = Path(base_dir).expanduser().resolve()
        cls.TRAIN_IMG_DIR = cls.BASE_DIR / "train_images"
        cls.TRAIN_LBL_DIR = cls.BASE_DIR / "train_labels"
        cls.TEST_IMG_DIR = cls.BASE_DIR / "test_images"
        cls.TEMPLATE_CSV = cls.BASE_DIR / "building_detection_template.csv"

    @classmethod
    def set_work_dir(cls, work_dir: Path) -> None:
        work_dir = Path(work_dir).expanduser().resolve()
        cls.YOLO_ROOT = work_dir
        cls.RUNS_DIR = work_dir
        cls.SUBMISSION_CSV = work_dir / "submission.csv"

    @classmethod
    def ensure_writable_dirs(cls) -> None:
        """클라우드 환경에서 권한/경로 문제로 생성 실패 시 CWD로 fallback."""
        for attr, path in [("YOLO_ROOT", cls.YOLO_ROOT), ("RUNS_DIR", cls.RUNS_DIR)]:
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                test = Path(path) / ".write_test"
                test.write_text("ok"); test.unlink()
            except Exception as e:
                fallback = Path.cwd() / Path(path).name
                print(f"[INIT] {attr}={path} 쓰기 불가 ({e}) → {fallback}로 fallback")
                fallback.mkdir(parents=True, exist_ok=True)
                setattr(cls, attr, fallback)
                if attr == "YOLO_ROOT":
                    cls.SUBMISSION_CSV = Path.cwd() / "submission.csv"


@dataclass(frozen=True)
class RuntimePlan:
    """실행 전에 계산 가능한 작업량 요약."""

    train_images: int
    test_images: int
    val_images_est: int
    seeds: int
    models: int
    train_jobs: int
    calibration_grid: int
    tta_count: int
    approx_tiles_per_image: int
    test_prediction_units: int
    calibration_prediction_units: int


# ──────────────────────────────────────────────────────────────
# 3. 예상 시간 / 경과 시간 출력
# ──────────────────────────────────────────────────────────────
class RuntimeEstimator:
    """
    실행 시간 관찰과 ETA 출력을 담당한다.

    - 학습처럼 내부 epoch 진행 시간을 직접 알기 어려운 작업은 모델 1개가 끝난 뒤 평균 기반으로 남은 시간을 추정한다.
    - 보정/추론처럼 루프 단위가 보이는 작업은 중간 ETA를 출력한다.
    """

    def __init__(self, report_every: int = 10):
        self.start_time = time.perf_counter()
        self.stage_starts: dict[str, float] = {}
        self.job_durations: dict[str, list[float]] = {}
        self.report_every = max(1, int(report_every))

    @staticmethod
    def fmt(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "계산 중"
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}시간 {m}분 {s}초"
        if m > 0:
            return f"{m}분 {s}초"
        return f"{s}초"

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def print_plan(self, plan: RuntimePlan) -> None:
        print("=" * 60)
        print("실행 작업량 / 예상 시간 산정 기준")
        print("=" * 60)
        print(f"  학습 이미지 수              : {plan.train_images}")
        print(f"  예상 validation 이미지 수    : {plan.val_images_est}")
        print(f"  테스트 이미지 수            : {plan.test_images}")
        print(f"  모델 수 × seed 수           : {plan.models} × {plan.seeds} = {plan.train_jobs}개 학습 job")
        print(f"  conf/iou 보정 grid 수        : {plan.calibration_grid}개 조합 / 모델")
        print(f"  TTA 개수                    : {plan.tta_count}")
        print(f"  이미지당 예상 tile 수        : {plan.approx_tiles_per_image}")
        print(f"  보정 예측 단위 대략          : {plan.calibration_prediction_units:,}")
        print(f"  테스트 예측 단위 대략        : {plan.test_prediction_units:,}")
        print("  실제 시간은 첫 학습/보정/추론이 끝난 뒤 평균 기반 ETA로 갱신됩니다.")
        print("=" * 60 + "\n")

    def start_stage(self, name: str) -> None:
        self.stage_starts[name] = time.perf_counter()
        print("=" * 60)
        print(f"[TIME] 시작: {name} | 전체 경과 {self.fmt(self.elapsed())}")
        print("=" * 60)

    def end_stage(self, name: str) -> float:
        start = self.stage_starts.get(name, time.perf_counter())
        duration = time.perf_counter() - start
        print(f"[TIME] 완료: {name} | 단계 소요 {self.fmt(duration)} | 전체 경과 {self.fmt(self.elapsed())}\n")
        return duration

    def start_job(self, kind: str) -> float:
        self.stage_starts[f"job::{kind}"] = time.perf_counter()
        return self.stage_starts[f"job::{kind}"]

    def end_job(self, kind: str, done: int, total: int) -> None:
        start = self.stage_starts.get(f"job::{kind}", time.perf_counter())
        duration = time.perf_counter() - start
        self.job_durations.setdefault(kind, []).append(duration)
        avg = float(np.mean(self.job_durations[kind]))
        remain_jobs = max(0, total - done)
        eta = avg * remain_jobs
        print(
            f"[ETA] {kind}: {done}/{total} 완료 | "
            f"이번 {self.fmt(duration)} | 평균 {self.fmt(avg)} | 남은 예상 {self.fmt(eta)}"
        )

    def progress(self, label: str, done: int, total: int, started_at: float) -> None:
        if total <= 0:
            return
        if done != total and done % self.report_every != 0:
            return
        elapsed = time.perf_counter() - started_at
        per_unit = elapsed / max(1, done)
        remain = per_unit * max(0, total - done)
        print(f"[ETA] {label}: {done}/{total} | 경과 {self.fmt(elapsed)} | 남은 예상 {self.fmt(remain)}")


# ──────────────────────────────────────────────────────────────
# 4. CUDA / 디바이스
# ──────────────────────────────────────────────────────────────
class DeviceManager:
    """CUDA/CPU 선택, AMP/half, 메모리 정리 및 사전 진단까지 담당."""

    @staticmethod
    def preflight_cuda() -> None:
        """학습 시작 전 CUDA 환경 사전 검사. 명확한 에러 메시지로 빠른 실패 유도."""
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch가 없습니다. CUDA 12.1 빌드 설치:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )

        if not torch.cuda.is_available():
            # nvidia-smi가 동작하는지 확인 (드라이버는 있는데 torch가 CPU 빌드일 수도)
            smi_out = ""
            if shutil.which("nvidia-smi"):
                try:
                    smi_out = subprocess.check_output(["nvidia-smi", "-L"], timeout=10).decode().strip()
                except Exception:
                    smi_out = ""
            if smi_out:
                raise RuntimeError(
                    "nvidia-smi는 GPU를 인식하는데 torch.cuda.is_available()=False.\n"
                    f"nvidia-smi 결과:\n{smi_out}\n\n"
                    "현재 설치된 torch가 CPU 전용이거나 CUDA 버전이 안 맞습니다. 재설치:\n"
                    "  pip uninstall -y torch torchvision\n"
                    "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                )
            else:
                raise RuntimeError(
                    "CUDA를 사용할 수 없습니다. GPU가 보이지 않습니다.\n"
                    "- nvidia-smi를 실행해 드라이버 상태를 확인하세요.\n"
                    "- 클라우드 인스턴스가 GPU 노드인지 확인하세요."
                )

        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError(
                "torch.cuda.is_available()=True 인데 device_count()=0. "
                "드라이버/runtime mismatch. nvidia-smi와 torch.version.cuda 버전을 맞춰 재설치하세요."
            )

        # H100 외 다른 GPU에서도 동작은 하지만 안내
        for i in range(n):
            prop = torch.cuda.get_device_properties(i)
            if "H100" not in prop.name and "A100" not in prop.name and "RTX" not in prop.name:
                print(f"[DEVICE] GPU {i}={prop.name} (H100/A100/RTX 외 GPU - preset 시간 산정과 다를 수 있음)")

    @staticmethod
    def resolve_device(requested_device: str = CFG.DEVICE, force_cuda: bool = CFG.FORCE_CUDA) -> str:
        requested = str(requested_device).strip().lower()

        if requested == "cpu":
            print("[DEVICE] 사용자가 CPU 모드를 명시했습니다.")
            return "cpu"

        try:
            import torch
        except ImportError as e:
            msg = (
                "PyTorch(torch)가 설치되어 있지 않습니다. CUDA 지원 PyTorch가 필요합니다.\n"
                "예: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            )
            if force_cuda or requested in {"cuda", "gpu"} or requested.isdigit() or "," in requested:
                raise RuntimeError(msg) from e
            print(f"[DEVICE] {msg}\n[DEVICE] CPU 모드로 진행합니다.")
            return "cpu"

        print("=" * 60)
        print("CUDA / PyTorch 환경 확인")
        print("=" * 60)
        print(f"  torch version       : {torch.__version__}")
        print(f"  torch cuda build    : {torch.version.cuda}")
        print(f"  cuda available      : {torch.cuda.is_available()}")
        print(f"  alloc conf          : {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(미설정)')}")

        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            print(f"  cuda device count   : {n_gpu}")
            for i in range(n_gpu):
                prop = torch.cuda.get_device_properties(i)
                vram_gb = prop.total_memory / (1024**3)
                print(f"  GPU {i}              : {prop.name} ({vram_gb:.1f} GB)")

            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

            # multi-GPU 요청이라도 단일 GPU 환경이면 자동 다운그레이드 (이 파일은 단일 GPU용)
            if "," in requested:
                parts = [x.strip() for x in requested.split(",") if x.strip() != ""]
                if len(parts) > n_gpu:
                    fallback = parts[0] if parts else "0"
                    print(f"[DEVICE] 요청 device={requested}인데 가용 GPU는 {n_gpu}개 → 단일 GPU {fallback}로 자동 변경")
                    device = fallback
                else:
                    device = ",".join(parts)
            elif requested.isdigit():
                if int(requested) >= n_gpu:
                    print(f"[DEVICE] 요청 device={requested}가 가용 GPU 인덱스({n_gpu-1})를 초과 → GPU 0으로 변경")
                    device = "0"
                else:
                    device = requested
            else:
                device = "0"  # "auto"/"cuda"/"gpu" → 첫 번째 GPU

            print(f"[DEVICE] CUDA 사용: device={device}\n")
            return device

        diagnosis = [
            "CUDA 사용 불가: torch.cuda.is_available() == False",
            f"torch cuda build: {torch.version.cuda}",
            "가능한 원인: CPU 전용 PyTorch 설치, NVIDIA 드라이버/CUDA 런타임 문제, GPU 미탑재 환경",
        ]

        if force_cuda or requested in {"cuda", "gpu"} or requested.isdigit() or "," in requested:
            raise RuntimeError("\n".join(diagnosis + [
                "--force-cuda 또는 CUDA device를 요청했으므로 CPU fallback 없이 중단합니다.",
                "CUDA 지원 PyTorch를 설치한 뒤 다시 실행하세요.",
            ]))

        print("[DEVICE] " + " | ".join(diagnosis))
        print("[DEVICE] CPU 모드로 진행합니다. CUDA를 반드시 쓰려면 --device cuda --force-cuda 를 사용하세요.\n")
        return "cpu"

    @staticmethod
    def cuda_amp_enabled(device: str, no_amp: bool = False) -> bool:
        return False if no_amp else str(device).lower() != "cpu"

    @staticmethod
    def use_half_precision(device: str, no_half: bool = False) -> bool:
        return False if no_half else str(device).lower() != "cpu"

    @staticmethod
    def clear_cuda_memory() -> None:
        """GC + 모든 GPU의 캐시 정리. 학습 job 사이 메모리 fragmentation 회피."""
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        with torch.cuda.device(i):
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    @staticmethod
    def print_vram_status(prefix: str = "") -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                return
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used_gb = (total - free) / (1024**3)
                total_gb = total / (1024**3)
                print(f"  {prefix}GPU {i} VRAM: {used_gb:.1f} / {total_gb:.1f} GB")
        except Exception:
            pass

    @staticmethod
    def optimize_torch_runtime(device: str) -> None:
        if str(device).lower() == "cpu":
            return
        try:
            import torch

            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            print("[DEVICE] CUDA 최적화 적용: cudnn.benchmark=True, TF32=True")
        except Exception as e:
            print(f"[DEVICE] CUDA 최적화 생략: {e}")


# ──────────────────────────────────────────────────────────────
# 4-1. 실제 데이터/라벨 경로 확인
# ──────────────────────────────────────────────────────────────
class DataDirResolver:
    """문제2 데이터 폴더를 확인하거나 자동 탐색한다. 클라우드 환경 대응 강화."""

    REQUIRED = ("train_images", "train_labels", "test_images", "building_detection_template.csv")

    # 클라우드 환경에서 데이터가 흔히 마운트되는 위치들
    CLOUD_PATHS = [
        Path("/workspace/data"), Path("/workspace"),
        Path("/content/data"), Path("/content"),  # Colab
        Path("/root/data"), Path("/root"),
        Path("/data"), Path("/mnt/data"), Path("/mnt"),
        Path("/home/data"),
    ]

    @classmethod
    def resolve(cls, requested: str | None, default_dir: Path) -> Path:
        candidates: list[Path] = []
        if requested:
            candidates.append(Path(requested))
        candidates.extend([
            default_dir,
            Path.cwd() / "data",
            Path.cwd(),
            CFG.PROJECT_ROOT / "data",
            CFG.PROJECT_ROOT,
        ])
        candidates.extend(cls.CLOUD_PATHS)

        for path in candidates:
            try:
                path = Path(path).expanduser()
                if cls.is_valid(path):
                    print(f"[DATA] 데이터 폴더 확인: {path}")
                    return path.resolve()
            except Exception:
                continue

        # 재귀 검색 (마지막 수단)
        search_roots = [CFG.PROJECT_ROOT, Path.cwd()] + cls.CLOUD_PATHS + [Path.home()]
        for root in search_roots:
            if not root.exists():
                continue
            try:
                for template in root.rglob("building_detection_template.csv"):
                    path = template.parent
                    if cls.is_valid(path):
                        print(f"[DATA] 자동 탐색으로 데이터 폴더를 찾았습니다: {path}")
                        return path.resolve()
            except (PermissionError, OSError):
                continue
            except Exception:
                continue

        searched = "\n  ".join(str(c) for c in candidates[:8])
        raise FileNotFoundError(
            f"문제2 데이터 폴더를 찾지 못했습니다.\n"
            f"탐색한 경로 (일부):\n  {searched}\n"
            f"--data-dir로 다음 4개가 모두 있는 폴더를 지정하세요:\n"
            f"  {', '.join(cls.REQUIRED)}"
        )

    @classmethod
    def is_valid(cls, path: Path) -> bool:
        path = Path(path)
        return path.exists() and all((path / name).exists() for name in cls.REQUIRED)


class LabelPathResolver:
    """이미지 stem에 대응하는 GeoJSON 라벨 파일명을 찾는다."""

    @staticmethod
    def find(train_label_dir: Path, image_stem: str) -> Path | None:
        candidates = [
            train_label_dir / f"{image_stem}.geojson",
            train_label_dir / f"{image_stem}_buildings.geojson",
        ]
        for path in candidates:
            if path.exists():
                return path

        matches = sorted(train_label_dir.glob(f"{image_stem}*.geojson"))
        return matches[0] if matches else None


# ──────────────────────────────────────────────────────────────
# 5. 이미지 / GeoJSON / YOLO label 변환
# ──────────────────────────────────────────────────────────────
class ImageIO:
    """이미지 읽기/쓰기 책임."""

    @staticmethod
    def read_tif(path: Path, fallback_size: int = CFG.IMG_SIZE) -> np.ndarray:
        try:
            with rasterio.open(path) as src:
                band_count = src.count
                h, w = src.height, src.width

                if band_count >= 3:
                    color_map = {ci: idx + 1 for idx, ci in enumerate(src.colorinterp)}
                    r_band = color_map.get(ColorInterp.red, 1)
                    g_band = color_map.get(ColorInterp.green, 2)
                    b_band = color_map.get(ColorInterp.blue, 3)
                    r = src.read(r_band).astype(np.float32)
                    g = src.read(g_band).astype(np.float32)
                    b = src.read(b_band).astype(np.float32)

                    def norm(arr: np.ndarray) -> np.ndarray:
                        mn, mx = arr.min(), arr.max()
                        if mx == mn:
                            return np.zeros_like(arr, dtype=np.uint8)
                        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

                    img = cv2.merge([norm(b), norm(g), norm(r)])
                else:
                    gray = src.read(1).astype(np.float32)
                    mn, mx = gray.min(), gray.max()
                    if mx > mn:
                        gray = ((gray - mn) / (mx - mn) * 255).astype(np.uint8)
                    else:
                        gray = np.zeros((h, w), dtype=np.uint8)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return img
        except Exception as e:
            print(f"  [WARN] rasterio 실패({path.name}): {e} → OpenCV fallback")
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return np.zeros((fallback_size, fallback_size, 3), dtype=np.uint8)
            return img

    @staticmethod
    def get_tif_size(path: Path, fallback_size: int = CFG.IMG_SIZE) -> tuple[int, int]:
        try:
            with rasterio.open(path) as src:
                return int(src.width), int(src.height)
        except Exception:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return fallback_size, fallback_size
            h, w = img.shape[:2]
            return int(w), int(h)

    @staticmethod
    def save_as_png(tif_path: Path, dst_path: Path, fallback_size: int = CFG.IMG_SIZE) -> None:
        img = ImageIO.read_tif(tif_path, fallback_size=fallback_size)
        cv2.imwrite(str(dst_path), img)


class GeoJsonParser:
    """GeoJSON 파싱 책임."""

    @classmethod
    def load(cls, path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls._parse_raw(raw)

    @classmethod
    def count(cls, path: Path) -> int:
        if not path.exists():
            return 0
        try:
            return len(cls.load(path))
        except Exception:
            return 0

    @classmethod
    def _parse_raw(cls, raw) -> list[dict]:
        buildings = []

        if isinstance(raw, dict) and raw.get("type") == "FeatureCollection":
            return cls._parse_raw(raw.get("features", []))

        if isinstance(raw, dict) and raw.get("type") == "Feature":
            geom = raw.get("geometry", {})
            props = raw.get("properties", {}) or {}
            coords = cls._extract_coords(geom)
            return [{**props, "coordinates": coords}] if coords else []

        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "Feature":
                    geom = item.get("geometry", {})
                    props = item.get("properties", {}) or {}
                    coords = cls._extract_coords(geom)
                    if coords:
                        buildings.append({**props, "coordinates": coords})
                else:
                    coords = cls._normalize_coords(item.get("coordinates", []))
                    if coords:
                        buildings.append({**item, "coordinates": coords})
            return buildings

        if isinstance(raw, dict):
            for key in ("buildings", "features", "items", "data"):
                nested = raw.get(key)
                if isinstance(nested, list):
                    return cls._parse_raw(nested)
            coords = cls._normalize_coords(raw.get("coordinates", []))
            if coords:
                return [{**raw, "coordinates": coords}]

        return []

    @classmethod
    def _extract_coords(cls, geometry: dict) -> list[list]:
        if not geometry:
            return []
        gtype = geometry.get("type", "")
        coords = geometry.get("coordinates", [])

        if gtype == "Polygon":
            ring = coords[0] if coords else []
            return cls._normalize_coords(ring)

        if gtype == "MultiPolygon":
            best = max(coords, key=lambda poly: len(poly[0])) if coords else []
            ring = best[0] if best else []
            return cls._normalize_coords(ring)

        return cls._normalize_coords(coords)

    @staticmethod
    def _normalize_coords(coords) -> list[list]:
        if not coords:
            return []
        while (
            isinstance(coords, list)
            and len(coords) == 1
            and isinstance(coords[0], list)
            and coords[0]
            and isinstance(coords[0][0], list)
        ):
            coords = coords[0]
        if not coords or isinstance(coords[0], (int, float)):
            return []
        return [[float(p[0]), float(p[1])] for p in coords if len(p) >= 2]


class YoloLabelConverter:
    """폴리곤 좌표를 YOLO label로 바꾸는 책임."""

    @classmethod
    def polygon_to_yolo(
        cls,
        coords: list[list],
        img_w: int,
        img_h: int,
        min_box_px: float = 0.0,
    ) -> tuple[float, float, float, float] | None:
        if not coords or len(coords) < 3:
            return None

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_min = float(np.clip(x_min, 0, img_w - 1))
        x_max = float(np.clip(x_max, 0, img_w - 1))
        y_min = float(np.clip(y_min, 0, img_h - 1))
        y_max = float(np.clip(y_max, 0, img_h - 1))

        if x_max <= x_min or y_max <= y_min:
            return None

        min_box_px = max(0.0, float(min_box_px))
        if min_box_px > 0:
            x_min, x_max = cls._expand_interval(x_min, x_max, min_box_px, img_w)
            y_min, y_max = cls._expand_interval(y_min, y_max, min_box_px, img_h)

        if (x_max - x_min) < 1.0 or (y_max - y_min) < 1.0:
            return None

        cx = (x_min + x_max) / 2 / img_w
        cy = (y_min + y_max) / 2 / img_h
        bw = (x_max - x_min) / img_w
        bh = (y_max - y_min) / img_h

        return (
            float(np.clip(cx, 0.0, 1.0)),
            float(np.clip(cy, 0.0, 1.0)),
            float(np.clip(bw, 1e-4, 1.0)),
            float(np.clip(bh, 1e-4, 1.0)),
        )

    @staticmethod
    def _expand_interval(low: float, high: float, min_size: float, limit: int) -> tuple[float, float]:
        size = high - low
        if size >= min_size or limit <= 1:
            return low, high

        target = min(float(min_size), float(limit - 1))
        center = (low + high) / 2.0
        new_low = center - target / 2.0
        new_high = center + target / 2.0

        if new_low < 0:
            new_high -= new_low
            new_low = 0.0
        if new_high > limit - 1:
            shift = new_high - (limit - 1)
            new_low = max(0.0, new_low - shift)
            new_high = float(limit - 1)
        return float(new_low), float(new_high)

    @classmethod
    def geojson_to_yolo_txt(
        cls,
        geojson_path: Path,
        out_txt: Path,
        img_w: int,
        img_h: int,
        min_box_px: float = 0.0,
    ) -> int:
        try:
            buildings = GeoJsonParser.load(geojson_path)
        except Exception as e:
            print(f"  [ERROR] {geojson_path.name} 로드 실패: {e}")
            out_txt.write_text("", encoding="utf-8")
            return 0

        lines = []
        for b in buildings:
            coords = b.get("coordinates", [])
            if not coords:
                continue
            result = cls.polygon_to_yolo(coords, img_w=img_w, img_h=img_h, min_box_px=min_box_px)
            if result is None:
                continue
            cx, cy, bw, bh = result
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out_txt.write_text("\n".join(lines), encoding="utf-8")
        return len(lines)

    @staticmethod
    def count_yolo_labels(label_path: Path) -> int:
        if not label_path.exists():
            return 0
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            return 0
        return sum(1 for line in text.splitlines() if line.strip())


# ──────────────────────────────────────────────────────────────
# 6. 데이터셋 구성
# ──────────────────────────────────────────────────────────────
class DatasetLayout:
    """YOLO 데이터셋 디렉토리 경로 생성 책임."""

    @staticmethod
    def dirs(yolo_dir: Path) -> dict[str, Path]:
        return {
            "train_img": yolo_dir / "images" / "train",
            "train_lbl": yolo_dir / "labels" / "train",
            "val_img": yolo_dir / "images" / "val",
            "val_lbl": yolo_dir / "labels" / "val",
        }


class SplitStrategy:
    """train/val index 생성 책임. OCP: split 전략이 늘어나면 이 클래스만 확장."""

    def __init__(self, train_label_dir: Path):
        self.train_label_dir = train_label_dir

    def quick_geojson_count(self, tif_path: Path) -> int:
        geojson_path = LabelPathResolver.find(self.train_label_dir, tif_path.stem)
        return GeoJsonParser.count(geojson_path) if geojson_path else 0

    def make_indices(
        self,
        tif_files: list[Path],
        seed: int,
        val_ratio: float,
        split_mode: str = CFG.SPLIT_MODE,
    ) -> tuple[set[int], set[int], list[int]]:
        n = len(tif_files)
        rng = np.random.default_rng(seed)
        n_val_total = max(1, int(round(n * val_ratio)))
        label_counts = [self.quick_geojson_count(t) for t in tif_files]

        if split_mode == "random" or n < 5:
            idx = rng.permutation(n)
            val_idx = set(idx[-n_val_total:])
            train_idx = set(idx[:-n_val_total])
            return train_idx, val_idx, label_counts

        if split_mode == "group-stratified":
            return self._make_group_stratified_indices(tif_files, label_counts, rng, val_ratio)

        try:
            ranks = pd.Series(label_counts).rank(method="first")
            q = min(5, max(2, n_val_total))
            bins = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
        except Exception:
            idx = rng.permutation(n)
            val_idx = set(idx[-n_val_total:])
            train_idx = set(idx[:-n_val_total])
            return train_idx, val_idx, label_counts

        val_idx: set[int] = set()
        for b in sorted(pd.Series(bins).dropna().unique()):
            group = [i for i, bb in enumerate(bins) if bb == b]
            rng.shuffle(group)
            take = max(1, int(round(len(group) * val_ratio)))
            val_idx.update(group[:take])

        all_indices = list(range(n))
        if len(val_idx) > n_val_total:
            drop_candidates = list(val_idx)
            rng.shuffle(drop_candidates)
            for i in drop_candidates[: len(val_idx) - n_val_total]:
                val_idx.remove(i)
        elif len(val_idx) < n_val_total:
            remain = [i for i in all_indices if i not in val_idx]
            rng.shuffle(remain)
            for i in remain[: n_val_total - len(val_idx)]:
                val_idx.add(i)

        train_idx = set(i for i in all_indices if i not in val_idx)
        return train_idx, val_idx, label_counts

    @staticmethod
    def _group_key(tif_path: Path) -> str:
        return tif_path.stem.split("_", 1)[0]

    def _make_group_stratified_indices(
        self,
        tif_files: list[Path],
        label_counts: list[int],
        rng: np.random.Generator,
        val_ratio: float,
    ) -> tuple[set[int], set[int], list[int]]:
        groups: dict[str, list[int]] = {}
        for idx, tif_path in enumerate(tif_files):
            groups.setdefault(self._group_key(tif_path), []).append(idx)

        group_keys = sorted(groups)
        n_val_groups = max(1, int(round(len(group_keys) * val_ratio)))
        group_scores = {g: float(np.mean([label_counts[i] for i in groups[g]])) for g in group_keys}

        try:
            ranks = pd.Series([group_scores[g] for g in group_keys]).rank(method="first")
            q = min(5, max(2, n_val_groups))
            bins = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
            val_groups: set[str] = set()
            for b in sorted(pd.Series(bins).dropna().unique()):
                candidates = [g for g, bb in zip(group_keys, bins) if bb == b]
                rng.shuffle(candidates)
                val_groups.add(candidates[0])
            if len(val_groups) > n_val_groups:
                drop = list(val_groups)
                rng.shuffle(drop)
                val_groups = set(drop[:n_val_groups])
            elif len(val_groups) < n_val_groups:
                remain = [g for g in group_keys if g not in val_groups]
                rng.shuffle(remain)
                val_groups.update(remain[: n_val_groups - len(val_groups)])
        except Exception:
            shuffled = list(group_keys)
            rng.shuffle(shuffled)
            val_groups = set(shuffled[:n_val_groups])

        val_idx = {i for g in val_groups for i in groups[g]}
        train_idx = {i for i in range(len(tif_files)) if i not in val_idx}
        print(f"  group-stratified split: val groups={sorted(val_groups)}")
        return train_idx, val_idx, label_counts


class DatasetBuilder:
    """GeoJSON/TIF 원본을 YOLO 데이터셋으로 구성하는 책임."""

    def __init__(self, cfg: type[CFG], image_io: type[ImageIO], split_strategy: SplitStrategy, estimator: RuntimeEstimator):
        self.cfg = cfg
        self.image_io = image_io
        self.split_strategy = split_strategy
        self.estimator = estimator

    def prepare(
        self,
        seed: int,
        yolo_dir: Path,
        val_ratio: float,
        overwrite: bool,
        split_mode: str,
        min_box_px: float,
    ) -> Path:
        stage_name = f"STEP 1 데이터셋 변환 seed={seed}"
        self.estimator.start_stage(stage_name)

        if overwrite and yolo_dir.exists():
            print(f"  기존 YOLO 데이터셋 삭제 후 재생성: {yolo_dir}")
            shutil.rmtree(yolo_dir)

        dirs = DatasetLayout.dirs(yolo_dir)
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(self.cfg.TRAIN_IMG_DIR.glob("*.tif"))
        if not tif_files:
            raise FileNotFoundError(
                f"학습 이미지를 찾을 수 없습니다: {self.cfg.TRAIN_IMG_DIR}\n"
                "data/train_images/ 폴더에 *.tif 파일을 넣어주세요."
            )

        train_set, val_set, pre_label_counts = self.split_strategy.make_indices(
            tif_files=tif_files,
            seed=seed,
            val_ratio=val_ratio,
            split_mode=split_mode,
        )

        print(f"  전체: {len(tif_files)}개 | train: {len(train_set)}개 | val: {len(val_set)}개 | split_mode={split_mode}")
        print(f"  작은 건물 라벨 보정: min_box_px={float(min_box_px):.1f}")
        if pre_label_counts:
            train_counts = [pre_label_counts[i] for i in sorted(train_set)]
            val_counts = [pre_label_counts[i] for i in sorted(val_set)]
            print(f"  train count 평균/중앙값: {np.mean(train_counts):.1f}/{np.median(train_counts):.1f}")
            print(f"  val   count 평균/중앙값: {np.mean(val_counts):.1f}/{np.median(val_counts):.1f}")

        total_buildings = 0
        missing_labels = 0
        split_rows = []
        loop_start = time.perf_counter()

        for i, tif_path in enumerate(tqdm(tif_files, desc="  변환"), start=1):
            zero_based = i - 1
            stem = tif_path.stem
            is_train = zero_based in train_set
            dst_img_dir = dirs["train_img"] if is_train else dirs["val_img"]
            dst_lbl_dir = dirs["train_lbl"] if is_train else dirs["val_lbl"]

            dst_img = dst_img_dir / f"{stem}.png"
            self.image_io.save_as_png(tif_path, dst_img, fallback_size=self.cfg.IMG_SIZE)

            dst_lbl = dst_lbl_dir / f"{stem}.txt"
            geojson_path = LabelPathResolver.find(self.cfg.TRAIN_LBL_DIR, stem)
            img_w, img_h = self.image_io.get_tif_size(tif_path, fallback_size=self.cfg.IMG_SIZE)

            if geojson_path is not None and geojson_path.exists():
                n = YoloLabelConverter.geojson_to_yolo_txt(
                    geojson_path,
                    dst_lbl,
                    img_w=img_w,
                    img_h=img_h,
                    min_box_px=min_box_px,
                )
                total_buildings += n
            else:
                dst_lbl.write_text("", encoding="utf-8")
                n = 0
                missing_labels += 1

            split_rows.append({
                "image": stem,
                "split": "train" if is_train else "val",
                "label_count": n,
                "width": img_w,
                "height": img_h,
            })
            self.estimator.progress("데이터 변환", i, len(tif_files), loop_start)

        if missing_labels > 0:
            print(f"\n  [WARN] GeoJSON 없는 이미지: {missing_labels}개 (빈 레이블 처리)")

        split_df = pd.DataFrame(split_rows)
        split_csv = yolo_dir / "split_info.csv"
        split_df.to_csv(split_csv, index=False)
        (yolo_dir / "split_seed.txt").write_text(str(seed), encoding="utf-8")
        (yolo_dir / "label_config.json").write_text(
            json.dumps({"min_box_px": float(min_box_px)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\n  변환 완료 | 총 건물 bbox: {total_buildings:,}개")
        print("  split별 라벨 수 요약:")
        print(split_df.groupby("split")["label_count"].agg(["count", "sum", "mean", "min", "max"]).to_string())

        yaml_path = yolo_dir / "dataset.yaml"
        yaml_content = (
            f"path: {yolo_dir.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n\n"
            "nc: 1\n"
            "names: ['building']\n"
        )
        yaml_path.write_text(yaml_content, encoding="utf-8")
        print(f"  dataset.yaml → {yaml_path}\n")

        self.estimator.end_stage(stage_name)
        return yaml_path


# ──────────────────────────────────────────────────────────────
# 7. 학습
# ──────────────────────────────────────────────────────────────
class TrainingResultSummarizer:
    """Ultralytics 결과 CSV 요약 책임."""

    @staticmethod
    def completed_epochs(save_dir: Path) -> int | None:
        results_csv = save_dir / "results.csv"
        if not results_csv.exists():
            return None
        try:
            df = pd.read_csv(results_csv)
        except Exception:
            return None
        return int(len(df)) if not df.empty else 0

    @staticmethod
    def summarize(save_dir: Path, expected_epochs: int) -> None:
        results_csv = save_dir / "results.csv"
        if not results_csv.exists():
            print("  [INFO] results.csv를 찾지 못해 학습 요약을 생략합니다.")
            return

        try:
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:
            print(f"  [WARN] results.csv 읽기 실패: {e}")
            return

        if df.empty:
            print("  [INFO] results.csv가 비어 있어 학습 요약을 생략합니다.")
            return

        ran_epochs = len(df)
        metric_candidates = [
            "metrics/mAP50-95(B)",
            "metrics/mAP50(B)",
            "metrics/recall(B)",
            "metrics/precision(B)",
        ]
        metric_name = next((c for c in metric_candidates if c in df.columns), None)

        print("\n  [학습 요약]")
        print(f"    실제 수행 epoch: {ran_epochs} / 요청 epoch: {expected_epochs}")
        print("    조기 종료: 발생한 것으로 판단됨" if ran_epochs < expected_epochs else "    조기 종료: 발생하지 않음 또는 최대 epoch까지 학습 완료")

        if metric_name is not None:
            metric_values = pd.to_numeric(df[metric_name], errors="coerce")
            if metric_values.notna().any():
                best_idx = int(metric_values.idxmax())
                best_epoch = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else best_idx + 1
                best_score = float(metric_values.loc[best_idx])
                print(f"    기준 검증 지표: {metric_name}")
                print(f"    best epoch: {best_epoch}")
                print(f"    best validation score: {best_score:.6f}")


class ModelTrainer:
    """Ultralytics 모델 학습 책임."""

    def __init__(self, cfg: type[CFG], estimator: RuntimeEstimator):
        self.cfg = cfg
        self.estimator = estimator

    @staticmethod
    def find_existing_weight(save_dir: Path) -> Path | None:
        for name in ("best.pt", "last.pt"):
            weight = save_dir / "weights" / name
            if weight.exists():
                return weight
        return None

    def train_one(
        self,
        yaml_path: Path,
        device: str,
        model_name: str,
        epochs: int,
        batch: int,
        patience: int,
        amp: bool,
        deterministic: bool,
        cache_mode: str,
        seed: int,
        run_name: str,
        imgsz: int,
        max_det: int,
        aug_degrees: float,
        aug_mosaic: float,
        aug_scale: float,
        aug_translate: float,
        aug_mixup: float,
        aug_copy_paste: float,
        close_mosaic: int,
        workers: int = None,
        save_plots: bool = True,
        resume: bool = False,
        reuse_short_runs: bool = False,
    ) -> Path:
        from ultralytics import YOLO

        save_dir = self.cfg.RUNS_DIR / run_name
        if resume:
            existing = self.find_existing_weight(save_dir)
            if existing is not None:
                done_epochs = TrainingResultSummarizer.completed_epochs(save_dir)
                if done_epochs is None or done_epochs >= epochs or reuse_short_runs:
                    print(f"[RESUME] 기존 학습 run 재사용: {existing}")
                    if done_epochs is not None and done_epochs < epochs:
                        print(
                            f"[RESUME] 기존 run은 {done_epochs}/{epochs} epoch입니다. "
                            "--reuse-short-runs가 켜져 있어 시간 절약을 위해 현재 best/last 가중치를 재사용합니다."
                        )
                    TrainingResultSummarizer.summarize(save_dir, expected_epochs=epochs)
                    return existing

                exact_run_name = f"{run_name}_e{epochs}"
                exact_save_dir = self.cfg.RUNS_DIR / exact_run_name
                exact_existing = self.find_existing_weight(exact_save_dir)
                exact_done = TrainingResultSummarizer.completed_epochs(exact_save_dir) if exact_existing is not None else None
                if exact_existing is not None and (exact_done is None or exact_done >= epochs):
                    print(f"[RESUME] 목표 epoch를 만족하는 run 재사용: {exact_existing}")
                    TrainingResultSummarizer.summarize(exact_save_dir, expected_epochs=epochs)
                    return exact_existing

                print(
                    f"[RESUME] 기존 run은 {done_epochs}/{epochs} epoch라 목표보다 짧습니다. "
                    f"정확한 {epochs} epoch 앙상블을 위해 새 run으로 학습합니다."
                )
                run_name = exact_run_name
                save_dir = exact_save_dir
                if save_dir.exists():
                    suffix = time.strftime("%Y%m%d_%H%M%S")
                    run_name = f"{exact_run_name}_{suffix}"
                    save_dir = self.cfg.RUNS_DIR / run_name
                    print(f"[RESUME] 기존 e{epochs} run과 충돌하지 않도록 새 이름 사용: {run_name}")
                resume = False

        # 클라우드 안전 기본값
        if workers is None:
            workers = self.cfg.WORKERS
        effective_workers = max(0, int(workers))
        plots_flag = bool(save_plots)

        current_batch = int(batch)
        attempt = 1
        consecutive_oom = 0

        while True:
            if save_dir.exists() and not resume:
                print(f"  기존 run 삭제 후 재학습: {save_dir}")
                shutil.rmtree(save_dir)

            print("=" * 60)
            print(f"STEP 2 : Ultralytics YOLO 학습 | seed={seed} | attempt={attempt}")
            print(f"  모델: {model_name} | imgsz: {imgsz} | 최대 epoch: {epochs} | batch: {current_batch}")
            print(f"  device: {device} | AMP: {amp} | deterministic: {deterministic} | cache: {cache_mode} | patience: {patience} | max_det: {max_det} | plots: {plots_flag}")
            print(f"  workers: {effective_workers} | mixup: {aug_mixup} | copy_paste: {aug_copy_paste}")
            print("  학습 중 epoch별 ETA는 Ultralytics 로그를 참고하고, 학습 job 종료 후 전체 ETA가 갱신됩니다.")
            print("=" * 60)

            DeviceManager.print_vram_status(prefix="[학습 전] ")

            try:
                model = YOLO(model_name)
                model.train(
                    data=str(yaml_path),
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=current_batch,
                    device=device,
                    workers=effective_workers,
                    project=str(self.cfg.RUNS_DIR),
                    name=run_name,
                    exist_ok=True,
                    patience=patience,
                    seed=seed,
                    deterministic=deterministic,
                    verbose=True,
                    cache=False if cache_mode == "none" else cache_mode,
                    single_cls=True,
                    max_det=max_det,
                    # 위성 이미지 특화 증강 (degrees=0 권장: axis-aligned bbox 보호)
                    degrees=float(aug_degrees),
                    flipud=0.5,
                    fliplr=0.5,
                    mosaic=float(aug_mosaic),
                    scale=float(aug_scale),
                    translate=float(aug_translate),
                    shear=0.0,
                    perspective=0.0,
                    hsv_h=0.01,
                    hsv_s=0.35,
                    hsv_v=0.35,
                    mixup=float(aug_mixup),
                    copy_paste=float(aug_copy_paste),
                    # 학습 설정
                    optimizer="AdamW",
                    lr0=6e-4,
                    lrf=0.005,
                    weight_decay=7e-4,
                    warmup_epochs=8,
                    cos_lr=True,
                    # 손실 가중치
                    box=7.5,
                    cls=0.5,
                    dfl=1.5,
                    # 후반 안정화
                    close_mosaic=int(close_mosaic),
                    amp=amp,
                    val=True,
                    save=True,
                    save_period=-1,
                    plots=plots_flag,
                )
                # 학습 종료 후 명시적 메모리 정리
                del model
                DeviceManager.clear_cuda_memory()
                DeviceManager.print_vram_status(prefix="[학습 후] ")
                break
            except RuntimeError as e:
                msg = str(e).lower()
                is_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
                if is_oom and current_batch > 1:
                    next_batch = max(1, current_batch // 2)
                    print("\n  [OOM WARN] CUDA 메모리 부족으로 추정되어 batch를 낮춰 재시도합니다.")
                    print(f"             batch {current_batch} → {next_batch}\n")
                    current_batch = next_batch
                    attempt += 1
                    consecutive_oom += 1
                    DeviceManager.clear_cuda_memory()
                    # 연속 OOM 2회면 imgsz도 한 단계 강하 (마지막 보루)
                    if consecutive_oom >= 2 and imgsz > 640:
                        new_imgsz = max(640, ((imgsz - 128) // 32) * 32)
                        print(f"  [OOM FALLBACK] 연속 OOM {consecutive_oom}회 → imgsz {imgsz} → {new_imgsz}로 추가 강하")
                        imgsz = new_imgsz
                    continue
                # OOM이 아니거나 batch=1까지 갔는데도 실패 → 원본 예외 그대로
                print("\n[FATAL] 학습 중 복구 불가능한 오류:")
                traceback.print_exc()
                raise

        best_pt = save_dir / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = save_dir / "weights" / "last.pt"

        TrainingResultSummarizer.summarize(save_dir, expected_epochs=epochs)
        print(f"\n  학습 완료 → 가중치: {best_pt}\n")
        return best_pt


# ──────────────────────────────────────────────────────────────
# 8. 추론 엔진
# ──────────────────────────────────────────────────────────────
class TileHelper:
    """타일 좌표 생성과 NMS 책임."""

    @staticmethod
    def make_tiles(height: int, width: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
        tile_size = int(tile_size)
        if tile_size <= 0 or (height <= tile_size and width <= tile_size):
            return [(0, 0, width, height)]

        overlap = float(np.clip(overlap, 0.0, 0.8))
        step = max(1, int(round(tile_size * (1.0 - overlap))))

        def starts(length: int) -> list[int]:
            if length <= tile_size:
                return [0]
            values = list(range(0, max(1, length - tile_size + 1), step))
            last = length - tile_size
            if values[-1] != last:
                values.append(last)
            return sorted(set(values))

        xs = starts(width)
        ys = starts(height)
        return [(x, y, min(x + tile_size, width), min(y + tile_size, height)) for y in ys for x in xs]

    @staticmethod
    def count_after_nms(
        boxes_xyxy_score: np.ndarray,
        nms_iou: float,
        max_det: int,
        score_threshold: float = 0.0,
    ) -> int:
        if boxes_xyxy_score.size == 0:
            return 0

        boxes_xyxy_score = boxes_xyxy_score[boxes_xyxy_score[:, 4] >= float(score_threshold)]
        if boxes_xyxy_score.size == 0:
            return 0

        boxes = boxes_xyxy_score[:, :4].astype(float)
        scores = boxes_xyxy_score[:, 4].astype(float).tolist()
        xywh = []
        for x1, y1, x2, y2 in boxes:
            w = max(1.0, float(x2 - x1))
            h = max(1.0, float(y2 - y1))
            xywh.append([float(x1), float(y1), w, h])

        indices = cv2.dnn.NMSBoxes(
            bboxes=xywh,
            scores=scores,
            score_threshold=float(score_threshold),
            nms_threshold=float(nms_iou),
            top_k=int(max_det) if max_det else 0,
        )
        if indices is None or len(indices) == 0:
            return 0
        return int(len(np.array(indices).reshape(-1)))


class TTAFactory:
    """TTA 이미지 생성 책임."""

    @staticmethod
    def make(img: np.ndarray, level: str) -> list[tuple[str, np.ndarray]]:
        level = level.lower()
        if level == "none":
            return [("orig", img)]
        if level == "light":
            return [
                ("orig", img),
                ("fliplr", cv2.flip(img, 1)),
                ("flipud", cv2.flip(img, 0)),
                ("rot180", cv2.rotate(img, cv2.ROTATE_180)),
            ]
        if level == "strong":
            return [
                ("orig", img),
                ("fliplr", cv2.flip(img, 1)),
                ("flipud", cv2.flip(img, 0)),
                ("flipboth", cv2.flip(img, -1)),
                ("rot90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
                ("rot180", cv2.rotate(img, cv2.ROTATE_180)),
                ("rot270", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ("transpose", cv2.transpose(img)),
            ]
        raise ValueError("tta-level은 none, light, strong 중 하나여야 합니다.")

    @staticmethod
    def count(level: str) -> int:
        return {"none": 1, "light": 4, "strong": 8}.get(str(level).lower(), 1)


class DetectionPredictor:
    """단일 모델의 full/tile/hybrid count 예측 책임."""

    def __init__(self, device: str, half: bool, estimator: RuntimeEstimator | None = None):
        self.device = device
        self.half = half
        self.estimator = estimator

    def predict_boxes_raw(self, model, img: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int) -> np.ndarray:
        preds = model.predict(
            source=img,
            conf=float(conf),
            iou=float(iou),
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            augment=False,
            max_det=max_det,
            half=self.half,
        )
        if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        boxes = preds[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        return np.concatenate([xyxy, scores], axis=1)

    def predict_count_raw(self, model, img: np.ndarray, conf: float, iou: float, imgsz: int, max_det: int) -> int:
        return int(len(self.predict_boxes_raw(model, img, conf, iou, imgsz, max_det)))

    def predict_boxes_tiled(
        self,
        model,
        img: np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        tile_size: int,
        tile_overlap: float,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        all_boxes = []
        for x1, y1, x2, y2 in TileHelper.make_tiles(h, w, tile_size=tile_size, overlap=tile_overlap):
            tile = img[y1:y2, x1:x2]
            boxes = self.predict_boxes_raw(model, tile, conf, iou, imgsz, max_det)
            if boxes.size == 0:
                continue
            boxes[:, [0, 2]] += x1
            boxes[:, [1, 3]] += y1
            boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
            all_boxes.append(boxes)

        if not all_boxes:
            return np.empty((0, 5), dtype=np.float32)
        return np.concatenate(all_boxes, axis=0)

    def predict_count_tiled(
        self,
        model,
        img: np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        tile_size: int,
        tile_overlap: float,
        tile_nms_iou: float,
    ) -> int:
        merged = self.predict_boxes_tiled(
            model=model,
            img=img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        if merged.size == 0:
            return 0
        return TileHelper.count_after_nms(merged, nms_iou=tile_nms_iou, max_det=max_det)

    def predict_count(
        self,
        model,
        img: np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        infer_mode: str,
        tile_size: int,
        tile_overlap: float,
        tile_nms_iou: float,
    ) -> int:
        infer_mode = str(infer_mode).lower()
        if infer_mode == "full":
            return self.predict_count_raw(model, img, conf, iou, imgsz, max_det)
        if infer_mode == "tile":
            return self.predict_count_tiled(model, img, conf, iou, imgsz, max_det, tile_size, tile_overlap, tile_nms_iou)
        if infer_mode == "hybrid":
            full_n = self.predict_count_raw(model, img, conf, iou, imgsz, max_det)
            tile_n = self.predict_count_tiled(model, img, conf, iou, imgsz, max_det, tile_size, tile_overlap, tile_nms_iou)
            return int(round(0.35 * full_n + 0.65 * tile_n))
        raise ValueError("infer-mode는 full, tile, hybrid 중 하나여야 합니다.")


# ──────────────────────────────────────────────────────────────
# 9. Calibration / Inference / Ensemble / Submission
# ──────────────────────────────────────────────────────────────
class ThresholdCalibrator:
    """validation count 기준 conf/iou 보정 책임."""

    def __init__(self, cfg: type[CFG], predictor: DetectionPredictor, estimator: RuntimeEstimator):
        self.cfg = cfg
        self.predictor = predictor
        self.estimator = estimator

    def calibrate(
        self,
        weight_path: Path,
        yolo_dir: Path,
        imgsz: int,
        max_det: int,
        conf_values: Iterable[float],
        iou_values: Iterable[float],
        infer_mode: str,
        tile_size: int,
        tile_overlap: float,
        tile_nms_iou: float,
        fast: bool = CFG.FAST_CALIBRATION,
    ) -> tuple[float, float, float, pd.DataFrame, dict[str, int], dict[str, int], dict[str, float]]:
        from ultralytics import YOLO

        dirs = DatasetLayout.dirs(yolo_dir)
        val_imgs = sorted(dirs["val_img"].glob("*.png"))
        if not val_imgs:
            print("  [WARN] validation 이미지가 없어 conf/iou 보정을 생략합니다.")
            return self.cfg.CONF_THRESHOLD, self.cfg.IOU_THRESHOLD, tile_nms_iou, pd.DataFrame(), {}, {}, {}

        model = YOLO(str(weight_path))
        rows = []
        conf_values = list(conf_values)
        iou_values = list(iou_values)

        print("=" * 60)
        print("STEP 2-1 : validation count 기준 conf/iou 자동 보정")
        print("=" * 60)
        print(f"  val 이미지: {len(val_imgs)}개 | infer_mode={infer_mode} | conf 후보: {conf_values} | iou 후보: {iou_values}")
        if infer_mode in {"tile", "hybrid"}:
            print(f"  tile_size={tile_size} | tile_overlap={tile_overlap} | tile_nms_iou={tile_nms_iou}")

        gt_counts = {}
        imgs = {}
        for img_path in val_imgs:
            stem = img_path.stem
            label_path = dirs["val_lbl"] / f"{stem}.txt"
            gt_counts[stem] = YoloLabelConverter.count_yolo_labels(label_path)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                img = ImageIO.read_tif(self.cfg.TRAIN_IMG_DIR / f"{stem}.tif", fallback_size=self.cfg.IMG_SIZE)
            imgs[stem] = img

        cached_boxes = None
        if fast:
            cached_boxes = self._cache_validation_boxes(
                model=model,
                imgs=imgs,
                conf_values=conf_values,
                iou_values=iou_values,
                imgsz=imgsz,
                max_det=max_det,
                infer_mode=infer_mode,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )

        total = len(conf_values) * len(iou_values)
        done = 0
        started = time.perf_counter()

        for conf in conf_values:
            for iou in iou_values:
                errors = []
                signed_errors = []
                pred_counts = []
                true_counts = []

                for stem, img in imgs.items():
                    if cached_boxes is not None:
                        pred_n = self._count_cached_prediction(
                            cached_boxes=cached_boxes[stem],
                            conf=float(conf),
                            iou=float(iou),
                            max_det=max_det,
                            infer_mode=infer_mode,
                        )
                    else:
                        pred_n = self.predictor.predict_count(
                            model=model,
                            img=img,
                            conf=float(conf),
                            iou=float(iou),
                            imgsz=imgsz,
                            max_det=max_det,
                            infer_mode=infer_mode,
                            tile_size=tile_size,
                            tile_overlap=tile_overlap,
                            tile_nms_iou=tile_nms_iou,
                        )
                    true_n = gt_counts[stem]
                    err = pred_n - true_n
                    errors.append(abs(err))
                    signed_errors.append(err)
                    pred_counts.append(pred_n)
                    true_counts.append(true_n)

                mae = float(np.mean(errors))
                rmse = float(np.sqrt(np.mean(np.square(signed_errors))))
                bias = float(np.mean(signed_errors))
                rows.append({
                    "conf": float(conf),
                    "iou": float(iou),
                    "mae_count": mae,
                    "rmse_count": rmse,
                    "bias_pred_minus_true": bias,
                    "mean_pred_count": float(np.mean(pred_counts)),
                    "mean_true_count": float(np.mean(true_counts)),
                    "calibration_mode": "fast_cache" if cached_boxes is not None else "exact_predict",
                })
                done += 1
                print(f"  conf={conf:.2f}, iou={iou:.2f} → MAE={mae:.3f}, RMSE={rmse:.3f}, bias={bias:.3f}")
                self.estimator.progress("threshold 보정 grid", done, total, started)

        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["rmse_count", "mae_count", "bias_pred_minus_true"],
            key=lambda s: s.abs() if s.name == "bias_pred_minus_true" else s,
        )
        best = df.iloc[0]
        best_conf = float(best["conf"])
        best_scan_iou = float(best["iou"])
        best_iou = best_scan_iou
        best_tile_nms_iou = float(tile_nms_iou)
        # Fast tiled calibration scans the final tile-merge NMS IoU on cached high-IoU boxes.
        if cached_boxes is not None and infer_mode in {"tile", "hybrid"}:
            best_iou = float(max(iou_values)) if iou_values else best_scan_iou
            best_tile_nms_iou = best_scan_iou
        best_val_pred = self._validation_predictions_at_threshold(
            model=model,
            imgs=imgs,
            gt_counts=gt_counts,
            cached_boxes=cached_boxes,
            conf=best_conf,
            iou=best_scan_iou if cached_boxes is not None else best_iou,
            imgsz=imgsz,
            max_det=max_det,
            infer_mode=infer_mode,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            tile_nms_iou=best_tile_nms_iou,
        )
        val_stats = self._count_stats(best_val_pred, gt_counts)

        out_csv = weight_path.parent.parent / f"count_threshold_calibration_{infer_mode}.csv"
        df.to_csv(out_csv, index=False)
        val_detail_csv = weight_path.parent.parent / f"validation_count_detail_{infer_mode}.csv"
        pd.DataFrame([
            {
                "image": stem,
                "true_count": int(gt_counts[stem]),
                "pred_count": int(best_val_pred.get(stem, 0)),
                "error": int(best_val_pred.get(stem, 0)) - int(gt_counts[stem]),
                "conf": best_conf,
                "iou": best_iou,
                "calibration_iou": best_scan_iou,
                "tile_nms_iou": best_tile_nms_iou if infer_mode in {"tile", "hybrid"} else np.nan,
            }
            for stem in sorted(gt_counts)
        ]).to_csv(val_detail_csv, index=False)

        print("\n  [보정 결과]")
        print(f"    best conf: {best_conf:.2f}")
        print(f"    best iou : {best_iou:.2f}")
        if infer_mode in {"tile", "hybrid"}:
            print(f"    tile NMS : {best_tile_nms_iou:.2f}")
        print(f"    best MAE : {best['mae_count']:.3f}")
        print(f"    best RMSE: {val_stats.get('rmse', float('nan')):.3f}")
        print(f"    저장: {out_csv}\n")
        return best_conf, best_iou, best_tile_nms_iou, df, best_val_pred, gt_counts, val_stats

    def _validation_predictions_at_threshold(
        self,
        model,
        imgs: dict[str, np.ndarray],
        gt_counts: dict[str, int],
        cached_boxes: dict[str, dict[str, np.ndarray]] | None,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        infer_mode: str,
        tile_size: int,
        tile_overlap: float,
        tile_nms_iou: float,
    ) -> dict[str, int]:
        pred = {}
        for stem, img in imgs.items():
            if cached_boxes is not None:
                pred_n = self._count_cached_prediction(
                    cached_boxes=cached_boxes[stem],
                    conf=float(conf),
                    iou=float(iou),
                    max_det=max_det,
                    infer_mode=infer_mode,
                )
            else:
                pred_n = self.predictor.predict_count(
                    model=model,
                    img=img,
                    conf=float(conf),
                    iou=float(iou),
                    imgsz=imgsz,
                    max_det=max_det,
                    infer_mode=infer_mode,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    tile_nms_iou=tile_nms_iou,
                )
            if stem in gt_counts:
                pred[stem] = int(pred_n)
        return pred

    @staticmethod
    def _count_stats(pred: dict[str, int], true: dict[str, int]) -> dict[str, float]:
        keys = sorted(set(pred) & set(true))
        if not keys:
            return {}
        errors = np.array([float(pred[k]) - float(true[k]) for k in keys], dtype=float)
        return {
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "n": float(len(keys)),
        }

    def _cache_validation_boxes(
        self,
        model,
        imgs: dict[str, np.ndarray],
        conf_values: Sequence[float],
        iou_values: Sequence[float],
        imgsz: int,
        max_det: int,
        infer_mode: str,
        tile_size: int,
        tile_overlap: float,
    ) -> dict[str, dict[str, np.ndarray]]:
        min_conf = float(min(conf_values))
        cache_iou = float(max(iou_values))
        cached: dict[str, dict[str, np.ndarray]] = {}

        print("  [FAST CALIB] validation 이미지를 low-conf/high-iou로 1회 예측한 뒤 threshold grid는 캐시에서 계산합니다.")
        print(f"               cache_conf={min_conf:.2f} | cache_iou={cache_iou:.2f}")

        started = time.perf_counter()
        total = len(imgs)
        for idx, (stem, img) in enumerate(imgs.items(), start=1):
            entry: dict[str, np.ndarray] = {}
            if infer_mode in {"full", "hybrid"}:
                entry["full"] = self.predictor.predict_boxes_raw(
                    model=model,
                    img=img,
                    conf=min_conf,
                    iou=cache_iou,
                    imgsz=imgsz,
                    max_det=max_det,
                )
            if infer_mode in {"tile", "hybrid"}:
                entry["tile"] = self.predictor.predict_boxes_tiled(
                    model=model,
                    img=img,
                    conf=min_conf,
                    iou=cache_iou,
                    imgsz=imgsz,
                    max_det=max_det,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )
            cached[stem] = entry
            self.estimator.progress("보정 캐시 추론", idx, total, started)
        return cached

    @staticmethod
    def _count_cached_prediction(
        cached_boxes: dict[str, np.ndarray],
        conf: float,
        iou: float,
        max_det: int,
        infer_mode: str,
    ) -> int:
        if infer_mode == "full":
            return TileHelper.count_after_nms(cached_boxes["full"], nms_iou=iou, max_det=max_det, score_threshold=conf)
        if infer_mode == "tile":
            return TileHelper.count_after_nms(cached_boxes["tile"], nms_iou=iou, max_det=max_det, score_threshold=conf)
        if infer_mode == "hybrid":
            full_n = TileHelper.count_after_nms(cached_boxes["full"], nms_iou=iou, max_det=max_det, score_threshold=conf)
            tile_n = TileHelper.count_after_nms(cached_boxes["tile"], nms_iou=iou, max_det=max_det, score_threshold=conf)
            return int(round(0.35 * full_n + 0.65 * tile_n))
        raise ValueError("infer-mode는 full, tile, hybrid 중 하나여야 합니다.")


class TestInferencer:
    """테스트 이미지 TTA 추론 책임."""

    def __init__(self, cfg: type[CFG], predictor: DetectionPredictor, estimator: RuntimeEstimator):
        self.cfg = cfg
        self.predictor = predictor
        self.estimator = estimator

    def predict_with_tta(
        self,
        weight_path: Path,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        tta_level: str,
        infer_mode: str,
        tile_size: int,
        tile_overlap: float,
        tile_nms_iou: float,
    ) -> dict[str, int]:
        from ultralytics import YOLO

        print("=" * 60)
        print("STEP 3 : 테스트 이미지 추론")
        print(f"  weight={weight_path}")
        print(f"  conf={conf:.2f} | iou={iou:.2f} | imgsz={imgsz} | max_det={max_det} | TTA={tta_level} | infer_mode={infer_mode}")
        if infer_mode in {"tile", "hybrid"}:
            print(f"  tile_size={tile_size} | tile_overlap={tile_overlap} | tile_nms_iou={tile_nms_iou}")
        print("=" * 60)

        model = YOLO(str(weight_path))
        tif_list = sorted(self.cfg.TEST_IMG_DIR.glob("*.tif"))
        if not tif_list:
            raise FileNotFoundError(
                f"테스트 이미지가 없습니다: {self.cfg.TEST_IMG_DIR}\n"
                "data/test_images/ 폴더에 *.tif 파일을 넣어주세요."
            )

        results_dict: dict[str, int] = {}
        detail_rows = []
        started = time.perf_counter()

        for idx, tif_path in enumerate(tqdm(tif_list, desc="  추론"), start=1):
            stem = tif_path.stem
            img = ImageIO.read_tif(tif_path, fallback_size=self.cfg.IMG_SIZE)
            tta_imgs = TTAFactory.make(img, tta_level)

            counts = []
            for aug_name, aug_img in tta_imgs:
                n = self.predictor.predict_count(
                    model=model,
                    img=aug_img,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    max_det=max_det,
                    infer_mode=infer_mode,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    tile_nms_iou=tile_nms_iou,
                )
                counts.append(n)
                detail_rows.append({
                    "target_image": stem,
                    "augmentation": aug_name,
                    "count": n,
                    "conf": conf,
                    "iou": iou,
                    "weight": str(weight_path),
                    "infer_mode": infer_mode,
                    "tile_size": tile_size if infer_mode in {"tile", "hybrid"} else 0,
                    "tile_nms_iou": tile_nms_iou if infer_mode in {"tile", "hybrid"} else np.nan,
                })

            final_count = int(round(float(np.median(counts))))
            results_dict[stem] = final_count
            tqdm.write(f"    {stem:<20} → TTA counts={counts} | median={final_count}")
            self.estimator.progress("테스트 이미지 추론", idx, len(tif_list), started)

        detail_csv = weight_path.parent.parent / f"test_tta_detail_{infer_mode}_{tta_level}.csv"
        pd.DataFrame(detail_rows).to_csv(detail_csv, index=False)
        print(f"\n  모델별 TTA 상세 저장: {detail_csv}")
        print(f"  모델별 추론 완료. 총 {sum(results_dict.values())}개 건물 탐지\n")
        return results_dict


class PredictionAggregator:
    """여러 모델/seed 예측값 앙상블 책임."""

    @classmethod
    def aggregate(
        cls,
        pred_dicts: list[dict[str, int]],
        method: str = "auto",
        detail_path: Path | None = None,
        validation_records: list[dict] | None = None,
    ) -> dict[str, int]:
        if not pred_dicts:
            raise ValueError("앙상블할 예측 결과가 없습니다.")
        if len(pred_dicts) == 1:
            return pred_dicts[0]

        keys = sorted(set().union(*[set(d.keys()) for d in pred_dicts]))
        selected_method = method
        weights = None
        biases = None

        if method == "auto":
            selected_method, weights, biases = cls._choose_by_validation(validation_records, model_count=len(pred_dicts))
        elif method == "val_weighted":
            weights, biases = cls._weights_from_validation(validation_records, model_count=len(pred_dicts))
            selected_method = "val_weighted" if weights is not None else "median"
        elif method == "bias_weighted":
            weights, biases = cls._weights_from_validation(validation_records, use_bias=True, model_count=len(pred_dicts))
            selected_method = "bias_weighted" if weights is not None and biases is not None else "median"

        final = {}
        rows = []

        for key in keys:
            vals = [d[key] for d in pred_dicts if key in d]
            if selected_method == "mean":
                agg = int(round(float(np.mean(vals))))
            elif selected_method == "median":
                agg = int(round(float(np.median(vals))))
            elif selected_method == "val_weighted":
                agg = cls._weighted_value(vals, weights=weights, biases=None)
            elif selected_method == "bias_weighted":
                agg = cls._weighted_value(vals, weights=weights, biases=biases)
            elif selected_method.startswith("single_"):
                idx = int(selected_method.split("_", 1)[1])
                agg = int(round(float(pred_dicts[idx].get(key, 0))))
            else:
                raise ValueError("ensemble-method는 auto, median, mean, val_weighted, bias_weighted 중 하나여야 합니다.")
            final[key] = agg
            rows.append({
                "target_image": key,
                "model_counts": vals,
                "ensemble_count": agg,
                "requested_method": method,
                "selected_method": selected_method,
                "weights": weights,
                "biases": biases,
            })

        out_csv = Path(detail_path) if detail_path is not None else Path("ensemble_prediction_detail.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("=" * 60)
        print("STEP 3-1 : 모델 앙상블")
        print("=" * 60)
        print(f"  모델 수: {len(pred_dicts)} | 요청 방법: {method} | 선택 방법: {selected_method}")
        print(f"  상세 저장: {out_csv}\n")
        return final

    @classmethod
    def _choose_by_validation(
        cls,
        validation_records: list[dict] | None,
        model_count: int,
    ) -> tuple[str, list[float] | None, list[float] | None]:
        valid = cls._valid_validation_records(validation_records, model_count=model_count)
        if len(valid) != model_count or len(valid) < 2:
            print("[ENSEMBLE] 공통 validation 예측이 부족하여 median 앙상블을 사용합니다.")
            return "median", None, None

        true = cls._common_true(valid)
        if len(true) < 3:
            print("[ENSEMBLE] 공통 validation 이미지가 너무 적어 median 앙상블을 사용합니다.")
            return "median", None, None

        candidates: list[tuple[str, float, list[float] | None, list[float] | None]] = []
        for idx, rec in enumerate(valid):
            rmse = cls._rmse(rec["val_pred"], true)
            pred_index = int(rec.get("pred_index", idx))
            candidates.append((f"single_{pred_index}", rmse, None, None))

        for name in ("mean", "median"):
            val_pred = cls._aggregate_validation(valid, true, method=name)
            candidates.append((name, cls._rmse(val_pred, true), None, None))

        weights, _ = cls._weights_from_validation(valid, model_count=model_count)
        val_pred = cls._aggregate_validation(valid, true, method="val_weighted", weights=weights)
        candidates.append(("val_weighted", cls._rmse(val_pred, true), weights, None))

        bias_weights, biases = cls._weights_from_validation(valid, use_bias=True, model_count=model_count)
        val_pred = cls._aggregate_validation(valid, true, method="bias_weighted", weights=bias_weights, biases=biases)
        candidates.append(("bias_weighted", cls._rmse(val_pred, true), bias_weights, biases))

        candidates = sorted(candidates, key=lambda x: x[1])
        print("=" * 60)
        print("STEP 3-0 : validation 기준 최종 제출 방식 자동 선택")
        print("=" * 60)
        for name, rmse, cand_weights, cand_biases in candidates:
            extra = ""
            if cand_weights is not None:
                extra += f" | weights={np.round(cand_weights, 3).tolist()}"
            if cand_biases is not None:
                extra += f" | biases={np.round(cand_biases, 1).tolist()}"
            print(f"  {name:<14} val RMSE={rmse:.3f}{extra}")
        print(f"  선택: {candidates[0][0]}\n")
        return candidates[0][0], candidates[0][2], candidates[0][3]

    @staticmethod
    def _valid_validation_records(validation_records: list[dict] | None, model_count: int | None = None) -> list[dict]:
        if not validation_records:
            return []
        valid = []
        for rec in validation_records:
            if rec and rec.get("val_pred") and rec.get("val_true"):
                valid.append(rec)
        if model_count is not None:
            valid = [rec for rec in valid if 0 <= int(rec.get("pred_index", -1)) < model_count]
            valid = sorted(valid, key=lambda rec: int(rec.get("pred_index", 0)))
            indices = [int(rec.get("pred_index", -1)) for rec in valid]
            if indices != list(range(model_count)):
                return []
        return valid

    @classmethod
    def _common_true(cls, records: list[dict]) -> dict[str, int]:
        keys = set(records[0]["val_true"])
        for rec in records:
            keys &= set(rec["val_true"]) & set(rec["val_pred"])
        return {k: int(records[0]["val_true"][k]) for k in sorted(keys)}

    @staticmethod
    def _rmse(pred: dict[str, int], true: dict[str, int]) -> float:
        keys = sorted(set(pred) & set(true))
        if not keys:
            return float("inf")
        errors = np.array([float(pred[k]) - float(true[k]) for k in keys], dtype=float)
        return float(np.sqrt(np.mean(np.square(errors))))

    @classmethod
    def _weights_from_validation(
        cls,
        validation_records: list[dict] | None,
        use_bias: bool = False,
        model_count: int | None = None,
    ) -> tuple[list[float] | None, list[float] | None]:
        valid = cls._valid_validation_records(validation_records, model_count=model_count)
        if len(valid) < 2:
            return None, None
        true = cls._common_true(valid)
        if len(true) >= 3:
            rmses = np.array([max(cls._rmse(rec["val_pred"], true), 1.0) for rec in valid], dtype=float)
        else:
            rmses = np.array([max(float(rec.get("rmse", float("inf"))), 1.0) for rec in valid], dtype=float)
        if not np.isfinite(rmses).all():
            rmses = np.ones(len(valid), dtype=float)
        inv = 1.0 / np.square(rmses)
        weights = (inv / inv.sum()).astype(float).tolist() if inv.sum() > 0 else None
        biases = [float(rec.get("bias", 0.0)) for rec in valid] if use_bias else None
        return weights, biases

    @classmethod
    def _aggregate_validation(
        cls,
        records: list[dict],
        true: dict[str, int],
        method: str,
        weights: list[float] | None = None,
        biases: list[float] | None = None,
    ) -> dict[str, int]:
        pred = {}
        for key in true:
            vals = [int(rec["val_pred"][key]) for rec in records]
            if method == "mean":
                agg = int(round(float(np.mean(vals))))
            elif method == "median":
                agg = int(round(float(np.median(vals))))
            elif method == "val_weighted":
                agg = cls._weighted_value(vals, weights=weights, biases=None)
            elif method == "bias_weighted":
                agg = cls._weighted_value(vals, weights=weights, biases=biases)
            else:
                raise ValueError(method)
            pred[key] = agg
        return pred

    @staticmethod
    def _weighted_value(vals: Sequence[int], weights: list[float] | None, biases: list[float] | None = None) -> int:
        arr = np.array(vals, dtype=float)
        if biases is not None:
            arr = arr - np.array(biases[: len(arr)], dtype=float)
        if weights is None:
            value = float(np.mean(arr))
        else:
            w = np.array(weights[: len(arr)], dtype=float)
            value = float(np.sum(arr * w) / max(float(np.sum(w)), 1e-9))
        return max(0, int(round(value)))


class SubmissionWriter:
    """제출 CSV 생성 책임."""

    def __init__(self, cfg: type[CFG]):
        self.cfg = cfg

    @staticmethod
    def _lookup_count(target_image, pred_dict: dict[str, int]) -> int:
        key = str(target_image)
        if key in pred_dict:
            return int(pred_dict[key])
        stem = Path(key).stem
        return int(pred_dict.get(stem, 0))

    def make_submission(self, pred_dict: dict[str, int], output_path: Path = CFG.SUBMISSION_CSV) -> pd.DataFrame:
        print("=" * 60)
        print("STEP 4 : 제출 파일 생성")
        print("=" * 60)

        if self.cfg.TEMPLATE_CSV.exists():
            df = pd.read_csv(self.cfg.TEMPLATE_CSV)
            print(f"  템플릿 로드: {self.cfg.TEMPLATE_CSV} ({len(df)}행)")
        else:
            print("  [WARN] 템플릿 없음 → pred_dict로 직접 생성")
            df = pd.DataFrame({"target_image": sorted(pred_dict.keys()), "building_count": 0})

        df["building_count"] = df["target_image"].apply(lambda x: self._lookup_count(x, pred_dict)).astype(int)
        df = df.sort_values("target_image").reset_index(drop=True)
        df.to_csv(output_path, index=False)

        print(f"\n  저장 완료: {output_path}")
        print(f"\n{'─' * 35}")
        print(df.to_string(index=False))
        print(f"{'─' * 35}")
        print(f"\n  평균 건물 수: {df['building_count'].mean():.1f}")
        print(f"  최솟값: {df['building_count'].min()} | 최댓값: {df['building_count'].max()}")
        return df


class DataInspector:
    """데이터 구조/라벨 분포 검사 책임."""

    def __init__(self, cfg: type[CFG]):
        self.cfg = cfg

    def inspect(self) -> None:
        print("=" * 60)
        print("데이터 구조 검사")
        print("=" * 60)

        train_imgs = sorted(self.cfg.TRAIN_IMG_DIR.glob("*.tif"))
        test_imgs = sorted(self.cfg.TEST_IMG_DIR.glob("*.tif"))
        geojsons = sorted(self.cfg.TRAIN_LBL_DIR.glob("*.geojson"))
        print(f"  학습 이미지: {len(train_imgs)}개")
        print(f"  테스트 이미지: {len(test_imgs)}개")
        print(f"  GeoJSON 레이블: {len(geojsons)}개\n")
        if train_imgs:
            matched = sum(1 for t in train_imgs if LabelPathResolver.find(self.cfg.TRAIN_LBL_DIR, t.stem) is not None)
            print(f"  이미지-라벨 매칭: {matched}/{len(train_imgs)}개")
            if matched != len(train_imgs):
                missing = [t.name for t in train_imgs if LabelPathResolver.find(self.cfg.TRAIN_LBL_DIR, t.stem) is None][:10]
                print(f"  [WARN] 라벨 매칭 실패 예시: {missing}")
            print()

        if train_imgs:
            sample_tif = train_imgs[0]
            print(f"  [TIF 샘플] {sample_tif.name}")
            try:
                with rasterio.open(sample_tif) as src:
                    print(f"    크기: {src.width} × {src.height}")
                    print(f"    밴드 수: {src.count}")
                    print(f"    데이터 타입: {src.dtypes}")
                    print(f"    CRS: {src.crs}")
                    print(f"    색상 해석: {src.colorinterp}")
            except Exception as e:
                print(f"    [ERROR] {e}")
            print()

        if geojsons:
            counts = []
            for g in tqdm(geojsons, desc="  GeoJSON count 검사"):
                counts.append(len(GeoJsonParser.load(g)))
            print("  [전체 GeoJSON 건물 수 분포]")
            print(f"    평균: {np.mean(counts):.1f} | 중앙값: {np.median(counts):.1f}")
            print(f"    최소: {min(counts)} | 최대: {max(counts)}")
            print(f"    상위 5개: {sorted(counts, reverse=True)[:5]}")


# ──────────────────────────────────────────────────────────────
# 10. 파서/유틸
# ──────────────────────────────────────────────────────────────
class ArgParserFactory:
    """CLI 파서 생성 책임."""

    @staticmethod
    def create() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="문제2 건물 탐지 SOLID 리팩토링 + ETA 출력 파이프라인")
        p.add_argument(
            "--preset",
            choices=[
                "manual",
                # 클라우드 H100 단일 GPU - 5시간 예산
                "h100-cloud-5h", "h100-cloud-5h-safe",
                # 기존 단일 GPU preset (호환)
                "rtx6000-150m", "rtx6000-small-3h", "rtx6000-small-3h-hires",
                "rtx6000-small-3h-stable", "rtx6000-180x2", "rtx6000-200x2",
            ],
            default="h100-cloud-5h",
            help="실행 환경/시간 예산 preset. 기본 h100-cloud-5h는 H100 80GB 단일 GPU 클라우드 환경 안전 설정.",
        )
        p.add_argument("--inspect", action="store_true", help="데이터 구조 확인 후 종료")
        p.add_argument("--data-dir", type=str, default=None, help="문제2 데이터 폴더(train_images/train_labels/test_images 포함). 생략 시 자동 탐색")
        p.add_argument("--work-dir", type=str, default=None, help="기존 seed_* 데이터셋과 학습 run 폴더가 있는 작업 폴더")
        p.add_argument("--skip-convert", action="store_true", help="YOLO 데이터셋 변환 생략")
        p.add_argument("--skip-train", action="store_true", help="학습 생략, 기존 가중치 사용")
        p.add_argument("--resume", action="store_true", help="기존 dataset.yaml/학습 run이 있으면 삭제하지 않고 이어서 실행")
        p.add_argument("--start-after-seed", type=int, default=None, help="지정 seed 이후의 seed만 새로 학습. 예: 42")
        p.add_argument("--reuse-short-runs", action="store_true", help="기존 run이 목표 epoch보다 짧아도 재사용. 빠르지만 정확한 180x2는 아님")
        p.add_argument("--weight", type=str, default=None, help="단일 가중치 경로")
        p.add_argument("--weights", type=str, default=None, help="앙상블 가중치 경로들. 예: a.pt,b.pt,c.pt")

        p.add_argument("--model", type=str, default=CFG.MODEL, help=f"사전학습 모델 (기본: {CFG.MODEL})")
        p.add_argument("--models", type=str, default=None, help="여러 모델 앙상블. 예: yolo11x.pt,yolov8x.pt")
        p.add_argument("--imgsz", type=int, default=CFG.IMG_SIZE, help=f"학습/추론 이미지 크기 (기본: {CFG.IMG_SIZE})")
        p.add_argument("--epochs", type=int, default=CFG.EPOCHS, help=f"최대 학습 epoch (기본: {CFG.EPOCHS})")
        p.add_argument("--patience", type=int, default=CFG.PATIENCE, help=f"early stopping patience (기본: {CFG.PATIENCE})")
        p.add_argument("--batch", type=int, default=CFG.BATCH, help=f"배치 크기 (기본: {CFG.BATCH})")
        p.add_argument("--seeds", type=str, default=CFG.ENSEMBLE_SEEDS, help=f"앙상블 seed 목록 (기본: {CFG.ENSEMBLE_SEEDS})")
        p.add_argument("--val-ratio", type=float, default=CFG.VAL_RATIO, help=f"validation 비율 (기본: {CFG.VAL_RATIO})")
        p.add_argument("--min-box-px", type=float, default=CFG.MIN_BOX_PX, help=f"작은 건물 학습 라벨 최소 bbox 픽셀 크기 (기본: {CFG.MIN_BOX_PX})")
        p.add_argument("--split-mode", choices=["random", "stratified", "group-stratified"], default=CFG.SPLIT_MODE, help="train/val 분할 방식")
        p.add_argument("--split-seed", type=int, default=42, help="seed가 달라도 같은 validation split을 쓰기 위한 분할 seed")
        p.add_argument("--vary-split-by-seed", action="store_true", help="seed마다 train/val split도 다르게 생성")
        p.add_argument("--workers", type=int, default=CFG.WORKERS, help=f"DataLoader workers (기본: {CFG.WORKERS})")
        p.add_argument("--cache-mode", choices=["ram", "disk", "none"], default="ram", help="Ultralytics 데이터 캐시 방식")
        p.add_argument("--deterministic", action="store_true", help="재현성 우선 학습 사용. 속도는 다소 느려질 수 있음")
        p.add_argument("--aug-degrees", type=float, default=CFG.AUG_DEGREES, help=f"회전 증강 각도 범위 (기본: {CFG.AUG_DEGREES}). 0 권장 (axis-aligned bbox 보호)")
        p.add_argument("--aug-mosaic", type=float, default=CFG.AUG_MOSAIC, help=f"mosaic 증강 확률 (기본: {CFG.AUG_MOSAIC})")
        p.add_argument("--aug-scale", type=float, default=CFG.AUG_SCALE, help=f"scale 증강 범위 (기본: {CFG.AUG_SCALE})")
        p.add_argument("--aug-translate", type=float, default=CFG.AUG_TRANSLATE, help=f"translate 증강 범위 (기본: {CFG.AUG_TRANSLATE})")
        p.add_argument("--aug-mixup", type=float, default=CFG.AUG_MIXUP, help=f"mixup 증강 확률 (기본: {CFG.AUG_MIXUP})")
        p.add_argument("--aug-copy-paste", type=float, default=CFG.AUG_COPY_PASTE, help=f"copy_paste 증강 확률 (기본: {CFG.AUG_COPY_PASTE})")
        p.add_argument("--close-mosaic", type=int, default=CFG.CLOSE_MOSAIC, help=f"마지막 N epoch에서 mosaic 비활성화 (기본: {CFG.CLOSE_MOSAIC})")
        p.add_argument("--no-plots", action="store_true", help="학습 중 matplotlib plots 생성 비활성화 (메모리/시간 절약)")
        p.add_argument("--cuda-alloc-conf", type=str, default=None, help="PYTORCH_CUDA_ALLOC_CONF 값 덮어쓰기")

        p.add_argument("--device", type=str, default=CFG.DEVICE, help="실행 장치: auto, cuda, cpu, 0, 0,1 등")
        p.add_argument("--force-cuda", action="store_true", default=CFG.FORCE_CUDA, help="CUDA가 없으면 즉시 중단")
        p.add_argument("--no-amp", action="store_true", help="CUDA AMP 비활성화")
        p.add_argument("--no-half", action="store_true", help="CUDA half precision 추론 비활성화")

        p.add_argument("--conf", type=float, default=CFG.CONF_THRESHOLD, help=f"기본 conf (보정 생략 시 사용, 기본: {CFG.CONF_THRESHOLD})")
        p.add_argument("--iou", type=float, default=CFG.IOU_THRESHOLD, help=f"기본 iou (보정 생략 시 사용, 기본: {CFG.IOU_THRESHOLD})")
        p.add_argument("--no-calibrate", action="store_true", help="validation count 기준 conf/iou 자동 보정 비활성화")
        p.add_argument("--slow-calibrate", action="store_true", help="빠른 캐시 보정 대신 기존처럼 grid마다 YOLO 추론을 다시 수행")
        p.add_argument("--tta-level", choices=["none", "light", "strong"], default="strong", help="테스트 TTA 강도")
        p.add_argument("--infer-mode", choices=["full", "tile", "hybrid"], default=CFG.INFER_MODE, help="추론 방식")
        p.add_argument("--tile-size", type=int, default=CFG.TILE_SIZE, help=f"tile 추론 크기 (기본: {CFG.TILE_SIZE})")
        p.add_argument("--tile-overlap", type=float, default=CFG.TILE_OVERLAP, help=f"tile overlap 비율 (기본: {CFG.TILE_OVERLAP})")
        p.add_argument("--tile-nms-iou", type=float, default=CFG.TILE_NMS_IOU, help=f"tile 병합 NMS IoU (기본: {CFG.TILE_NMS_IOU})")
        p.add_argument("--max-det", type=int, default=CFG.MAX_DET, help=f"이미지당 최대 탐지 박스 수 (기본: {CFG.MAX_DET})")
        p.add_argument(
            "--ensemble-method",
            choices=["auto", "median", "mean", "val_weighted", "bias_weighted"],
            default="auto",
            help="여러 모델 count 앙상블 방식. auto는 validation RMSE가 가장 낮은 단일/앙상블 방식을 하나만 선택",
        )
        p.add_argument("--submission", type=str, default=str(CFG.SUBMISSION_CSV), help="제출 CSV 저장 경로")
        p.add_argument("--time-report-every", type=int, default=10, help="ETA 중간 출력 주기. 기본 10단위마다 출력")
        p.add_argument("--time-budget-min", type=float, default=CFG.TIME_BUDGET_MIN, help="전체 실행 시간 예산(분). 0이면 제한 없음")
        p.add_argument("--reserve-submit-min", type=float, default=CFG.RESERVE_SUBMIT_MIN, help="마지막 보정/추론/제출을 위해 남길 시간(분)")
        return p


class ParseUtils:
    """문자열 파싱 유틸리티."""

    @staticmethod
    def parse_seed_list(seed_text: str) -> list[int]:
        seeds = []
        for part in str(seed_text).split(","):
            part = part.strip()
            if part:
                seeds.append(int(part))
        if not seeds:
            raise ValueError("seed가 비어 있습니다.")
        return seeds

    @staticmethod
    def filter_seeds_after(seeds: Sequence[int], start_after_seed: int | None) -> list[int]:
        seeds = list(seeds)
        if start_after_seed is None:
            return seeds
        if start_after_seed not in seeds:
            filtered = [s for s in seeds if s > start_after_seed]
        else:
            start_idx = seeds.index(start_after_seed) + 1
            filtered = seeds[start_idx:]
        if not filtered:
            raise ValueError(f"--start-after-seed {start_after_seed} 이후에 실행할 seed가 없습니다. 현재 seed 목록: {seeds}")
        print(f"[RESUME] start-after-seed={start_after_seed} 적용: {seeds} → {filtered}")
        return filtered

    @staticmethod
    def parse_weight_list(weight_text: str | None, fallback_weight: str | None = None) -> list[Path]:
        raw = weight_text or fallback_weight
        if not raw:
            return []
        return [Path(p.strip()) for p in raw.split(",") if p.strip()]

    @staticmethod
    def parse_model_list(models_text: str | None, fallback_model: str) -> list[str]:
        raw = models_text or fallback_model
        models = [m.strip() for m in str(raw).split(",") if m.strip()]
        if not models:
            raise ValueError("모델 목록이 비어 있습니다.")
        return models

    @staticmethod
    def safe_model_stem(model_name: str) -> str:
        return Path(model_name).stem.replace(".", "_").replace("-", "_")


class PresetManager:
    """GPU/시간 예산별 권장 실행값을 적용한다."""

    PRESETS = {
        # ============================================================
        # H100 80GB 단일 GPU (클라우드) - 5시간 예산, 기본 권장
        # ============================================================
        #
        # VRAM 예상 (yolo11x BF16/AMP, 학습+val peak):
        #   imgsz=1024 batch=12 → 약 45~55GB  ← h100-cloud-5h (기본)
        #   imgsz=1024 batch=8  → 약 30~40GB  ← h100-cloud-5h-safe
        #
        # 시간 예상 (H100 80GB 단일):
        #   yolo11x imgsz=1024 batch=12 200ep ≈ 70~90분/job
        #   3 seed × 80분 = 240분 + 보정/추론 = 약 280분 < 300분(5h) ✓
        #
        # 클라우드 안전 옵션:
        #   - cache='disk' (RAM 부담 ↓)
        #   - workers=4 (도커 컨테이너 DataLoader 안정)
        #   - no_plots=True (matplotlib 메모리 회피)
        #   - device='cuda' (자동으로 GPU 0 선택)
        "h100-cloud-5h": {
            "model": "yolo11x.pt",
            "models": None,
            "seeds": "42,77,123",
            "imgsz": 1024,
            "epochs": 200,
            "patience": 60,
            "batch": 12,                  # H100 80GB peak ~50GB
            "workers": 4,
            "cache_mode": "disk",
            "deterministic": False,
            "val_ratio": 0.15,
            "min_box_px": 5.0,
            "split_mode": "group-stratified",
            "split_seed": 42,
            "vary_split_by_seed": False,
            "device": "cuda",
            "force_cuda": True,
            "resume": True,
            "start_after_seed": None,
            "reuse_short_runs": False,
            "conf": 0.25,
            "iou": 0.50,
            "no_calibrate": False,
            "slow_calibrate": False,
            "tta_level": "strong",
            "infer_mode": "tile",
            "tile_size": 768,
            "tile_overlap": 0.20,
            "tile_nms_iou": 0.50,
            "max_det": 15000,
            "ensemble_method": "auto",
            "time_report_every": 5,
            "time_budget_min": 300.0,
            "reserve_submit_min": 25.0,
            "aug_degrees": 0.0,
            "aug_mosaic": 0.55,
            "aug_scale": 0.40,
            "aug_translate": 0.08,
            "aug_mixup": 0.05,
            "aug_copy_paste": 0.10,
            "close_mosaic": 35,
            "no_plots": True,
        },
        # OOM 위험 0에 가까운 보수 설정 (다른 작업과 GPU 공유 / 첫 OOM 폴백)
        "h100-cloud-5h-safe": {
            "model": "yolo11x.pt",
            "models": None,
            "seeds": "42,77,123",
            "imgsz": 1024,
            "epochs": 220,
            "patience": 65,
            "batch": 8,                   # peak ~35GB
            "workers": 4,
            "cache_mode": "disk",
            "deterministic": False,
            "val_ratio": 0.15,
            "min_box_px": 5.0,
            "split_mode": "group-stratified",
            "split_seed": 42,
            "vary_split_by_seed": False,
            "device": "cuda",
            "force_cuda": True,
            "resume": True,
            "start_after_seed": None,
            "reuse_short_runs": False,
            "conf": 0.25,
            "iou": 0.50,
            "no_calibrate": False,
            "slow_calibrate": False,
            "tta_level": "strong",
            "infer_mode": "tile",
            "tile_size": 768,
            "tile_overlap": 0.20,
            "tile_nms_iou": 0.50,
            "max_det": 15000,
            "ensemble_method": "auto",
            "time_report_every": 5,
            "time_budget_min": 300.0,
            "reserve_submit_min": 25.0,
            "aug_degrees": 0.0,
            "aug_mosaic": 0.50,
            "aug_scale": 0.40,
            "aug_translate": 0.08,
            "aug_mixup": 0.05,
            "aug_copy_paste": 0.10,
            "close_mosaic": 35,
            "no_plots": True,
        },
        # ============================================================
        # 기존 RTX PRO 6000 preset (호환용)
        # ============================================================
        "rtx6000-150m": {
            "models": "yolo11x.pt,yolov8x.pt",
            "seeds": "42",
            "imgsz": 1280,
            "epochs": 360,
            "patience": 80,
            "batch": 16,
            "workers": 12,
            "cache_mode": "ram",
            "deterministic": False,
            "val_ratio": 0.15,
            "min_box_px": 5.0,
            "split_mode": "group-stratified",
            "split_seed": 42,
            "vary_split_by_seed": False,
            "device": "cuda",
            "force_cuda": True,
            "conf": 0.25,
            "iou": 0.50,
            "no_calibrate": False,
            "slow_calibrate": False,
            "tta_level": "strong",
            "infer_mode": "full",
            "tile_size": 1024,
            "tile_overlap": 0.10,
            "tile_nms_iou": 0.45,
            "max_det": 30000,
            "ensemble_method": "auto",
            "time_report_every": 5,
            "time_budget_min": 150.0,
            "reserve_submit_min": 18.0,
        },
        "rtx6000-small-3h": {
            "model": "yolo11x.pt",
            "models": None,
            "seeds": "42,77",
            "imgsz": 1024,
            "epochs": 140,
            "patience": 35,
            "batch": 16,
            "workers": 12,
            "cache_mode": "ram",
            "deterministic": False,
            "val_ratio": 0.15,
            "min_box_px": 5.0,
            "split_mode": "group-stratified",
            "split_seed": 42,
            "vary_split_by_seed": False,
            "device": "cuda",
            "force_cuda": True,
            "resume": True,
            "start_after_seed": None,
            "reuse_short_runs": False,
            "conf": 0.25,
            "iou": 0.50,
            "no_calibrate": False,
            "slow_calibrate": False,
            "tta_level": "strong",
            "infer_mode": "tile",
            "tile_size": 768,
            "tile_overlap": 0.20,
            "tile_nms_iou": 0.50,
            "max_det": 30000,
            "ensemble_method": "auto",
            "time_report_every": 5,
            "time_budget_min": 180.0,
            "reserve_submit_min": 12.0,
        },
        "rtx6000-small-3h-hires": {
            "model": "yolo11x.pt",
            "models": None,
            "seeds": "42,77",
            "imgsz": 1280,
            "epochs": 100,
            "patience": 30,
            "batch": 16,
            "workers": 12,
            "cache_mode": "ram",
            "deterministic": False,
            "val_ratio": 0.15,
            "min_box_px": 5.0,
            "split_mode": "group-stratified",
            "split_seed": 42,
            "vary_split_by_seed": False,
            "device": "cuda",
            "force_cuda": True,
            "resume": True,
            "start_after_seed": None,
            "reuse_short_runs": False,
            "conf": 0.25,
            "iou": 0.50,
            "no_calibrate": False,
            "slow_calibrate": False,
            "tta_level": "strong",
            "infer_mode": "tile",
            "tile_size": 768,
            "tile_overlap": 0.20,
            "tile_nms_iou": 0.50,
            "max_det": 30000,
            "ensemble_method": "auto",
            "time_report_every": 5,
            "time_budget_min": 180.0,
            "reserve_submit_min": 12.0,
        },
        "rtx6000-small-3h-stable": {
            "model": "yolo11x.pt",
            "models": None,
            "seeds": "42,77",
            "imgsz": 1024,
            "epochs": 130,
            "patience": 30,
            "batch": 8,
            "workers": 4,
            "cache_mode": "disk",
            "deterministic": False,
            "val_ratio": 0.15,
            "min_box_px": 5.0,
            "split_mode": "group-stratified",
            "split_seed": 42,
            "vary_split_by_seed": False,
            "device": "cuda",
            "force_cuda": True,
            "resume": True,
            "start_after_seed": None,
            "reuse_short_runs": False,
            "conf": 0.25,
            "iou": 0.50,
            "no_calibrate": False,
            "slow_calibrate": False,
            "tta_level": "strong",
            "infer_mode": "tile",
            "tile_size": 768,
            "tile_overlap": 0.20,
            "tile_nms_iou": 0.50,
            "max_det": 22000,
            "ensemble_method": "auto",
            "time_report_every": 5,
            "time_budget_min": 180.0,
            "reserve_submit_min": 14.0,
            "aug_degrees": 45.0,
            "aug_mosaic": 0.65,
            "aug_scale": 0.35,
            "aug_translate": 0.08,
            "close_mosaic": 25,
        },
        "rtx6000-180x2": {
            "model": "yolo11x.pt",
            "models": None,
            "seeds": "42,77",
            "imgsz": 1024,
            "epochs": 180,
            "patience": 40,
            "batch": 16,
            "workers": 12,
            "cache_mode": "ram",
            "deterministic": False,
            "val_ratio": 0.15,
            "min_box_px": 5.0,
            "split_mode": "group-stratified",
            "split_seed": 42,
            "vary_split_by_seed": False,
            "device": "cuda",
            "force_cuda": True,
            "resume": True,
            "start_after_seed": None,
            "reuse_short_runs": False,
            "conf": 0.25,
            "iou": 0.50,
            "no_calibrate": False,
            "slow_calibrate": False,
            "tta_level": "strong",
            "infer_mode": "tile",
            "tile_size": 768,
            "tile_overlap": 0.20,
            "tile_nms_iou": 0.50,
            "max_det": 30000,
            "ensemble_method": "auto",
            "time_report_every": 5,
            "time_budget_min": 0.0,
            "reserve_submit_min": 12.0,
        },
        "rtx6000-200x2": {
            "model": "yolo11x.pt",
            "models": None,
            "seeds": "42,77",
            "imgsz": 1024,
            "epochs": 200,
            "patience": 45,
            "batch": 16,
            "workers": 12,
            "cache_mode": "ram",
            "deterministic": False,
            "val_ratio": 0.15,
            "min_box_px": 5.0,
            "split_mode": "group-stratified",
            "split_seed": 42,
            "vary_split_by_seed": False,
            "device": "cuda",
            "force_cuda": True,
            "resume": True,
            "start_after_seed": None,
            "reuse_short_runs": False,
            "conf": 0.25,
            "iou": 0.50,
            "no_calibrate": False,
            "slow_calibrate": False,
            "tta_level": "strong",
            "infer_mode": "tile",
            "tile_size": 768,
            "tile_overlap": 0.20,
            "tile_nms_iou": 0.50,
            "max_det": 30000,
            "ensemble_method": "auto",
            "time_report_every": 5,
            "time_budget_min": 0.0,
            "reserve_submit_min": 12.0,
        }
    }

    OPTION_TO_ATTR = {
        "--models": "models",
        "--data-dir": "data_dir",
        "--work-dir": "work_dir",
        "--resume": "resume",
        "--start-after-seed": "start_after_seed",
        "--reuse-short-runs": "reuse_short_runs",
        "--model": "model",
        "--seeds": "seeds",
        "--imgsz": "imgsz",
        "--epochs": "epochs",
        "--patience": "patience",
        "--batch": "batch",
        "--workers": "workers",
        "--cache-mode": "cache_mode",
        "--deterministic": "deterministic",
        "--aug-degrees": "aug_degrees",
        "--aug-mosaic": "aug_mosaic",
        "--aug-scale": "aug_scale",
        "--aug-translate": "aug_translate",
        "--aug-mixup": "aug_mixup",
        "--aug-copy-paste": "aug_copy_paste",
        "--close-mosaic": "close_mosaic",
        "--no-plots": "no_plots",
        "--val-ratio": "val_ratio",
        "--min-box-px": "min_box_px",
        "--split-mode": "split_mode",
        "--split-seed": "split_seed",
        "--vary-split-by-seed": "vary_split_by_seed",
        "--device": "device",
        "--force-cuda": "force_cuda",
        "--conf": "conf",
        "--iou": "iou",
        "--no-calibrate": "no_calibrate",
        "--slow-calibrate": "slow_calibrate",
        "--tta-level": "tta_level",
        "--infer-mode": "infer_mode",
        "--tile-size": "tile_size",
        "--tile-overlap": "tile_overlap",
        "--tile-nms-iou": "tile_nms_iou",
        "--max-det": "max_det",
        "--ensemble-method": "ensemble_method",
        "--time-report-every": "time_report_every",
        "--time-budget-min": "time_budget_min",
        "--reserve-submit-min": "reserve_submit_min",
    }

    @classmethod
    def apply(cls, args, argv: Sequence[str]):
        preset_name = getattr(args, "preset", "manual")
        if preset_name == "manual":
            return args
        if preset_name not in cls.PRESETS:
            raise ValueError(f"알 수 없는 preset: {preset_name}")

        explicit = cls._explicit_attrs(argv)
        applied = {}
        for attr, value in cls.PRESETS[preset_name].items():
            if attr in explicit:
                continue
            if attr == "force_cuda" and "device" in explicit and str(getattr(args, "device", "")).lower() == "cpu":
                continue
            setattr(args, attr, value)
            applied[attr] = value

        if applied:
            print(f"[PRESET] {preset_name} 적용: {applied}")
        return args

    @classmethod
    def _explicit_attrs(cls, argv: Sequence[str]) -> set[str]:
        explicit = set()
        for token in argv[1:]:
            if not token.startswith("--"):
                continue
            attr = cls.OPTION_TO_ATTR.get(token.split("=", 1)[0])
            if attr:
                explicit.add(attr)
        return explicit


class RuntimePlanBuilder:
    """실행 전에 계산 가능한 작업량을 산정한다."""

    def __init__(self, cfg: type[CFG]):
        self.cfg = cfg

    def build(self, args, seeds: Sequence[int], model_names: Sequence[str]) -> RuntimePlan:
        train_imgs = sorted(self.cfg.TRAIN_IMG_DIR.glob("*.tif"))
        test_imgs = sorted(self.cfg.TEST_IMG_DIR.glob("*.tif"))
        n_train = len(train_imgs)
        n_test = len(test_imgs)
        val_est = max(1, int(round(n_train * float(args.val_ratio)))) if n_train else 0
        tta_count = TTAFactory.count(args.tta_level)
        calibration_grid = 0 if args.no_calibrate else len(self._conf_values(args.infer_mode)) * len(self._iou_values(args.infer_mode))
        train_jobs = 0 if args.skip_train else len(seeds) * len(model_names)
        inference_models = len(ParseUtils.parse_weight_list(args.weights, args.weight)) if args.skip_train and (args.weights or args.weight) else max(1, train_jobs)
        approx_tiles = self._approx_tiles_per_image(test_imgs, args.tile_size, args.tile_overlap, args.infer_mode)
        multiplier = approx_tiles if args.infer_mode == "tile" else approx_tiles + 1 if args.infer_mode == "hybrid" else 1
        test_units = inference_models * n_test * tta_count * multiplier
        if args.no_calibrate:
            calib_units = 0
        elif args.slow_calibrate:
            calib_units = train_jobs * val_est * calibration_grid * multiplier
        else:
            calib_units = train_jobs * val_est * multiplier
        return RuntimePlan(
            train_images=n_train,
            test_images=n_test,
            val_images_est=val_est,
            seeds=len(seeds),
            models=len(model_names),
            train_jobs=train_jobs,
            calibration_grid=calibration_grid,
            tta_count=tta_count,
            approx_tiles_per_image=approx_tiles,
            test_prediction_units=test_units,
            calibration_prediction_units=calib_units,
        )

    def _conf_values(self, infer_mode: str) -> tuple[float, ...]:
        return self.cfg.TILE_CALIB_CONFS if infer_mode in {"tile", "hybrid"} else self.cfg.CALIB_CONFS

    def _iou_values(self, infer_mode: str) -> tuple[float, ...]:
        return self.cfg.TILE_CALIB_IOUS if infer_mode in {"tile", "hybrid"} else self.cfg.CALIB_IOUS

    @staticmethod
    def _approx_tiles_per_image(test_imgs: list[Path], tile_size: int, tile_overlap: float, infer_mode: str) -> int:
        if infer_mode == "full" or not test_imgs:
            return 1
        sample = test_imgs[0]
        try:
            with rasterio.open(sample) as src:
                h, w = int(src.height), int(src.width)
        except Exception:
            img = cv2.imread(str(sample), cv2.IMREAD_COLOR)
            if img is None:
                return 1
            h, w = img.shape[:2]
        return len(TileHelper.make_tiles(h, w, tile_size=tile_size, overlap=tile_overlap))


# ──────────────────────────────────────────────────────────────
# 11. 메인 파이프라인
# ──────────────────────────────────────────────────────────────
class PipelineRunner:
    """전체 실행 흐름 조율 책임. 각 세부 책임은 서비스 클래스에 위임한다."""

    def __init__(self, cfg: type[CFG]):
        self.cfg = cfg

    def run(self, args) -> None:
        self.cfg.WORKERS = int(args.workers)
        estimator = RuntimeEstimator(report_every=args.time_report_every)

        # [클라우드 안전] 1) 데이터 폴더 해결 → set_base_dir
        try:
            data_dir = DataDirResolver.resolve(args.data_dir, self.cfg.BASE_DIR)
        except FileNotFoundError as e:
            print(f"\n[FATAL] {e}")
            print("실행을 중단합니다. --data-dir로 경로를 지정한 뒤 다시 시도하세요.")
            sys.exit(2)
        self.cfg.set_base_dir(data_dir)
        print(f"[DATA] 사용 데이터 폴더: {self.cfg.BASE_DIR}")

        if args.work_dir:
            old_submission = Path(args.submission)
            old_default_submission = self.cfg.SUBMISSION_CSV
            self.cfg.set_work_dir(Path(args.work_dir))
            if old_submission == old_default_submission:
                args.submission = str(self.cfg.SUBMISSION_CSV)
            print(f"[WORK] 이어달리기 작업 폴더: {self.cfg.YOLO_ROOT}")

        # [클라우드 안전] 2) 출력 디렉터리 쓰기 권한 사전 확인 + fallback
        self.cfg.ensure_writable_dirs()

        if args.inspect:
            DataInspector(self.cfg).inspect()
            return

        # [클라우드 안전] 3) CUDA 환경 사전 진단 (학습 시작 전에 빠르게 실패)
        if str(args.device).lower() != "cpu":
            try:
                DeviceManager.preflight_cuda()
            except RuntimeError as e:
                if args.force_cuda or str(args.device).lower() != "cpu":
                    print(f"\n[FATAL] CUDA 사전 진단 실패:\n{e}")
                    sys.exit(3)

        device = DeviceManager.resolve_device(requested_device=args.device, force_cuda=args.force_cuda)
        DeviceManager.optimize_torch_runtime(device=device)
        amp = DeviceManager.cuda_amp_enabled(device=device, no_amp=args.no_amp)
        half = DeviceManager.use_half_precision(device=device, no_half=args.no_half)

        model_names = ParseUtils.parse_model_list(args.models, args.model)
        all_seeds = ParseUtils.parse_seed_list(args.seeds)
        seeds = ParseUtils.filter_seeds_after(all_seeds, args.start_after_seed)
        if args.start_after_seed is not None and not args.skip_train:
            print(
                "[WARN] --start-after-seed를 쓰면 이전 seed는 이번 실행의 최종 앙상블에 포함되지 않습니다. "
                "180x2/200x2 최종 제출을 만들려면 --resume만 쓰고 seeds=42,77을 유지하세요."
            )
        self._print_header(args, model_names)

        plan = RuntimePlanBuilder(self.cfg).build(args=args, seeds=seeds, model_names=model_names)
        estimator.print_plan(plan)

        split_strategy = SplitStrategy(self.cfg.TRAIN_LBL_DIR)
        dataset_builder = DatasetBuilder(self.cfg, ImageIO, split_strategy, estimator)
        trainer = ModelTrainer(self.cfg, estimator)
        predictor = DetectionPredictor(device=device, half=half, estimator=estimator)
        calibrator = ThresholdCalibrator(self.cfg, predictor, estimator)
        inferencer = TestInferencer(self.cfg, predictor, estimator)
        submission_writer = SubmissionWriter(self.cfg)

        pred_dicts: list[dict[str, int]] = []
        validation_records: list[dict] = []

        if args.skip_train:
            pred_dicts, validation_records = self._run_skip_train(args, model_names, inferencer, estimator)
        else:
            pred_dicts, validation_records = self._run_full_training(
                args=args,
                seeds=seeds,
                model_names=model_names,
                dataset_builder=dataset_builder,
                trainer=trainer,
                calibrator=calibrator,
                inferencer=inferencer,
                estimator=estimator,
                device=device,
                amp=amp,
            )

        ensemble_detail = Path(args.submission).with_name("ensemble_prediction_detail.csv")
        final_pred = PredictionAggregator.aggregate(
            pred_dicts,
            method=args.ensemble_method,
            detail_path=ensemble_detail,
            validation_records=validation_records,
        )
        submission_writer.make_submission(final_pred, output_path=Path(args.submission))

        print("\n" + "=" * 60)
        print(f"  완료! → {args.submission} 를 제출하세요.")
        print(f"  전체 실행 시간: {estimator.fmt(estimator.elapsed())}")
        print("=" * 60 + "\n")

    def _print_header(self, args, model_names: Sequence[str]) -> None:
        print("\n" + "=" * 60)
        print("  2026 지능IoT해커톤 - 문제 2 건물 탐지 SOLID + ETA 파이프라인")
        print("=" * 60)
        print(f"  모델 목록: {list(model_names)}")
        print(f"  imgsz: {args.imgsz} | epochs: {args.epochs} | patience: {args.patience} | batch: {args.batch}")
        print(f"  min_box_px: {args.min_box_px} | val_ratio: {args.val_ratio} | split_mode: {args.split_mode}")
        print(f"  split_seed: {args.split_seed} | vary_split_by_seed: {args.vary_split_by_seed}")
        print(f"  device: {args.device} | workers: {args.workers} | cache: {args.cache_mode} | plots: {not args.no_plots}")
        print(f"  aug: degrees={args.aug_degrees} | mosaic={args.aug_mosaic} | scale={args.aug_scale} | translate={args.aug_translate} | close_mosaic={args.close_mosaic}")
        print(f"  aug: mixup={args.aug_mixup} | copy_paste={args.aug_copy_paste}")
        print(f"  deterministic: {args.deterministic} | time_budget: {args.time_budget_min}분 | reserve: {args.reserve_submit_min}분")
        print(f"  resume: {args.resume} | start_after_seed: {args.start_after_seed} | reuse_short_runs: {args.reuse_short_runs}")
        calib_mode = "off" if args.no_calibrate else "exact" if args.slow_calibrate else "fast-cache"
        print(f"  TTA: {args.tta_level} | infer_mode: {args.infer_mode} | calibration: {calib_mode} | max_det: {args.max_det}")
        if args.infer_mode in {"tile", "hybrid"}:
            print(f"  tile_size: {args.tile_size} | tile_overlap: {args.tile_overlap} | tile_nms_iou: {args.tile_nms_iou}")
        print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(미설정)')}")
        print("=" * 60 + "\n")

    def _run_skip_train(
        self,
        args,
        model_names: Sequence[str],
        inferencer: TestInferencer,
        estimator: RuntimeEstimator,
    ) -> tuple[list[dict[str, int]], list[dict]]:
        weights = ParseUtils.parse_weight_list(args.weights, args.weight)
        if not weights:
            default_model_stem = ParseUtils.safe_model_stem(model_names[0])
            default_weight = self.cfg.RUNS_DIR / f"{self.cfg.RUN_NAME_BASE}_s{self.cfg.SEED}_{default_model_stem}_img{args.imgsz}" / "weights" / "best.pt"
            weights = [default_weight]

        pred_dicts = []
        total = len(weights)
        for idx, w in enumerate(weights, start=1):
            if not w.exists():
                raise FileNotFoundError(f"가중치 파일 없음: {w}")
            estimator.start_job("skip-train 추론 job")
            pred = inferencer.predict_with_tta(
                weight_path=w,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                tta_level=args.tta_level,
                infer_mode=args.infer_mode,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                tile_nms_iou=args.tile_nms_iou,
            )
            pred_dicts.append(pred)
            estimator.end_job("skip-train 추론 job", idx, total)
        return pred_dicts, []

    def _run_full_training(
        self,
        args,
        seeds: Sequence[int],
        model_names: Sequence[str],
        dataset_builder: DatasetBuilder,
        trainer: ModelTrainer,
        calibrator: ThresholdCalibrator,
        inferencer: TestInferencer,
        estimator: RuntimeEstimator,
        device: str,
        amp: bool,
    ) -> tuple[list[dict[str, int]], list[dict]]:
        print(f"[앙상블] 학습 seed 목록: {list(seeds)}")
        print(f"[앙상블] 모델 목록: {list(model_names)}\n")

        pred_dicts: list[dict[str, int]] = []
        validation_records: list[dict] = []
        total_train_jobs = len(seeds) * len(model_names)
        train_done = 0
        experiment_tag = self._experiment_tag(args)
        if experiment_tag:
            print(f"[실험 태그] 기존 학습 결과와 분리: {experiment_tag}\n")

        for seed in seeds:
            yolo_dir = self.cfg.YOLO_ROOT / f"seed_{seed}{experiment_tag}"
            yaml_path = self._get_or_prepare_dataset(args, seed, yolo_dir, dataset_builder)

            for model_name in model_names:
                if self._should_skip_next_training_for_budget(args, estimator, train_done, total_train_jobs):
                    print("[BUDGET] 남은 시간 예산상 추가 학습 job을 생략하고 현재까지의 앙상블로 제출 파일을 생성합니다.\n")
                    return pred_dicts, validation_records

                model_stem = ParseUtils.safe_model_stem(model_name)
                run_name = f"{self.cfg.RUN_NAME_BASE}_s{seed}_{model_stem}_img{args.imgsz}{experiment_tag}"

                estimator.start_job("학습 job")
                best_pt = trainer.train_one(
                    yaml_path=yaml_path,
                    device=device,
                    model_name=model_name,
                    epochs=args.epochs,
                    batch=args.batch,
                    patience=args.patience,
                    amp=amp,
                    deterministic=args.deterministic,
                    cache_mode=args.cache_mode,
                    seed=seed,
                    run_name=run_name,
                    imgsz=args.imgsz,
                    max_det=args.max_det,
                    aug_degrees=args.aug_degrees,
                    aug_mosaic=args.aug_mosaic,
                    aug_scale=args.aug_scale,
                    aug_translate=args.aug_translate,
                    aug_mixup=args.aug_mixup,
                    aug_copy_paste=args.aug_copy_paste,
                    close_mosaic=args.close_mosaic,
                    workers=args.workers,
                    save_plots=not args.no_plots,
                    resume=args.resume,
                    reuse_short_runs=args.reuse_short_runs,
                )
                train_done += 1
                estimator.end_job("학습 job", train_done, total_train_jobs)

                # 학습 job 사이 메모리 정리
                DeviceManager.clear_cuda_memory()

                if not best_pt.exists():
                    raise FileNotFoundError(f"가중치 파일 없음: {best_pt}")

                best_conf, best_iou, best_tile_nms_iou, val_record = self._maybe_calibrate(
                    args,
                    best_pt,
                    yolo_dir,
                    calibrator,
                    estimator,
                    train_done,
                    total_train_jobs,
                    run_name=run_name,
                )

                estimator.start_job("테스트 추론 job")
                pred = inferencer.predict_with_tta(
                    weight_path=best_pt,
                    conf=best_conf,
                    iou=best_iou,
                    imgsz=args.imgsz,
                    max_det=args.max_det,
                    tta_level=args.tta_level,
                    infer_mode=args.infer_mode,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    tile_nms_iou=best_tile_nms_iou,
                )
                pred_dicts.append(pred)
                if val_record:
                    val_record["pred_index"] = len(pred_dicts) - 1
                    validation_records.append(val_record)
                estimator.end_job("테스트 추론 job", len(pred_dicts), total_train_jobs)

        return pred_dicts, validation_records

    @staticmethod
    def _experiment_tag(args) -> str:
        parts = []
        min_box_px = float(getattr(args, "min_box_px", 0.0) or 0.0)
        if min_box_px > 0:
            tag_value = f"{min_box_px:g}".replace(".", "p")
            parts.append(f"mbox{tag_value}")
        if getattr(args, "preset", "") == "rtx6000-small-3h-stable":
            parts.append("stable")
        preset_name = str(getattr(args, "preset", ""))
        if preset_name.startswith("h100-cloud"):
            parts.append(preset_name.replace("-", ""))
        return "_" + "_".join(parts) if parts else ""

    @staticmethod
    def _should_skip_next_training_for_budget(args, estimator: RuntimeEstimator, train_done: int, total_train_jobs: int) -> bool:
        budget_min = float(getattr(args, "time_budget_min", 0.0) or 0.0)
        if budget_min <= 0 or train_done <= 0 or train_done >= total_train_jobs:
            return False

        train_durations = estimator.job_durations.get("학습 job", [])
        if not train_durations:
            return False

        avg_train_sec = float(np.mean(train_durations))
        reserve_sec = max(0.0, float(getattr(args, "reserve_submit_min", 0.0) or 0.0)) * 60.0
        budget_sec = budget_min * 60.0
        projected_sec = estimator.elapsed() + avg_train_sec + reserve_sec

        if projected_sec <= budget_sec:
            return False

        remain_sec = max(0.0, budget_sec - estimator.elapsed())
        print(
            "[BUDGET] 다음 학습 job 예상치를 더하면 시간 예산을 넘을 가능성이 큽니다. "
            f"남은 시간 {RuntimeEstimator.fmt(remain_sec)}, "
            f"평균 학습 job {RuntimeEstimator.fmt(avg_train_sec)}, "
            f"보정/추론 reserve {RuntimeEstimator.fmt(reserve_sec)}"
        )
        return True

    def _get_or_prepare_dataset(self, args, seed: int, yolo_dir: Path, dataset_builder: DatasetBuilder) -> Path:
        yaml_path = yolo_dir / "dataset.yaml"
        split_seed = int(seed if args.vary_split_by_seed else args.split_seed)
        min_box_px = float(getattr(args, "min_box_px", 0.0) or 0.0)
        if args.resume and yaml_path.exists():
            split_seed_file = yolo_dir / "split_seed.txt"
            if split_seed_file.exists():
                old_split_seed = split_seed_file.read_text(encoding="utf-8").strip()
                if old_split_seed != str(split_seed):
                    print(
                        f"[WARN] 기존 데이터셋 split_seed={old_split_seed}, 현재 요청 split_seed={split_seed}. "
                        "공통 validation 자동 선택 정확도를 위해 새 작업 폴더 사용을 권장합니다."
                    )
            else:
                print("[WARN] 기존 데이터셋의 split_seed를 확인할 수 없습니다. 기존 seed_42 데이터셋이면 보통 문제 없습니다.")
            label_config = yolo_dir / "label_config.json"
            if label_config.exists():
                try:
                    old_min_box = float(json.loads(label_config.read_text(encoding="utf-8")).get("min_box_px", 0.0))
                    if abs(old_min_box - min_box_px) > 1e-6:
                        raise ValueError(f"min_box_px mismatch: existing={old_min_box}, requested={min_box_px}")
                except Exception as e:
                    raise RuntimeError(
                        f"기존 데이터셋 라벨 설정이 현재 설정과 다릅니다: {label_config}\n"
                        f"{e}\n"
                        "새 설정으로 학습하려면 다른 --work-dir을 쓰거나 해당 seed_* 데이터셋을 지운 뒤 실행하세요."
                    ) from e
            elif min_box_px > 0:
                raise RuntimeError(
                    f"기존 데이터셋에 label_config.json이 없습니다: {yolo_dir}\n"
                    "작은 건물 min_box_px 보정을 적용하려면 새 작업 폴더를 쓰거나 기존 seed 데이터셋을 지운 뒤 실행하세요."
                )
            print(f"[RESUME] 기존 seed 데이터셋 사용: {yaml_path}\n")
            return yaml_path

        if args.skip_convert:
            if not yaml_path.exists():
                raise FileNotFoundError(
                    f"dataset.yaml 없음: {yaml_path}\n"
                    "해당 seed 데이터셋이 없으면 --skip-convert 없이 먼저 실행하세요."
                )
            print(f"[STEP 1] 변환 생략 → 기존 yaml 사용: {yaml_path}\n")
            return yaml_path

        return dataset_builder.prepare(
            seed=split_seed,
            yolo_dir=yolo_dir,
            val_ratio=args.val_ratio,
            overwrite=not args.resume,
            split_mode=args.split_mode,
            min_box_px=min_box_px,
        )

    def _maybe_calibrate(
        self,
        args,
        best_pt: Path,
        yolo_dir: Path,
        calibrator: ThresholdCalibrator,
        estimator: RuntimeEstimator,
        done: int,
        total: int,
        run_name: str,
    ) -> tuple[float, float, float, dict]:
        if args.no_calibrate:
            print(f"[보정 생략] conf={args.conf:.2f}, iou={args.iou:.2f}, tile_nms_iou={args.tile_nms_iou:.2f} 사용\n")
            return args.conf, args.iou, args.tile_nms_iou, {}

        conf_values = self.cfg.TILE_CALIB_CONFS if args.infer_mode in {"tile", "hybrid"} else self.cfg.CALIB_CONFS
        iou_values = self.cfg.TILE_CALIB_IOUS if args.infer_mode in {"tile", "hybrid"} else self.cfg.CALIB_IOUS

        estimator.start_job("보정 job")
        best_conf, best_iou, best_tile_nms_iou, _, val_pred, val_true, val_stats = calibrator.calibrate(
            weight_path=best_pt,
            yolo_dir=yolo_dir,
            imgsz=args.imgsz,
            max_det=args.max_det,
            conf_values=conf_values,
            iou_values=iou_values,
            infer_mode=args.infer_mode,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            tile_nms_iou=args.tile_nms_iou,
            fast=not args.slow_calibrate,
        )
        estimator.end_job("보정 job", done, total)
        val_record = {
            "name": run_name,
            "val_pred": val_pred,
            "val_true": val_true,
            "rmse": val_stats.get("rmse", float("inf")),
            "mae": val_stats.get("mae", float("inf")),
            "bias": val_stats.get("bias", 0.0),
            "n": val_stats.get("n", 0.0),
            "conf": best_conf,
            "iou": best_iou,
            "tile_nms_iou": best_tile_nms_iou,
        }
        return best_conf, best_iou, best_tile_nms_iou, val_record


def main() -> None:
    args = ArgParserFactory.create().parse_args()
    args = PresetManager.apply(args, sys.argv)
    PipelineRunner(CFG).run(args)


if __name__ == "__main__":
    main()
