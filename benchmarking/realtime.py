"""End-to-end real-time benchmark for neonatal pain models.

The benchmark measures the deployed path (acquisition queue, face detection,
preprocessing, device transfer, prediction/uncertainty, XAI and temporal
smoothing), rather than timing an isolated forward pass.  Patient filenames are
never written to the result files.
"""

from __future__ import annotations

import copy
import csv
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import platform
import queue
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import cv2
import numpy as np
import psutil
import torch
import yaml
from PIL import Image, ImageFilter

from dataloaders.presets import PresetTransform

LOGGER = logging.getLogger(__name__)
_END = object()
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RGU_XAI_METHODS = (
    "IntegratedGradients",
    "Saliency",
    "DeepLift",
    "Occlusion",
    "GradCAM",
    "GuidedGradCAM",
    "Deconvolution",
    "GradientShap",
    "DeepLiftShap",
    "Lime",
)


class BenchmarkError(RuntimeError):
    """Expected configuration, input or runtime error with a concise message."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkError(message)


def _as_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the benchmark YAML without initializing any model."""
    config_path = _as_path(path)
    if not config_path.is_file():
        raise BenchmarkError(f"Arquivo de configuração não encontrado: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BenchmarkError(f"YAML inválido em {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise BenchmarkError("A configuração do benchmark deve ser um objeto YAML.")
    config["_config_path"] = str(config_path)
    validate_benchmark_config(config, check_files=False)
    return config


def _enabled_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = config.get("scenarios", [])
    _require(isinstance(scenarios, list) and scenarios, "'scenarios' deve conter ao menos um cenário.")
    enabled = [copy.deepcopy(item) for item in scenarios if item.get("enabled", True)]
    _require(enabled, "Nenhum cenário está habilitado na configuração.")
    return enabled


def resolve_checkpoint_patterns(patterns: Sequence[str]) -> list[Path]:
    """Resolve checkpoint paths/globs deterministically and without duplicates."""
    matches: list[Path] = []
    for pattern in patterns:
        expanded = os.path.expanduser(str(pattern))
        if not Path(expanded).is_absolute():
            expanded = str(Path.cwd() / expanded)
        found = [Path(p) for p in glob.glob(expanded, recursive=True)]
        if not found and not any(char in expanded for char in "*?[]"):
            found = [Path(expanded)]
        matches.extend(path.resolve() for path in found if path.is_file())
    return sorted(dict.fromkeys(matches), key=lambda path: str(path).lower())


def validate_benchmark_config(config: dict[str, Any], *, check_files: bool) -> None:
    source = config.get("source")
    _require(isinstance(source, dict), "Campo obrigatório ausente: 'source'.")
    source_kind = str(source.get("kind", "auto")).lower()
    _require(source_kind in {"auto", "video", "images", "camera"},
             "source.kind deve ser auto, video, images ou camera.")
    _require("path" in source or source_kind == "camera",
             "source.path é obrigatório, exceto para câmera.")
    queue_policy = str(source.get("queue_policy", "drop_oldest")).lower()
    _require(queue_policy in {"drop_oldest", "block"},
             "source.queue_policy deve ser drop_oldest ou block.")
    _require(int(source.get("queue_size", 4)) >= 1, "source.queue_size deve ser >= 1.")
    _require(int(source.get("max_frames", 300)) >= 1, "source.max_frames deve ser >= 1.")

    names: set[str] = set()
    for scenario in _enabled_scenarios(config):
        name = str(scenario.get("name", "")).strip()
        _require(name and name not in names, "Cada cenário deve ter um nome único e não vazio.")
        names.add(name)
        architecture = str(scenario.get("architecture", ""))
        _require(architecture in {"NCNN", "VGGFace", "ViT"},
                 f"Arquitetura inválida no cenário '{name}': {architecture}")
        mode = str(scenario.get("mode", "single")).lower()
        _require(mode in {"single", "mcdp", "ensemble"},
                 f"Modo inválido no cenário '{name}': {mode}")
        patterns = scenario.get("checkpoints")
        _require(isinstance(patterns, list) and patterns,
                 f"O cenário '{name}' precisa de uma lista 'checkpoints'.")
        if mode == "mcdp":
            _require(int(scenario.get("mcdp_passes", 50)) >= 2,
                     f"mcdp_passes deve ser >= 2 em '{name}'.")
            dropout_p = float(scenario.get("dropout_p", 0.5))
            _require(0.0 < dropout_p < 1.0, f"dropout_p inválido em '{name}'.")
        xai = scenario.get("xai", {}) or {}
        method = str(xai.get("method", "none")).lower()
        _require(method in {"none", "integrated_gradients", "gradcam", "both", "rgu_all_average"},
                 f"Método XAI inválido em '{name}': {method}")
        _require(int(xai.get("repetitions", 1)) >= 1, f"xai.repetitions inválido em '{name}'.")
        if method == "rgu_all_average":
            _require(int(xai.get("repetitions", 1)) == 1,
                     f"O workflow RGU executa cada XAI uma vez; repetitions deve ser 1 em '{name}'.")
        _require(int(xai.get("ig_steps", 1)) >= 1, f"xai.ig_steps inválido em '{name}'.")
        _require(str(xai.get("scope", "first")).lower() in {"first", "all"},
                 f"xai.scope inválido em '{name}'.")
        _require(int(scenario.get("smoothing_window", 1)) >= 1,
                 f"smoothing_window inválido em '{name}'.")

        if check_files:
            checkpoints = resolve_checkpoint_patterns([str(value) for value in patterns])
            _require(checkpoints, f"Nenhum checkpoint encontrado para '{name}': {patterns}")
            expected = 1 if mode in {"single", "mcdp"} else int(scenario.get("ensemble_size", 10))
            _require(len(checkpoints) == expected,
                     f"'{name}' requer {expected} checkpoint(s), mas encontrou {len(checkpoints)}.")

    if check_files and source_kind != "camera":
        source_path = _as_path(str(source["path"]))
        _require(source_path.exists(), f"Fonte de entrada não encontrada: {source_path}")
        if source_path.is_dir():
            has_image = any(path.suffix.lower() in _IMAGE_SUFFIXES for path in source_path.iterdir())
            _require(has_image, f"Nenhuma imagem compatível encontrada em: {source_path}")

    detector = config.get("detector", {}) or {}
    if check_files and detector.get("enabled", True):
        _require(importlib.util.find_spec("insightface") is not None,
                 "Dependência ausente: insightface. Instale requirements.txt.")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def latency_stats(values: Iterable[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "p95": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": len(clean),
        "mean": float(statistics.fmean(clean)),
        "std": float(statistics.pstdev(clean)) if len(clean) > 1 else 0.0,
        "median": float(np.median(clean)),
        "p95": percentile(clean, 95),
        "min": min(clean),
        "max": max(clean),
    }


def merge_rgu_xai_masks(attributions: Sequence[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    """Reproduce the simple-average merge from ``RGU_XAI_PAPER.ipynb``.

    Each HxWxC attribution is collapsed with ``abs(sum(channels))``, normalized
    independently to [0, 1], and then all normalized maps are averaged.
    """
    _require(bool(attributions), "Nenhuma atribuição XAI foi fornecida para a fusão.")
    normalized: list[np.ndarray] = []
    expected_shape: Optional[tuple[int, int]] = None
    for attribution in attributions:
        mask = np.asarray(attribution)
        _require(mask.ndim == 3, f"Atribuição XAI deve ser HxWxC; recebido {mask.shape}.")
        collapsed = np.abs(np.sum(mask, axis=2))
        current_shape = tuple(int(value) for value in collapsed.shape)
        if expected_shape is None:
            expected_shape = current_shape
        _require(current_shape == expected_shape,
                 f"Máscaras XAI com formas incompatíveis: {expected_shape} e {current_shape}.")
        minimum = float(np.nanmin(collapsed))
        maximum = float(np.nanmax(collapsed))
        safe = np.nan_to_num(collapsed, nan=minimum, posinf=maximum, neginf=minimum)
        normalized.append((safe - minimum) / (maximum - minimum + eps))
    return np.mean(np.stack(normalized, axis=0), axis=0)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class FrameItem:
    index: int
    frame: np.ndarray
    captured_at: float
    enqueued_at: float
    acquisition_ms: float
    queue_depth: int = 0


@dataclass
class SourceStats:
    frames_read: int = 0
    read_failures: int = 0
    frames_enqueued: int = 0
    frames_dropped_queue: int = 0
    acquisition_ms: list[float] = field(default_factory=list)
    effective_source_fps: Optional[float] = None
    first_capture_at: Optional[float] = None
    last_capture_at: Optional[float] = None


class FrameSource:
    """OpenCV/image-directory source that never exposes filenames in results."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.kind = str(config.get("kind", "auto")).lower()
        self.path_value = config.get("path", 0)
        self._image_paths: list[Path] = []
        self._capture: Optional[cv2.VideoCapture] = None
        self.fps: Optional[float] = None

    def open(self) -> None:
        if self.kind == "auto":
            path = _as_path(str(self.path_value))
            self.kind = "images" if path.is_dir() else "video"
        if self.kind == "images":
            directory = _as_path(str(self.path_value))
            self._image_paths = sorted(
                (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
                key=lambda path: path.name.lower(),
            )
            _require(bool(self._image_paths), f"Nenhuma imagem compatível em: {directory}")
            configured_fps = self.config.get("source_fps")
            self.fps = float(configured_fps) if configured_fps else None
            return
        capture_arg: int | str
        if self.kind == "camera":
            capture_arg = int(self.config.get("camera_index", self.path_value or 0))
        else:
            capture_arg = str(_as_path(str(self.path_value)))
        self._capture = cv2.VideoCapture(capture_arg)
        _require(self._capture.isOpened(), f"Não foi possível abrir a fonte {self.kind}.")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if math.isfinite(fps) and fps > 0 else None

    def read(self, index: int) -> tuple[bool, Optional[np.ndarray]]:
        if self.kind == "images":
            if index >= len(self._image_paths):
                return False, None
            return True, cv2.imread(str(self._image_paths[index]), cv2.IMREAD_COLOR)
        assert self._capture is not None
        return self._capture.read()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class FrameProducer(threading.Thread):
    def __init__(self, source_config: dict[str, Any], output_queue: queue.Queue[Any]):
        super().__init__(name="benchmark-frame-producer", daemon=True)
        self.source_config = source_config
        self.output_queue = output_queue
        self.stats = SourceStats()
        self.error: Optional[BaseException] = None
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        try:
            self._produce()
        except BaseException as exc:  # forwarded to the consumer thread
            self.error = exc
        finally:
            while not self._stop_requested.is_set():
                try:
                    self.output_queue.put(_END, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _produce(self) -> None:
        max_frames = int(self.source_config.get("max_frames", 300))
        realtime = bool(self.source_config.get("realtime", True))
        target_fps_value = self.source_config.get("target_fps")
        policy = str(self.source_config.get("queue_policy", "drop_oldest")).lower()
        with FrameSource(self.source_config) as source:
            target_fps = float(target_fps_value) if target_fps_value else source.fps
            if realtime:
                _require(target_fps is not None and target_fps > 0,
                         "Modo realtime requer target_fps ou FPS válido na fonte.")
            self.stats.effective_source_fps = target_fps
            start = time.perf_counter()
            for index in range(max_frames):
                if self._stop_requested.is_set():
                    break
                if realtime and target_fps:
                    scheduled = start + index / target_fps
                    remaining = scheduled - time.perf_counter()
                    if remaining > 0 and self._stop_requested.wait(remaining):
                        break
                read_start = time.perf_counter()
                ok, frame = source.read(index)
                read_end = time.perf_counter()
                if not ok:
                    break
                self.stats.frames_read += 1
                if self.stats.first_capture_at is None:
                    self.stats.first_capture_at = read_end
                self.stats.last_capture_at = read_end
                acquisition_ms = (read_end - read_start) * 1000.0
                self.stats.acquisition_ms.append(acquisition_ms)
                if frame is None or frame.size == 0:
                    self.stats.read_failures += 1
                    continue
                item = FrameItem(
                    index=index,
                    frame=frame,
                    captured_at=read_end,
                    enqueued_at=time.perf_counter(),
                    acquisition_ms=acquisition_ms,
                    queue_depth=min(self.output_queue.qsize() + 1, self.output_queue.maxsize),
                )
                if policy == "block":
                    while not self._stop_requested.is_set():
                        try:
                            self.output_queue.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if self._stop_requested.is_set():
                        break
                else:
                    try:
                        self.output_queue.put_nowait(item)
                    except queue.Full:
                        try:
                            old = self.output_queue.get_nowait()
                            if old is not _END:
                                self.stats.frames_dropped_queue += 1
                        except queue.Empty:
                            pass
                        self.output_queue.put_nowait(item)
                self.stats.frames_enqueued += 1


def read_warmup_frame(source_config: dict[str, Any]) -> np.ndarray:
    with FrameSource(source_config) as source:
        ok, frame = source.read(0)
    if not ok or frame is None or frame.size == 0:
        raise BenchmarkError("Não foi possível ler um quadro para o aquecimento do pipeline.")
    return frame


class ResourceSampler:
    """Low-frequency process/system/GPU sampler with graceful NVML fallback."""

    def __init__(self, interval_s: float = 0.1, gpu_index: int = 0):
        self.interval_s = max(0.02, float(interval_s))
        self.gpu_index = gpu_index
        self.rows: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid())
        self._pynvml: Any = None
        self._gpu_handle: Any = None
        self._smi_process: Optional[subprocess.Popen[str]] = None
        self._smi_thread: Optional[threading.Thread] = None
        self._smi_lock = threading.Lock()
        self._smi_latest: dict[str, float] = {}
        self.gpu_backend = "unavailable"

    def _initialize_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self.gpu_backend = "pynvml"
        except Exception:
            self._pynvml = None
            self._gpu_handle = None

    def _initialize_nvidia_smi(self) -> None:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return
        command = [
            executable,
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            str(max(100, int(self.interval_s * 1000))),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._smi_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                creationflags=creationflags,
            )
            self._smi_thread = threading.Thread(
                target=self._read_nvidia_smi,
                name="benchmark-nvidia-smi-reader",
                daemon=True,
            )
            self._smi_thread.start()
            self.gpu_backend = "nvidia-smi"
        except OSError:
            self._smi_process = None

    def _read_nvidia_smi(self) -> None:
        process = self._smi_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            values = [value.strip() for value in line.strip().split(",")]
            if len(values) < 3:
                continue
            try:
                latest = {
                    "gpu_util_pct": float(values[0]),
                    "gpu_memory_util_pct": float(values[1]),
                    "gpu_memory_used_mb": float(values[2]),
                }
            except ValueError:
                continue
            with self._smi_lock:
                self._smi_latest = latest
            if self._stop.is_set():
                break

    def start(self) -> None:
        self._initialize_nvml()
        if self._pynvml is None:
            self._initialize_nvidia_smi()
        self._process.cpu_percent(None)
        self._thread = threading.Thread(target=self._sample_loop, name="benchmark-resource-sampler", daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        started = time.perf_counter()
        while not self._stop.is_set():
            memory = self._process.memory_info()
            row = {
                "time_s": time.perf_counter() - started,
                "process_cpu_pct": self._process.cpu_percent(None),
                "process_rss_mb": memory.rss / (1024.0 ** 2),
                "system_memory_pct": psutil.virtual_memory().percent,
            }
            if self._pynvml is not None and self._gpu_handle is not None:
                try:
                    utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    gpu_memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    row.update({
                        "gpu_util_pct": float(utilization.gpu),
                        "gpu_memory_util_pct": float(utilization.memory),
                        "gpu_memory_used_mb": gpu_memory.used / (1024.0 ** 2),
                    })
                except Exception:
                    pass
            elif self._smi_process is not None:
                with self._smi_lock:
                    row.update(self._smi_latest)
            self.rows.append(row)
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._smi_process is not None:
            self._smi_process.terminate()
            try:
                self._smi_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._smi_process.kill()
            if self._smi_process.stdout is not None:
                self._smi_process.stdout.close()
        if self._smi_thread is not None:
            self._smi_thread.join(timeout=2.0)
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 3))
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def summary(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "sample_count": len(self.rows),
            "sample_interval_s": self.interval_s,
            "gpu_backend": self.gpu_backend,
        }
        keys = sorted({key for row in self.rows for key in row if key != "time_s"})
        for key in keys:
            stats = latency_stats(row[key] for row in self.rows if key in row)
            output[key] = {"mean": stats["mean"], "p95": stats["p95"], "max": stats["max"]}
        return output


class FaceDetector:
    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", True))
        self.selection = str(config.get("selection", "first")).lower()
        self.min_score = float(config.get("min_score", 0.0))
        self.app: Any = None
        self.providers: list[str] = []
        if not self.enabled:
            return
        _require(self.selection in {"first", "largest", "highest_score"},
                 "detector.selection deve ser first, largest ou highest_score.")
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise BenchmarkError("InsightFace não está instalado; execute pip install -r requirements.txt.") from exc
        modules = list(config.get("allowed_modules", ["detection", "landmark_2d_106"]))
        providers = list(config.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
        self.app = FaceAnalysis(allowed_modules=modules, providers=providers)
        det_size = tuple(int(value) for value in config.get("det_size", [640, 640]))
        _require(len(det_size) == 2 and min(det_size) > 0, "detector.det_size inválido.")
        self.app.prepare(ctx_id=int(config.get("ctx_id", 0)), det_size=det_size)
        try:
            self.providers = list(self.app.models["detection"].session.get_providers())
        except Exception:
            self.providers = providers

    def detect_and_crop(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], int, Optional[float]]:
        if not self.enabled:
            return frame, 1, None
        faces = self.app.get(frame)
        if not faces:
            return None, 0, None

        def score(face: Any) -> float:
            value = getattr(face, "det_score", None)
            if value is None:
                try:
                    value = face["det_score"]
                except Exception:
                    value = 0.0
            return float(value)

        if self.selection == "largest":
            selected = max(faces, key=lambda face: max(0.0, float(face.bbox[2] - face.bbox[0])) *
                                                   max(0.0, float(face.bbox[3] - face.bbox[1])))
        elif self.selection == "highest_score":
            selected = max(faces, key=score)
        else:
            selected = faces[0]
        det_score = score(selected)
        if det_score < self.min_score:
            return None, len(faces), det_score
        bbox = np.asarray(selected.bbox, dtype=float)
        height, width = frame.shape[:2]
        x1 = max(0, min(width, int(math.floor(bbox[0]))))
        y1 = max(0, min(height, int(math.floor(bbox[1]))))
        x2 = max(0, min(width, int(math.ceil(bbox[2]))))
        y2 = max(0, min(height, int(math.ceil(bbox[3]))))
        if x2 <= x1 or y2 <= y1:
            return None, len(faces), det_score
        crop = frame[y1:y2, x1:x2]
        return (crop if crop.size else None), len(faces), det_score


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, function: Callable[[], Any]) -> tuple[Any, float]:
    """Return synchronized wall-clock time for a CPU or CUDA stage."""
    _synchronize(device)
    started = time.perf_counter()
    result = function()
    _synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    _ = device  # weights are staged on CPU to avoid transient double VRAM use
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise BenchmarkError(f"Checkpoint não contém state_dict: {path.name}")
    return state


def _new_model(architecture: str) -> torch.nn.Module:
    from models import NCNN, VGGFace, ViT

    if architecture == "NCNN":
        return NCNN()
    if architecture == "VGGFace":
        return VGGFace()
    if architecture == "ViT":
        return ViT(weights=None)
    raise BenchmarkError(f"Arquitetura desconhecida: {architecture}")


def _enable_mc_dropout(model: torch.nn.Module, probability: float) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.p = probability
            module.train()
            count += 1
    return count


class ModelRunner:
    def __init__(self, scenario: dict[str, Any], device: torch.device):
        self.scenario = scenario
        self.device = device
        self.architecture = str(scenario["architecture"])
        self.mode = str(scenario.get("mode", "single")).lower()
        self.checkpoints = resolve_checkpoint_patterns([str(item) for item in scenario["checkpoints"]])
        expected = 1 if self.mode in {"single", "mcdp"} else int(scenario.get("ensemble_size", 10))
        _require(len(self.checkpoints) == expected,
                 f"{scenario['name']}: esperado(s) {expected} checkpoint(s), encontrado(s) {len(self.checkpoints)}.")
        self.models: list[torch.nn.Module] = []
        for checkpoint in self.checkpoints:
            model = _new_model(self.architecture)
            model.load_state_dict(_load_state_dict(checkpoint, device), strict=True)
            model.to(device).eval()
            self.models.append(model)
        self.dropout_layers = 0
        if self.mode == "mcdp":
            self.dropout_layers = _enable_mc_dropout(self.models[0], float(scenario.get("dropout_p", 0.5)))
            _require(self.dropout_layers > 0,
                     f"{scenario['name']}: MCDP solicitado, mas o modelo não possui Dropout.")
        self.transform = PresetTransform(self.architecture).transforms
        self.xai_objects: list[dict[str, Any]] = []
        self._configure_xai()

    @property
    def parameter_count(self) -> int:
        return sum(sum(parameter.numel() for parameter in model.parameters()) for model in self.models)

    @property
    def checkpoint_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.checkpoints)

    def _target_layer(self, model: torch.nn.Module) -> tuple[torch.nn.Module, bool]:
        if self.architecture == "NCNN":
            return model.merge_branch[0], False  # type: ignore[attr-defined]
        if self.architecture == "VGGFace":
            return model.VGGFace.features.conv5_3, False  # type: ignore[attr-defined]
        return model.ViT.encoder.layers.encoder_layer_11.ln_1, True  # type: ignore[attr-defined]

    def _rgu_target_layer(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.architecture == "NCNN":
            return model.merge_branch[0]  # type: ignore[attr-defined]
        if self.architecture == "VGGFace":
            return model.VGGFace.features.conv5_3  # type: ignore[attr-defined]
        return model.ViT.conv_proj  # type: ignore[attr-defined]

    def _configure_xai(self) -> None:
        xai = self.scenario.get("xai", {}) or {}
        method = str(xai.get("method", "none")).lower()
        if method == "none":
            return
        selected_models = self.models if str(xai.get("scope", "first")).lower() == "all" else self.models[:1]
        if method == "rgu_all_average":
            from captum.attr import (
                DeepLift,
                DeepLiftShap,
                Deconvolution,
                GradientShap,
                GuidedGradCam,
                IntegratedGradients,
                LayerGradCam,
                Lime,
                Occlusion,
                Saliency,
            )

            for model in selected_models:
                layer = self._rgu_target_layer(model)
                self.xai_objects.append({
                    "IntegratedGradients": IntegratedGradients(model),
                    "Saliency": Saliency(model),
                    "DeepLift": DeepLift(model),
                    "Occlusion": Occlusion(model),
                    "GradCAM": LayerGradCam(model, layer),
                    "GuidedGradCAM": GuidedGradCam(model, layer),
                    "Deconvolution": Deconvolution(model),
                    "GradientShap": GradientShap(model),
                    "DeepLiftShap": DeepLiftShap(model),
                    "Lime": Lime(model),
                })
            return
        from XAI.GradCAM import GradCAM
        from XAI.IntegratedGradients import IntegratedGradients

        for model in selected_models:
            objects: dict[str, Any] = {}
            if method in {"integrated_gradients", "both"}:
                objects["integrated_gradients"] = IntegratedGradients(
                    model, device=str(self.device), n_steps=int(xai.get("ig_steps", 1))
                )
            if method in {"gradcam", "both"}:
                layer, reshape = self._target_layer(model)
                objects["gradcam"] = GradCAM(
                    model, layer, device=str(self.device), reshape_transform_ViT=reshape
                )
            self.xai_objects.append(objects)

    def preprocess(self, face_bgr: np.ndarray) -> tuple[torch.Tensor, Image.Image]:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return self.transform(image).unsqueeze(0), image

    def infer(self, tensor: torch.Tensor) -> tuple[float, float]:
        probabilities: list[float] = []
        with torch.inference_mode():
            if self.mode == "mcdp":
                passes = int(self.scenario.get("mcdp_passes", 50))
                model = self.models[0]
                for _ in range(passes):
                    probabilities.append(float(model.predict(tensor).reshape(-1)[0].item()))  # type: ignore[attr-defined]
            else:
                for model in self.models:
                    probabilities.append(float(model.predict(tensor).reshape(-1)[0].item()))  # type: ignore[attr-defined]
        return float(np.mean(probabilities)), float(np.std(probabilities))

    @staticmethod
    def _rgu_feature_mask(tensor: torch.Tensor, n_segments: int) -> torch.Tensor:
        from skimage.segmentation import slic

        image = tensor.detach().cpu().squeeze(0).numpy()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = np.transpose(image, (1, 2, 0))
        segments = slic(
            image,
            n_segments=n_segments,
            compactness=10.0,
            sigma=0.0,
            start_label=0,
            channel_axis=2,
        )
        return torch.from_numpy(segments).long().unsqueeze(0).unsqueeze(0).to(tensor.device).contiguous()

    def _explain_rgu_all_average(self, tensor: torch.Tensor, image: Image.Image) -> float:
        from captum.attr import LayerAttribution

        _ = tensor  # inference input is separate; the notebook rebuilds its resized XAI input
        xai = self.scenario.get("xai", {}) or {}
        image_size = 120 if self.architecture == "NCNN" else 224
        # The notebook explicitly resizes before constructing both the model
        # input and the Gaussian-blurred DeepLift/DeepLiftShap baseline.
        xai_image = image.resize((image_size, image_size))
        xai_tensor = self.transform(xai_image).unsqueeze(0).to(self.device, non_blocking=False)
        blurred_image = xai_image.filter(ImageFilter.GaussianBlur(radius=float(xai.get("blur_radius", 5))))
        blurred = self.transform(blurred_image).unsqueeze(0).to(self.device, non_blocking=False)
        target_shape = tuple(int(value) for value in xai_tensor.shape[-2:])
        model_maps: list[np.ndarray] = []

        for model, explainers in zip(
            self.models if str(xai.get("scope", "first")).lower() == "all" else self.models[:1],
            self.xai_objects,
        ):
            # The paper notebook explains an eval-mode model. MCDP is restored
            # afterwards for the next uncertainty inference.
            model.eval()
            feature_mask: Optional[torch.Tensor] = None
            attributions_for_merge: list[np.ndarray] = []
            for name, explainer in explainers.items():
                method_input = xai_tensor.clone().detach().requires_grad_(True)
                kwargs: dict[str, Any] = {}
                if name == "IntegratedGradients":
                    kwargs = {"internal_batch_size": int(xai.get("ig_internal_batch_size", 10))}
                elif name == "DeepLift":
                    method_input = method_input.contiguous()
                    kwargs = {"baselines": blurred}
                elif name == "Occlusion":
                    kwargs = {
                        "sliding_window_shapes": tuple(xai.get("occlusion_window", [3, 20, 20])),
                        "strides": tuple(xai.get("occlusion_strides", [3, 5, 5])),
                    }
                elif name == "GradientShap":
                    kwargs = {
                        "baselines": torch.zeros_like(method_input),
                        "n_samples": int(xai.get("gradient_shap_samples", 10)),
                        "stdevs": float(xai.get("gradient_shap_stdevs", 0.0)),
                    }
                elif name == "DeepLiftShap":
                    kwargs = {"baselines": blurred.repeat(int(xai.get("deeplift_shap_baselines", 10)), 1, 1, 1)}
                elif name == "Lime":
                    method_input = method_input.contiguous()
                    if feature_mask is None:
                        feature_mask = self._rgu_feature_mask(
                            method_input, n_segments=int(xai.get("lime_segments", 100))
                        )
                    kwargs = {
                        "baselines": torch.zeros_like(method_input),
                        "feature_mask": feature_mask,
                        "n_samples": int(xai.get("lime_samples", 500)),
                        "perturbations_per_eval": int(xai.get("lime_perturbations_per_eval", 64)),
                        "show_progress": False,
                    }

                attribution = explainer.attribute(method_input, **kwargs)
                if name == "GradCAM":
                    attribution = LayerAttribution.interpolate(
                        attribution, target_shape, interpolate_mode="bilinear"
                    ).repeat(1, 3, 1, 1)
                attribution_np = (
                    attribution.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                )
                attributions_for_merge.append(attribution_np)

            model_maps.append(merge_rgu_xai_masks(attributions_for_merge))
            if self.mode == "mcdp":
                _enable_mc_dropout(model, float(self.scenario.get("dropout_p", 0.5)))

        # scope=all first builds one 10-method map per ensemble member, then
        # averages those model maps into the single map returned for the image.
        merged = np.mean(np.stack(model_maps, axis=0), axis=0)
        return float(np.mean(merged))

    def explain(self, tensor: torch.Tensor, image: Image.Image) -> float:
        if not self.xai_objects:
            return float("nan")
        xai = self.scenario.get("xai", {}) or {}
        if str(xai.get("method", "none")).lower() == "rgu_all_average":
            return self._explain_rgu_all_average(tensor, image)
        repetitions = int(xai.get("repetitions", 1))
        postprocess = bool(xai.get("postprocess", False))
        processor: Optional[Callable[..., Any]] = None
        if postprocess:
            from XAI.post_processing import kmeans_post_processing

            processor = kmeans_post_processing
        for _ in range(repetitions):
            for objects in self.xai_objects:
                for explainer in objects.values():
                    masks = explainer.attribution_mask(tensor)
                    if processor is not None:
                        for mask in masks:
                            processor(mask, use_mini_batch=bool(xai.get("mini_batch_kmeans", False)))
        return float("nan")


class ProbabilitySmoother:
    """Causal moving average; unlike a centered window, it is deployable online."""

    def __init__(self, window: int):
        self.window = max(1, int(window))
        self.probabilities: deque[float] = deque(maxlen=self.window)
        self.uncertainties: deque[float] = deque(maxlen=self.window)

    def update(self, probability: float, uncertainty: float) -> tuple[float, float]:
        self.probabilities.append(float(probability))
        self.uncertainties.append(float(uncertainty))
        return statistics.fmean(self.probabilities), statistics.fmean(self.uncertainties)

    def reset(self) -> None:
        self.probabilities.clear()
        self.uncertainties.clear()


class RealtimePipeline:
    def __init__(self, scenario: dict[str, Any], detector_config: dict[str, Any], device: torch.device):
        self.device = device
        self.detector = FaceDetector(detector_config)
        self.runner = ModelRunner(scenario, device)
        self.smoother = ProbabilitySmoother(int(scenario.get("smoothing_window", 1)))

    def reset_stream_state(self) -> None:
        self.smoother.reset()

    def process(self, item: FrameItem) -> dict[str, Any]:
        process_started = time.perf_counter()
        queue_wait_ms = (process_started - item.enqueued_at) * 1000.0
        row: dict[str, Any] = {
            "frame_index": item.index,
            "acquisition_ms": item.acquisition_ms,
            "queue_wait_ms": queue_wait_ms,
            "queue_depth": item.queue_depth,
            "face_valid": False,
            "faces_detected": 0,
            "detection_score": float("nan"),
            "detection_ms": float("nan"),
            "preprocess_ms": float("nan"),
            "transfer_ms": float("nan"),
            "inference_ms": float("nan"),
            "xai_ms": float("nan"),
            "smoothing_ms": float("nan"),
            "compute_ms": float("nan"),
            "end_to_end_ms": float("nan"),
            "probability_raw": float("nan"),
            "uncertainty_raw": float("nan"),
            "probability_smoothed": float("nan"),
            "uncertainty_smoothed": float("nan"),
            "xai_merged_map_mean": float("nan"),
        }

        (face, face_count, detection_score), row["detection_ms"] = timed_call(
            self.device, lambda: self.detector.detect_and_crop(item.frame)
        )
        row["faces_detected"] = face_count
        row["detection_score"] = detection_score if detection_score is not None else float("nan")
        if face is None:
            completed = time.perf_counter()
            row["compute_ms"] = (completed - process_started) * 1000.0
            row["end_to_end_ms"] = (completed - item.captured_at) * 1000.0
            row["completed_at"] = completed
            row["captured_at"] = item.captured_at
            return row

        row["face_valid"] = True
        (cpu_tensor, pil_image), row["preprocess_ms"] = timed_call(
            self.device, lambda: self.runner.preprocess(face)
        )
        tensor, row["transfer_ms"] = timed_call(
            self.device, lambda: cpu_tensor.to(self.device, non_blocking=False)
        )
        (probability, uncertainty), row["inference_ms"] = timed_call(
            self.device, lambda: self.runner.infer(tensor)
        )
        row["xai_merged_map_mean"], row["xai_ms"] = timed_call(
            self.device, lambda: self.runner.explain(tensor, pil_image)
        )
        (smooth_probability, smooth_uncertainty), row["smoothing_ms"] = timed_call(
            self.device, lambda: self.smoother.update(probability, uncertainty)
        )
        completed = time.perf_counter()
        row.update({
            "probability_raw": probability,
            "uncertainty_raw": uncertainty,
            "probability_smoothed": smooth_probability,
            "uncertainty_smoothed": smooth_uncertainty,
            "compute_ms": (completed - process_started) * 1000.0,
            "end_to_end_ms": (completed - item.captured_at) * 1000.0,
            "completed_at": completed,
            "captured_at": item.captured_at,
        })
        return row


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if bool(config.get("allow_cpu_fallback", False)):
            LOGGER.warning("CUDA indisponível; usando CPU porque allow_cpu_fallback=true.")
            requested = "cpu"
        else:
            raise BenchmarkError("CUDA foi solicitada, mas não está disponível. Use device=cpu ou habilite fallback.")
    return torch.device(requested)


def _environment(device: torch.device, detector: FaceDetector) -> dict[str, Any]:
    output: dict[str, Any] = {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "opencv": cv2.__version__,
        "insightface": _package_version("insightface"),
        "onnxruntime_gpu": _package_version("onnxruntime-gpu"),
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "device": str(device),
        "detector_providers": detector.providers,
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "system_ram_gb": psutil.virtual_memory().total / (1024.0 ** 3),
    }
    if device.type == "cuda":
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)
        output.update({
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": props.name,
            "gpu_total_memory_gb": props.total_memory / (1024.0 ** 3),
        })
    return output


def _torch_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
        "allocated_end_mb": torch.cuda.memory_allocated(device) / (1024.0 ** 2),
        "reserved_end_mb": torch.cuda.memory_reserved(device) / (1024.0 ** 2),
    }


def summarize_run(
    scenario: dict[str, Any],
    rows: list[dict[str, Any]],
    source_stats: SourceStats,
    resources: ResourceSampler,
    pipeline: RealtimePipeline,
    environment: dict[str, Any],
) -> dict[str, Any]:
    _require(bool(rows), f"O cenário '{scenario['name']}' não processou nenhum quadro.")
    first_capture = min(float(row["captured_at"]) for row in rows)
    last_completion = max(float(row["completed_at"]) for row in rows)
    duration_s = max(last_completion - first_capture, 1e-9)
    valid_rows = [row for row in rows if row["face_valid"]]
    stages = [
        "acquisition_ms", "queue_wait_ms", "detection_ms", "preprocess_ms", "transfer_ms",
        "inference_ms", "xai_ms", "smoothing_ms", "compute_ms", "end_to_end_ms",
    ]
    latency = {stage: latency_stats(row[stage] for row in rows) for stage in stages}
    source_fps = source_stats.effective_source_fps
    deadline_ms = 1000.0 / source_fps if source_fps and source_fps > 0 else None
    smoothing_window = int(scenario.get("smoothing_window", 1))
    smoothing_fill_delay_s = (
        (smoothing_window - 1) / source_fps if source_fps and source_fps > 0 else float("nan")
    )
    deadline_misses = (
        sum(float(row["end_to_end_ms"]) > deadline_ms for row in rows)
        if deadline_ms is not None else None
    )
    face_valid_count = len(valid_rows)
    frames_read = source_stats.frames_read
    dropped_or_invalid = source_stats.frames_dropped_queue + source_stats.read_failures + (len(rows) - face_valid_count)
    if (source_stats.first_capture_at is not None and source_stats.last_capture_at is not None
            and source_stats.frames_read > 1 and source_stats.last_capture_at > source_stats.first_capture_at):
        observed_source_fps = (source_stats.frames_read - 1) / (
            source_stats.last_capture_at - source_stats.first_capture_at
        )
    else:
        observed_source_fps = float("nan")
    summary: dict[str, Any] = {
        "scenario": str(scenario["name"]),
        "architecture": str(scenario["architecture"]),
        "mode": str(scenario.get("mode", "single")),
        "status": "ok",
        "environment": environment,
        "configuration": {
            "model_count": len(pipeline.runner.models),
            "mcdp_passes": int(scenario.get("mcdp_passes", 1)),
            "dropout_p": float(scenario.get("dropout_p", 0.0)),
            "dropout_layers": pipeline.runner.dropout_layers,
            "xai": copy.deepcopy(scenario.get("xai", {"method": "none"})),
            "xai_methods": list(RGU_XAI_METHODS)
            if (scenario.get("xai", {}) or {}).get("method") == "rgu_all_average" else [],
            "xai_merge": "per_method_minmax_then_simple_average"
            if (scenario.get("xai", {}) or {}).get("method") == "rgu_all_average" else "none",
            "smoothing_window": smoothing_window,
            "smoothing_type": "causal_moving_average",
            "smoothing_full_window_delay_s": smoothing_fill_delay_s,
            "parameter_count_total": pipeline.runner.parameter_count,
            "checkpoint_size_mb_total": pipeline.runner.checkpoint_bytes / (1024.0 ** 2),
        },
        "counts": {
            "frames_read": frames_read,
            "read_failures": source_stats.read_failures,
            "frames_enqueued": source_stats.frames_enqueued,
            "frames_dropped_queue": source_stats.frames_dropped_queue,
            "frames_processed": len(rows),
            "faces_valid": face_valid_count,
            "face_detection_failures": len(rows) - face_valid_count,
            "total_dropped_or_invalid": dropped_or_invalid,
        },
        "rates": {
            "queue_drop_pct_of_read": 100.0 * source_stats.frames_dropped_queue / frames_read if frames_read else float("nan"),
            "read_failure_pct": 100.0 * source_stats.read_failures / frames_read if frames_read else float("nan"),
            "valid_face_pct_of_processed": 100.0 * face_valid_count / len(rows),
            "end_to_end_coverage_pct_of_read": 100.0 * face_valid_count / frames_read if frames_read else float("nan"),
            "deadline_miss_pct": 100.0 * deadline_misses / len(rows) if deadline_misses is not None else float("nan"),
        },
        "throughput": {
            "measurement_duration_s": duration_s,
            "processed_fps": len(rows) / duration_s,
            "valid_output_fps": face_valid_count / duration_s,
            "source_target_fps": source_fps,
            "source_observed_fps": observed_source_fps,
        },
        "source": {
            "acquisition_ms_all_read": latency_stats(source_stats.acquisition_ms),
        },
        "queue": {
            "depth_mean": float(statistics.fmean(float(row["queue_depth"]) for row in rows)),
            "depth_p95": percentile([float(row["queue_depth"]) for row in rows], 95),
            "depth_max": max(int(row["queue_depth"]) for row in rows),
        },
        "latency_ms": latency,
        "resources": resources.summary(),
        "torch_gpu_memory": _torch_memory(pipeline.device),
        "notes": [
            "Latência end-to-end começa após a leitura do quadro; exposição da câmera e buffers internos do driver não são medidos.",
            "Tempos CUDA são sincronizados por etapa; inicialização, carga de pesos e aquecimento são excluídos.",
            "A suavização é causal para evitar acesso a quadros futuros em operação real-time.",
        ],
    }
    if (scenario.get("xai", {}) or {}).get("method") == "rgu_all_average":
        summary["notes"].append(
            "XAI executa os dez métodos do RGU_XAI_PAPER uma vez por imagem e inclui a fusão por média simples."
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(({key: row.get(key) for key in columns} for row in rows))


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": summary.get("scenario"),
        "architecture": summary.get("architecture"),
        "mode": summary.get("mode"),
        "status": summary.get("status"),
    }
    for section in ("counts", "rates", "throughput", "queue", "torch_gpu_memory"):
        for key, value in (summary.get(section, {}) or {}).items():
            row[f"{section}.{key}"] = value
    for stage, stats in (summary.get("latency_ms", {}) or {}).items():
        for statistic in ("mean", "std", "median", "p95", "max"):
            row[f"latency_ms.{stage}.{statistic}"] = stats.get(statistic)
    resources = summary.get("resources", {}) or {}
    for key, stats in resources.items():
        if isinstance(stats, dict):
            for statistic, value in stats.items():
                row[f"resources.{key}.{statistic}"] = value
    if summary.get("error"):
        row["error"] = summary["error"]
    return row


def _redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(config)
    output.pop("_config_path", None)
    source = output.get("source", {})
    raw_path = str(source.get("path", ""))
    if raw_path and output.get("privacy", {}).get("redact_source_path", True):
        source["path_sha256"] = hashlib.sha256(raw_path.encode("utf-8")).hexdigest()
        source["path"] = "<redacted>"
    return output


def _scenario_folder(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return safe.strip("_") or "scenario"


def _select_config(config: dict[str, Any], scenario_names: Optional[Sequence[str]]) -> dict[str, Any]:
    selected = copy.deepcopy(config)
    if scenario_names:
        wanted = set(scenario_names)
        all_scenarios = copy.deepcopy(config.get("scenarios", []))
        known = {str(item.get("name", "")) for item in all_scenarios}
        missing = sorted(wanted - known)
        _require(not missing, f"Cenário(s) inexistente(s): {missing}")
        scenarios = [item for item in all_scenarios if str(item.get("name")) in wanted]
        for item in scenarios:
            item["enabled"] = True
    else:
        scenarios = _enabled_scenarios(config)
    selected["scenarios"] = scenarios
    return selected


def inspect_benchmark_config(
    config: dict[str, Any], scenario_names: Optional[Sequence[str]] = None
) -> list[dict[str, Any]]:
    """Validate paths and return a non-sensitive dry-run inventory."""
    selected = _select_config(config, scenario_names)
    validate_benchmark_config(selected, check_files=True)
    inventory = []
    for scenario in _enabled_scenarios(selected):
        checkpoints = resolve_checkpoint_patterns([str(item) for item in scenario["checkpoints"]])
        inventory.append({
            "name": scenario["name"],
            "architecture": scenario["architecture"],
            "mode": scenario.get("mode", "single"),
            "checkpoint_count": len(checkpoints),
            "checkpoint_size_mb": sum(path.stat().st_size for path in checkpoints) / (1024.0 ** 2),
            "xai": (scenario.get("xai", {}) or {}).get("method", "none"),
            "smoothing_window": int(scenario.get("smoothing_window", 1)),
        })
    return inventory


def _run_one_scenario(
    scenario: dict[str, Any],
    config: dict[str, Any],
    scenario_dir: Path,
) -> dict[str, Any]:
    seed = int(config.get("seed", 1234))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device_from_config(config)
    LOGGER.info("Inicializando cenário %s em %s...", scenario["name"], device)
    pipeline = RealtimePipeline(scenario, config.get("detector", {}) or {}, device)
    environment = _environment(device, pipeline.detector)

    warmup_iterations = int(scenario.get("warmup_iterations", config.get("warmup_iterations", 10)))
    if warmup_iterations > 0:
        warmup_frame = read_warmup_frame(config["source"])
        LOGGER.info("Aquecendo %s por %d iterações...", scenario["name"], warmup_iterations)
        for index in range(warmup_iterations):
            now = time.perf_counter()
            pipeline.process(FrameItem(index=-index - 1, frame=warmup_frame, captured_at=now,
                                       enqueued_at=now, acquisition_ms=0.0))
        pipeline.reset_stream_state()
        del warmup_frame

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    source_config = config["source"]
    frame_queue: queue.Queue[Any] = queue.Queue(maxsize=int(source_config.get("queue_size", 4)))
    producer = FrameProducer(source_config, frame_queue)
    gpu_index = device.index or 0 if device.type == "cuda" else 0
    resources = ResourceSampler(float(config.get("resource_sample_interval_s", 0.1)), gpu_index)
    rows: list[dict[str, Any]] = []
    resources.start()
    producer.start()
    try:
        while True:
            item = frame_queue.get()
            if item is _END:
                break
            rows.append(pipeline.process(item))
    finally:
        producer.request_stop()
        producer.join(timeout=10.0)
        resources.stop()
    if producer.is_alive():
        raise BenchmarkError("A thread de aquisição não encerrou no tempo esperado.")
    if producer.error is not None:
        if isinstance(producer.error, BenchmarkError):
            raise producer.error
        raise BenchmarkError(f"Falha na aquisição: {producer.error}") from producer.error

    summary = summarize_run(scenario, rows, producer.stats, resources, pipeline, environment)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    public_rows = [
        {key: value for key, value in row.items() if key not in {"captured_at", "completed_at"}}
        for row in rows
    ]
    _write_csv(scenario_dir / "per_frame.csv", public_rows)
    _write_csv(scenario_dir / "resource_samples.csv", resources.rows)
    (scenario_dir / "summary.json").write_text(
        json.dumps(_json_value(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "%s concluído: %.2f FPS válidos, p95 E2E %.2f ms, cobertura %.1f%%.",
        scenario["name"],
        summary["throughput"]["valid_output_fps"],
        summary["latency_ms"]["end_to_end_ms"]["p95"],
        summary["rates"]["end_to_end_coverage_pct_of_read"],
    )
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _metric(summary: dict[str, Any], *keys: str) -> Any:
    current: Any = summary
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _write_markdown_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark real-time end-to-end",
        "",
        "| Cenário | Status | Mediana E2E (ms) | p95 E2E (ms) | FPS válidos | Cobertura (%) | Perda fila (%) | VRAM pico (MB) | GPU média (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {scenario} | {status} | {median} | {p95} | {fps} | {coverage} | {drop} | {vram} | {gpu} |".format(
                scenario=summary.get("scenario", "?"),
                status=summary.get("status", "error"),
                median=_format_number(_metric(summary, "latency_ms", "end_to_end_ms", "median")),
                p95=_format_number(_metric(summary, "latency_ms", "end_to_end_ms", "p95")),
                fps=_format_number(_metric(summary, "throughput", "valid_output_fps")),
                coverage=_format_number(_metric(summary, "rates", "end_to_end_coverage_pct_of_read"), 1),
                drop=_format_number(_metric(summary, "rates", "queue_drop_pct_of_read"), 1),
                vram=_format_number(_metric(summary, "torch_gpu_memory", "max_allocated_mb"), 1),
                gpu=_format_number(_metric(summary, "resources", "gpu_util_pct", "mean"), 1),
            )
        )
    lines.extend([
        "",
        "A afirmação de tempo real deve ser confrontada com a taxa de entrada configurada. "
        "Além do FPS, reporte p95 end-to-end, perdas de fila, cobertura de faces e cenário exato.",
        "",
        "Os CSVs por cenário não contêm nomes de arquivos ou identificadores de pacientes.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark_suite(
    config: dict[str, Any],
    *,
    scenario_names: Optional[Sequence[str]] = None,
    source_override: Optional[str] = None,
    output_override: Optional[str | Path] = None,
) -> Path:
    """Run selected scenarios and return the timestamped results directory."""
    selected = _select_config(config, scenario_names)
    if source_override is not None:
        selected["source"]["path"] = source_override
        if selected["source"].get("kind") == "camera":
            selected["source"]["camera_index"] = int(source_override)
    validate_benchmark_config(selected, check_files=True)
    base_output = _as_path(output_override or selected.get("output_dir", "outputs/realtime_benchmark"))
    run_dir = base_output / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(_redacted_config(selected), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    summaries: list[dict[str, Any]] = []
    fail_fast = bool(selected.get("fail_fast", False))
    for scenario in _enabled_scenarios(selected):
        scenario_dir = run_dir / _scenario_folder(str(scenario["name"]))
        try:
            summary = _run_one_scenario(scenario, selected, scenario_dir)
        except Exception as exc:
            LOGGER.exception("Cenário %s falhou.", scenario["name"])
            summary = {
                "scenario": str(scenario["name"]),
                "architecture": str(scenario.get("architecture", "")),
                "mode": str(scenario.get("mode", "")),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            scenario_dir.mkdir(parents=True, exist_ok=True)
            (scenario_dir / "summary.json").write_text(
                json.dumps(_json_value(summary), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if fail_fast:
                summaries.append(summary)
                break
        summaries.append(summary)
        _write_csv(run_dir / "summary.csv", [_flatten_summary(item) for item in summaries])
        _write_markdown_report(run_dir / "REPORT.md", summaries)

    (run_dir / "summary.json").write_text(
        json.dumps(_json_value(summaries), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(run_dir / "summary.csv", [_flatten_summary(item) for item in summaries])
    _write_markdown_report(run_dir / "REPORT.md", summaries)
    return run_dir
