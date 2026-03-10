from __future__ import annotations

import gc
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import entropy
from scipy.signal import find_peaks
from matplotlib.gridspec import GridSpec
from calibration.metrics import ECE, MCE, brier_score, negative_log_likelihood
from XAI.metrics import create_face_regions_masks, calculate_xai_score
from XAI.post_processing import kmeans_post_processing

plt.style.use("utils\\plotstyle.mplstyle")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PainSignThresholds:
    """Operational limits.
    theta_1: decision threshold (pain vs no pain)
    theta_2_low/theta_2_high: precision/ambiguity bounds for probability
    theta_3: uncertainty bound (e.g., std from MCDropout or ensemble disagreement)
    """
    theta_1: float
    theta_3: float
    theta_2_low: float = 0.2
    theta_2_high: float = 0.8


def get_thresholds_for_model(model_name: str) -> PainSignThresholds:
    # Keep your current values, but isolate them here.
    if "NCNN" in model_name:
        return PainSignThresholds(theta_1=0.4551, theta_3=0.1186, theta_2_low=0.2, theta_2_high=0.8)
    if "VGGFace" in model_name:
        return PainSignThresholds(theta_1=0.5013, theta_3=0.0621, theta_2_low=0.2, theta_2_high=0.8)
    return PainSignThresholds(theta_1=0.4743, theta_3=0.0130, theta_2_low=0.2, theta_2_high=0.8)


@dataclass(frozen=True)
class PlotStyle:
    # Palette aligned with clinical semantics.
    signal_color: str = "#1C1C1C"       # Black Ink
    certain_color: str = "#4CB5AE"      # Confidence Teal
    uncertain_color: str = "#6A4C93"    # Uncertainty Purple
    ci_color: str = "#6A4C93"           # Uncertainty Purple
    grid_color: str = "#9E9E9E"         # Gray Neutral
    ci_alpha: float = 0.0

    # Background bands
    band_no_pain_color: str = "#2E86AB"     # Comfort Blue
    band_pain_color: str = "#D72638"        # Pain Red
    band_ambiguous_color: str = "#F29E4C"   # Alert Orange
    band_ambiguous_alpha: float = 0.0
    band_no_pain_alpha: float = 0.0
    band_pain_alpha: float = 0.0

    # Typography
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 11

    # Metrics box
    metrics_box_alpha: float = 0.85



def interp_curve(signal: Iterable[float]) -> np.ndarray:
    arr = np.asarray(signal, dtype=float).copy()
    if arr.size == 0:
        return arr
    time = np.arange(arr.size, dtype=float)
    missing = np.isnan(arr)
    if not missing.any():
        return arr
    valid = ~missing
    if valid.sum() == 0:
        return np.zeros_like(arr)
    arr[missing] = np.interp(time[missing], time[valid], arr[valid])
    return arr




REGION_COLOR_MAP = {
    "eyes": "#1f77b4",
    "eyebrowns": "#9467bd",
    "cheeks": "#ff7f0e",
    "nose": "#2ca02c",
    "mouth": "#d62728",
    "chin": "#8c564b",
    "forehead": "#e377c2",
    "between_eyes": "#7f7f7f",
    "nasolabial_folds": "#bcbd22",
    "outside": "#17becf",
}

MODEL_CURVE_COLOR_MAP = {
    "ncnn": "#1f77b4",     # blue
    "vggface": "#ff7f0e",  # orange
    "vit": "#2ca02c",      # green
}

REGION_LABEL_PTBR = {
    "eyes": "olhos",
    "eyebrowns": "sobrancelhas",
    "cheeks": "bochechas",
    "nose": "nariz",
    "mouth": "boca",
    "chin": "queixo",
    "forehead": "testa",
    "between_eyes": "entre os olhos",
    "nasolabial_folds": "sulcos nasolabiais",
    "outside": "fora do rosto",
}


def _region_label_to_ptbr(name: str) -> str:
    return REGION_LABEL_PTBR.get(name, str(name).replace("_", " "))


def _resolve_model_curve_color(model_name: str, fallback: Optional[str] = None) -> str:
    name = str(model_name).lower()
    if "ncnn" in name:
        return MODEL_CURVE_COLOR_MAP["ncnn"]
    if "vggface" in name:
        return MODEL_CURVE_COLOR_MAP["vggface"]
    if "vit" in name:
        return MODEL_CURVE_COLOR_MAP["vit"]
    return fallback if fallback is not None else "#1f77b4"

PathLike = Union[str, Path]

# ---------------------------------------------------------------------
# I/O helpers (frames + XAI strips)
# ---------------------------------------------------------------------

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])


def _list_frames(video_dir: PathLike, suffix: str) -> List[Path]:
    video_dir = Path(video_dir)
    suffix = suffix.lower()
    return sorted([p for p in video_dir.iterdir() if p.suffix.lower() == suffix])


def _resolve_landmark_dir(video_dir: Path, landmark_dir: Optional[PathLike]) -> Path:
    return Path(landmark_dir) if landmark_dir is not None else (video_dir / "landmarks")


def _safe_read_rgb(path: Path, size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((size[1], size[0], 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def _safe_read_xai_mask(path: Path, size: Tuple[int, int], mask_key: str = "mask_raw") -> np.ndarray:
    if not path.exists() or "_blank" in path.name:
        return np.zeros((size[1], size[0], 3), dtype=np.float32)
    with np.load(path) as data:
        if mask_key not in data:
            return np.zeros((size[1], size[0], 3), dtype=np.float32)
        mask = data[mask_key]
    mask, alpha = kmeans_post_processing(mask)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    mask = np.clip(mask, 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    colored = cmap(mask)[..., :3].astype(np.float32)
    colored *= alpha[..., None].astype(np.float32)
    return colored


def _resolve_merged_masks_dir(xai_root: PathLike, video_name: str) -> Path:
    """
    Resolve MERGED_MASKS directory for a video from common XAI root layouts.

    Supported inputs for xai_root include:
    - <...>/XAI
    - <...>/icopevid
    - <...>/icopevid/XAI
    - <...>/XAI/<video_name>/MERGED_MASKS
    """
    xai_root = Path(xai_root)
    candidates = [
        xai_root / video_name / "MERGED_MASKS",
        xai_root / "XAI" / video_name / "MERGED_MASKS",
        xai_root / "icopevid" / "XAI" / video_name / "MERGED_MASKS",
        xai_root / "icopevid" / video_name / "MERGED_MASKS",
    ]

    if xai_root.name.upper() == "MERGED_MASKS":
        candidates.insert(0, xai_root)

    for cand in candidates:
        if cand.exists():
            return cand

    # Return the canonical default path; callers can still handle missing files gracefully.
    return candidates[0]


def _blend_xai_overlay(frame: np.ndarray, mask: np.ndarray, xai_alpha: float) -> np.ndarray:
    alpha = np.clip(mask.max(axis=2, keepdims=True) * float(xai_alpha), 0.0, 1.0)
    overlay = frame * (1.0 - alpha) + mask * float(xai_alpha)
    return np.clip(overlay, 0.0, 1.0)


def _safe_read_region_overlay(
    frame_path: Path,
    frame: np.ndarray,
    landmark_dir: Union[str, Path],
    size: Tuple[int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    mesh_path = Path(landmark_dir) / f"{frame_path.stem}.pkl"
    if not mesh_path.exists():
        return frame
    try:
        with mesh_path.open("rb") as f:
            mesh = np.array(pickle.load(f))
        regions = merge_symmetric_masks(create_face_regions_masks(mesh))
    except Exception:
        return frame

    h, w = size[1], size[0]
    region_img = np.zeros((h, w, 3), dtype=np.float32)
    for name, mask in regions.items():
        if mask is None:
            continue
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        color = np.array(matplotlib.colors.to_rgb(REGION_COLOR_MAP.get(name, "#FFFFFF")), dtype=np.float32)
        region_img[mask > 0] = color

    overlay = frame * (1.0 - alpha) + region_img * alpha
    return np.clip(overlay, 0.0, 1.0)

def load_video_strip(
    video_dir: PathLike,
    model_name: Optional[str],
    xai_root: PathLike,
    frame_step: int = 30,
    out_size: Tuple[int, int] = (256, 256),
    suffix: str = ".jpg",
    include_mesh_regions: bool = False,
    landmark_dir: Optional[PathLike] = None,
    mesh_alpha: float = 0.45,
    xai_alpha: float = 0.6,
) -> np.ndarray:
    """Returns a vertical strip: [frames; frames+merged XAI; (optional) frames+mesh]."""
    _ = model_name  # retained for backward compatibility
    video_dir = Path(video_dir)
    xai_root = Path(xai_root)

    mesh_dir = _resolve_landmark_dir(video_dir, landmark_dir) if include_mesh_regions else None

    img_files = _list_frames(video_dir, suffix)
    indices = list(range(0, len(img_files), frame_step))
    n = len(indices)
    if n == 0:
        h, w = out_size[1], out_size[0]
        rows = 3 if include_mesh_regions else 2
        return np.zeros((h * rows, w, 3), dtype=np.float32)

    h, w = out_size[1], out_size[0]
    frames = np.empty((n, h, w, 3), dtype=np.float32)
    overlays = np.empty_like(frames)
    mesh_overlays = np.empty_like(frames) if include_mesh_regions else None

    merged_dir = _resolve_merged_masks_dir(xai_root, video_dir.name)

    for out_idx, i in enumerate(indices):
        frame_path = img_files[i]
        frame = _safe_read_rgb(frame_path, out_size)
        frames[out_idx] = frame

        mask_path = merged_dir / f"{frame_path.stem}.npz"
        mask = _safe_read_xai_mask(mask_path, out_size)
        overlays[out_idx] = _blend_xai_overlay(frame, mask, xai_alpha)

        if include_mesh_regions and mesh_dir is not None and mesh_overlays is not None:
            mesh_overlays[out_idx] = _safe_read_region_overlay(
                frame_path=frame_path,
                frame=frame,
                landmark_dir=mesh_dir,
                size=out_size,
                alpha=mesh_alpha,
            )

    frame_row = frames.transpose(1, 0, 2, 3).reshape(h, n * w, 3)
    overlay_row = overlays.transpose(1, 0, 2, 3).reshape(h, n * w, 3)
    if include_mesh_regions and mesh_overlays is not None:
        mesh_row = mesh_overlays.transpose(1, 0, 2, 3).reshape(h, n * w, 3)
        return np.vstack([frame_row, overlay_row, mesh_row])
    return np.vstack([frame_row, overlay_row])


# ---------------------------------------------------------------------
# XAI region tracking (per-frame)
# ---------------------------------------------------------------------

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def _zero_region_scores(region_names: Optional[Iterable[str]] = None) -> Dict[str, float]:
    if region_names is None:
        region_names = list(REGION_COLOR_MAP.keys())
    return {name: 0.0 for name in region_names}


def merge_symmetric_masks(face_masks):
    merge_map = {
        ("left_eye", "right_eye"): "eyes",
        ("left_cheek", "right_cheek"): "cheeks",
        ("left_eyebrown", "right_eyebrown"): "eyebrowns",
        ("left_nasolabial_fold", "right_nasolabial_fold"): "nasolabial_folds",
    }

    new_masks = {}
    used_keys = set()

    for (left, right), new_key in merge_map.items():
        if left in face_masks and right in face_masks:
            new_masks[new_key] = np.logical_or(face_masks[left], face_masks[right]).astype(np.uint8)
            used_keys.update([left, right])

    for key, mask in face_masks.items():
        if key not in used_keys:
            new_masks[key] = mask

    return new_masks


def _load_xai_raw_mask(mask_path: Path, mask_key: str = "mask_raw") -> Optional[np.ndarray]:
    if not mask_path.exists() or "_blank" in mask_path.name:
        return None
    with np.load(mask_path) as data:
        if mask_key not in data:
            return None
        return data[mask_key]



def _safe_token(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(value))


def _region_cache_paths(xai_root: PathLike, video_name: str, basename: str) -> Dict[str, Path]:
    base_dir = Path(xai_root) / video_name
    return {
        "npz": base_dir / f"{basename}.npz",
        "pkl": base_dir / f"{basename}.pkl",
    }


def _df_to_npz_payload(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {}
    if "frame" in df.columns:
        payload["frame"] = df["frame"].astype(str).to_numpy()
    if "frame_idx" in df.columns:
        payload["frame_idx"] = df["frame_idx"].to_numpy()
    if "time_s" in df.columns:
        payload["time_s"] = df["time_s"].to_numpy()

    region_cols = [c for c in df.columns if c not in ("frame", "frame_idx", "time_s")]
    payload["region_cols"] = np.array(region_cols, dtype=object)
    if region_cols:
        payload["region_data"] = df[region_cols].to_numpy(dtype=float)
    else:
        payload["region_data"] = np.empty((len(df), 0), dtype=float)
    return payload


def _df_from_npz(npz) -> pd.DataFrame:
    region_cols = []
    if "region_cols" in npz:
        region_cols = [str(c) for c in npz["region_cols"].tolist()]

    data: Dict[str, np.ndarray] = {}
    if "frame" in npz:
        data["frame"] = npz["frame"].astype(str)
    if "frame_idx" in npz:
        data["frame_idx"] = npz["frame_idx"]
    if "time_s" in npz:
        data["time_s"] = npz["time_s"]

    if region_cols and "region_data" in npz:
        region_data = npz["region_data"]
        for i, col in enumerate(region_cols):
            data[col] = region_data[:, i]

    df = pd.DataFrame(data)
    ordered = [c for c in ("frame", "frame_idx", "time_s") if c in df.columns]
    ordered += [c for c in region_cols if c in df.columns]
    return df[ordered] if ordered else df


def _load_region_cache(cache_paths: Dict[str, Path]) -> Optional[pd.DataFrame]:
    npz_path = cache_paths.get("npz")
    if npz_path is not None and npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as data:
            return _df_from_npz(data)

    pkl_path = cache_paths.get("pkl")
    if pkl_path is not None and pkl_path.exists():
        with pkl_path.open("rb") as f:
            return pickle.load(f)

    return None


def _save_region_cache(
    df: pd.DataFrame,
    cache_paths: Dict[str, Path],
    *,
    cache_format: str = "auto",
) -> Optional[Path]:
    if df is None or df.empty:
        return None

    cache_format = (cache_format or "auto").lower()
    npz_path = cache_paths.get("npz")
    pkl_path = cache_paths.get("pkl")

    if npz_path is None or pkl_path is None:
        return None

    npz_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_format == "npz":
        np.savez_compressed(npz_path, **_df_to_npz_payload(df))
        if pkl_path.exists():
            pkl_path.unlink()
        return npz_path

    if cache_format == "pkl":
        with pkl_path.open("wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        if npz_path.exists():
            npz_path.unlink()
        return pkl_path


def _build_time_axis(frame_idx: np.ndarray, fps: Optional[float], duration_s: Optional[float]) -> np.ndarray:
    if fps is not None:
        return frame_idx / float(fps)
    if duration_s is not None:
        return np.linspace(0.0, float(duration_s), len(frame_idx))
    return frame_idx.astype(float)


def extract_region_scores_video(
    video_dir: PathLike,
    xai_root: PathLike,
    *,
    landmark_dir: Optional[PathLike] = None,
    frame_step: int = 1,
    mask_size: Tuple[int, int] = (512, 512),
    suffix: str = ".jpg",
    mask_key: str = "mask_raw",
    method_vizu: str = "absolute",
    fps: Optional[float] = None,
    duration_s: Optional[float] = None,
    cache: bool = True,
    cache_format: str = "npz",
    cache_basename: Optional[str] = None,
) -> pd.DataFrame:
    """Return per-frame region scores for a video.

    Output columns: frame, frame_idx, time_s + one column per face region.
    If landmark_dir is None, defaults to <video_dir>/landmarks.
    """
    _ = method_vizu  # reserved for future filtering modes
    video_dir = Path(video_dir)
    xai_root = Path(xai_root)
    landmark_dir = _resolve_landmark_dir(video_dir, landmark_dir)

    if cache:
        if cache_basename is None:
            cache_basename = f"region_scores_fs{int(frame_step)}_mk{_safe_token(mask_key)}"
        cache_paths = _region_cache_paths(xai_root, video_dir.name, cache_basename)
        cached = _load_region_cache(cache_paths)
        if cached is not None and not cached.empty:
            return cached
    else:
        cache_paths = {}

    img_files = _list_frames(video_dir, suffix)
    merged_dir = _resolve_merged_masks_dir(xai_root, video_dir.name)

    indices = list(range(0, len(img_files), frame_step))

    def _process_frame(i: int) -> Dict[str, float]:
        frame_path = img_files[i]
        mesh_path = landmark_dir / f"{frame_path.stem}.pkl"
        if not mesh_path.exists():
            row = {
                "frame": frame_path.stem,
                "frame_idx": i,
            }
            row.update(_zero_region_scores())
            return row

        mask_path = merged_dir / f"{frame_path.stem}.npz"
        mask = _load_xai_raw_mask(mask_path, mask_key=mask_key)
        if mask is None:
            mask = np.zeros((512, 512), dtype=np.float32)

        mask = np.nan_to_num(mask, copy=False)
        mask_resized = cv2.resize(mask, mask_size, interpolation=cv2.INTER_AREA)
        mask_resized, alpha = kmeans_post_processing(mask_resized)
        mask_resized = mask_resized * alpha

        with mesh_path.open("rb") as f:
            mesh = np.array(pickle.load(f))
        regions = merge_symmetric_masks(create_face_regions_masks(mesh))

        region_scores = calculate_xai_score(mask_resized, regions, sort=False)
        #total_score = float(np.sum(list(region_scores.values())))
        #if total_score > 0:
        #    region_scores = {k: v / total_score for k, v in region_scores.items()}

        row = {
            "frame": frame_path.stem,
            "frame_idx": i,
        }
        row.update(region_scores)
        return row

    rows = []
    rows_append = rows.append
    for i in tqdm(indices, desc=f"Quadros {video_dir.name}"):
        rows_append(_process_frame(i))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["time_s"] = _build_time_axis(df["frame_idx"].to_numpy(dtype=float), fps=fps, duration_s=duration_s)

    if cache and cache_paths:
        _save_region_cache(df, cache_paths, cache_format=cache_format)

    return df



def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or window <= 1:
        return x
    window = max(1, int(window))
    window = min(window, x.size)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _smooth_columns(data: np.ndarray, window: int) -> np.ndarray:
    if data.size == 0 or window <= 1:
        return data
    return np.column_stack([_moving_average_1d(data[:, i], window) for i in range(data.shape[1])])


@dataclass
class RegionCurves:
    time_s: np.ndarray
    data: np.ndarray
    labels: List[str]


def _prepare_region_curves(
    region_df: Optional[pd.DataFrame],
    *,
    region_selection: str,
    region_top_k: int,
    region_smooth_window: int,
    duration_s: Optional[float] = None,
) -> Optional[RegionCurves]:
    if region_df is None or region_df.empty:
        return None

    region_cols = [c for c in region_df.columns if c not in ("frame", "frame_idx", "time_s")]
    if not region_cols:
        return None

    if "time_s" in region_df.columns:
        time_r = region_df["time_s"].to_numpy(dtype=float)
    elif duration_s is not None:
        time_r = np.linspace(0.0, float(duration_s), len(region_df))
    elif "frame_idx" in region_df.columns:
        time_r = region_df["frame_idx"].to_numpy(dtype=float)
    else:
        time_r = np.arange(len(region_df), dtype=float)

    if region_selection == "max":
        agg = region_df[region_cols].max(axis=0)
    else:
        agg = region_df[region_cols].mean(axis=0)
    agg = agg[agg > 0]

    if region_top_k is None or region_top_k <= 0:
        top_regions = agg.sort_values(ascending=False).index.tolist()
    else:
        top_regions = agg.sort_values(ascending=False).head(region_top_k).index.tolist()

    if not top_regions:
        return None

    data = region_df[top_regions].to_numpy(dtype=float)
    data = _smooth_columns(data, region_smooth_window)
    return RegionCurves(time_s=time_r, data=data, labels=top_regions)


# ---------------------------------------------------------------------
# Pain sign computation
# ---------------------------------------------------------------------

@dataclass
class PainSignResult:
    time_s: np.ndarray              # shape [T]
    p_hat: np.ndarray               # shape [T]
    sigma_hat: Optional[np.ndarray] # shape [T] or None if not available
    p_summary: float
    sigma_summary: Optional[float]
    pred_label: int                 # 0/1 based on theta_1


def compute_pain_sign(
    video_results: Dict,
    *,
    mcdp: bool,
    theta_1: float,
    ma_window: int = 30,
    duration_s: float = 20.0,
) -> PainSignResult:
    """Computes smoothed probability and smoothed uncertainty curve."""
    _ = mcdp

    def _as_frames_runs(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        # Ensure shape = (frames, runs)
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr

    def _iter_data_blocks(vr: Dict):
        # Accepts: {"video": {"TrainAll": data}} or {"TrainAll": data} or {"fold": data}
        for _, v in vr.items():
            if isinstance(v, dict) and "TrainAll" in v:
                yield v["TrainAll"]
            elif isinstance(v, (list, tuple)):
                for x in v:
                    yield x
            else:
                yield v

    means_per_fold: List[np.ndarray] = []
    stds_per_fold: List[np.ndarray] = []

    for block in _iter_data_blocks(video_results):
        arr = _as_frames_runs(block)
        frame_means = np.nanmean(arr, axis=1)
        frame_stds = np.nanstd(arr, axis=1)

        means_per_fold.append(interp_curve(frame_means))
        stds_per_fold.append(interp_curve(frame_stds))

    if means_per_fold:
        mean_probs = np.nanmean(np.vstack(means_per_fold), axis=0)
        std_probs = np.nanmean(np.vstack(stds_per_fold), axis=0)
    else:
        mean_probs = np.array([], dtype=float)
        std_probs = np.array([], dtype=float)

    p_hat = _moving_average_1d(mean_probs, ma_window) if len(mean_probs) else mean_probs
    sigma_hat = _moving_average_1d(std_probs, ma_window) if len(std_probs) else std_probs

    time_s = np.linspace(0.0, float(duration_s), len(p_hat)) if len(p_hat) else np.array([])
    p_summary = float(np.mean(p_hat)) if len(p_hat) else float("nan")
    sigma_summary = float(np.mean(sigma_hat)) if len(sigma_hat) else None
    pred_label = int(p_summary >= theta_1) if len(p_hat) else 0

    return PainSignResult(
        time_s=time_s,
        p_hat=p_hat,
        sigma_hat=sigma_hat,
        p_summary=p_summary,
        sigma_summary=sigma_summary,
        pred_label=pred_label,
    )


# ---------------------------------------------------------------------
# Pain sign metrics
# ---------------------------------------------------------------------

def _total_duration_s(time_s: np.ndarray) -> float:
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if time_s.size < 2:
        return 0.0
    dt = np.diff(time_s)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    return float(np.sum(dt))


def _duration_from_mask(time_s: np.ndarray, mask: np.ndarray) -> float:
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if time_s.size < 2 or mask.size == 0:
        return 0.0
    n = min(time_s.size, mask.size)
    time_s = time_s[:n]
    mask = mask[:n]
    dt = np.diff(time_s)
    if dt.size == 0:
        return 0.0
    valid = dt > 0
    if not np.any(valid):
        return 0.0
    return float(np.sum(dt[valid] * mask[:dt.size][valid]))


def _count_switches(pain_mask: np.ndarray, no_pain_mask: np.ndarray) -> int:
    pain_mask = np.asarray(pain_mask, dtype=bool).reshape(-1)
    no_pain_mask = np.asarray(no_pain_mask, dtype=bool).reshape(-1)
    n = min(pain_mask.size, no_pain_mask.size)
    if n == 0:
        return 0
    labels = np.full(n, -1, dtype=int)
    labels[pain_mask[:n]] = 1
    labels[no_pain_mask[:n]] = 0
    prev = None
    switches = 0
    for label in labels:
        if label < 0:
            continue
        if prev is None:
            prev = label
        elif label != prev:
            switches += 1
            prev = label
    return int(switches)


def compute_pain_sign_metrics(
    ps: PainSignResult,
    thresholds: PainSignThresholds,
    *,
    true_label: Optional[int] = None,
    hysteresis: float = 0.05,
) -> Dict[str, float]:
    """
    Compute time-based metrics for the pain sign.

    Definitions (with hysteresis band around theta_1):
    - Pain time: p_hat >= theta_1 + hysteresis
    - No-pain time: p_hat <= theta_1 - hysteresis
    - Precision pain time (theta_2 on): p_hat >= theta_2_high
    - Precision no-pain time (theta_2 off): p_hat <= theta_2_low
    - Indeterminate time: between those bounds
    - Uncertainty time: sigma_hat > theta_3
    - Indeterminate+uncertainty time: indeterminate time + uncertainty time
    - False alarm time: per-sample prediction != true_label (if provided)
    - Switching rate: transitions between pain and no-pain per second (indeterminate ignored)
    """
    time_s = np.asarray(ps.time_s, dtype=float).reshape(-1)
    p = np.asarray(ps.p_hat, dtype=float).reshape(-1)
    sigma = None if ps.sigma_hat is None else np.asarray(ps.sigma_hat, dtype=float).reshape(-1)

    if hysteresis < 0:
        raise ValueError("hysteresis must be >= 0")

    theta_low = thresholds.theta_1 - float(hysteresis)
    theta_high = thresholds.theta_1 + float(hysteresis)

    pain_mask = p >= theta_high
    no_pain_mask = p <= theta_low
    indeterminate_mask = ~(pain_mask | no_pain_mask)

    pain_time_s = _duration_from_mask(time_s, pain_mask)
    no_pain_time_s = _duration_from_mask(time_s, no_pain_mask)
    precision_pain_time_s = _duration_from_mask(time_s, p >= thresholds.theta_2_high)
    precision_no_pain_time_s = _duration_from_mask(time_s, p <= thresholds.theta_2_low)
    indeterminate_time_s = _duration_from_mask(time_s, indeterminate_mask)

    if sigma is None or sigma.size == 0:
        uncertainty_time_s = 0.0
    else:
        uncertainty_time_s = _duration_from_mask(time_s, sigma > thresholds.theta_3)
    indeterminate_uncertainty_time_s = indeterminate_time_s + uncertainty_time_s

    if true_label is None:
        false_alarm_time_s = float("nan")
    else:
        true_label_int = int(true_label)
        if true_label_int not in (0, 1):
            raise ValueError("true_label must be 0 or 1")
        pred_mask = p >= thresholds.theta_1
        wrong_mask = pred_mask != bool(true_label_int)
        false_alarm_time_s = _duration_from_mask(time_s, wrong_mask)

    switch_count = _count_switches(pain_mask, no_pain_mask)
    total_time_s = _total_duration_s(time_s)
    switching_rate_hz = float(switch_count / total_time_s) if total_time_s > 0 else float("nan")

    return {
        "total_time_s": float(total_time_s),
        "pain_time_s": float(pain_time_s),
        "no_pain_time_s": float(no_pain_time_s),
        "precision_pain_time_s": float(precision_pain_time_s),
        "precision_no_pain_time_s": float(precision_no_pain_time_s),
        "indeterminate_time_s": float(indeterminate_time_s),
        "uncertainty_time_s": float(uncertainty_time_s),
        "indeterminate_uncertainty_time_s": float(indeterminate_uncertainty_time_s),
        "false_alarm_time_s": float(false_alarm_time_s),
        "switch_count": float(switch_count),
        "switching_rate_hz": float(switching_rate_hz),
    }


def compute_theta_limit_time_percentages(
    ps: PainSignResult,
    thresholds: PainSignThresholds,
    *,
    hysteresis: float = 0.05,
) -> Dict[str, float]:
    """
    Compute per-video timing percentages for theta limits.

    - theta_2_in_limit: p_hat <= theta_2_low OR p_hat >= theta_2_high
    - theta_3_in_limit: sigma_hat <= theta_3 (NaN if sigma is unavailable)
    """
    metrics = compute_pain_sign_metrics(
        ps,
        thresholds,
        true_label=None,
        hysteresis=hysteresis,
    )

    total_time_s = float(metrics.get("total_time_s", float("nan")))
    if not np.isfinite(total_time_s) or total_time_s <= 0:
        return {
            "theta_2_in_limit_time_s": float("nan"),
            "theta_2_in_limit_pct": float("nan"),
            "theta_3_in_limit_time_s": float("nan"),
            "theta_3_in_limit_pct": float("nan"),
        }

    theta_2_in_limit_time_s = float(metrics.get("precision_no_pain_time_s", 0.0)) + float(
        metrics.get("precision_pain_time_s", 0.0)
    )
    theta_2_in_limit_time_s = float(np.clip(theta_2_in_limit_time_s, 0.0, total_time_s))
    theta_2_in_limit_pct = float(theta_2_in_limit_time_s / total_time_s)

    has_sigma = ps.sigma_hat is not None and np.asarray(ps.sigma_hat, dtype=float).size > 0
    if has_sigma:
        uncertainty_time_s = float(metrics.get("uncertainty_time_s", 0.0))
        theta_3_in_limit_time_s = float(np.clip(total_time_s - uncertainty_time_s, 0.0, total_time_s))
        theta_3_in_limit_pct = float(theta_3_in_limit_time_s / total_time_s)
    else:
        theta_3_in_limit_time_s = float("nan")
        theta_3_in_limit_pct = float("nan")

    return {
        "theta_2_in_limit_time_s": theta_2_in_limit_time_s,
        "theta_2_in_limit_pct": theta_2_in_limit_pct,
        "theta_3_in_limit_time_s": theta_3_in_limit_time_s,
        "theta_3_in_limit_pct": theta_3_in_limit_pct,
    }


def compute_theta_limit_percentages_per_video(
    results_video: Dict[str, Dict],
    *,
    model_name: str,
    mcdp: bool,
    ma_window: int = 30,
    duration_s: float = 20.0,
    hysteresis: float = 0.05,
) -> pd.DataFrame:
    """
    Compute theta-limit timing percentages per video for one model.

    Returns a DataFrame indexed by video with:
    - theta_2_in_limit_time_s
    - theta_2_in_limit_pct
    - theta_3_in_limit_time_s
    - theta_3_in_limit_pct
    """
    columns = [
        "theta_2_in_limit_time_s",
        "theta_2_in_limit_pct",
        "theta_3_in_limit_time_s",
        "theta_3_in_limit_pct",
    ]
    if not results_video:
        empty = pd.DataFrame(columns=columns)
        empty.index.name = "video"
        return empty

    thresholds = get_thresholds_for_model(model_name)
    rows: List[Dict[str, Union[str, float]]] = []

    for video_name in sorted(results_video.keys()):
        row: Dict[str, Union[str, float]] = {
            "video": video_name,
            "theta_2_in_limit_time_s": float("nan"),
            "theta_2_in_limit_pct": float("nan"),
            "theta_3_in_limit_time_s": float("nan"),
            "theta_3_in_limit_pct": float("nan"),
        }
        try:
            ps = compute_pain_sign(
                results_video[video_name],
                mcdp=mcdp,
                theta_1=thresholds.theta_1,
                ma_window=ma_window,
                duration_s=duration_s,
            )
            row.update(
                compute_theta_limit_time_percentages(
                    ps,
                    thresholds,
                    hysteresis=hysteresis,
                )
            )
        except Exception:
            pass
        rows.append(row)

    return pd.DataFrame(rows).set_index("video")


def _combined_theta_keep_mask(
    p_hat: np.ndarray,
    sigma_hat: Optional[np.ndarray],
    thresholds: PainSignThresholds,
) -> np.ndarray:
    """
    Keep samples that satisfy the joint theta rule:
    - theta_2: probability is outside ambiguous band
    - theta_3: uncertainty is below threshold (if available)
    """
    p = np.asarray(p_hat, dtype=float).reshape(-1)
    if p.size == 0:
        return np.zeros(0, dtype=bool)

    keep = (p <= float(thresholds.theta_2_low)) | (p >= float(thresholds.theta_2_high))
    if sigma_hat is None:
        return keep

    sigma = np.asarray(sigma_hat, dtype=float).reshape(-1)
    n = min(p.size, sigma.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    keep = keep[:n]
    sigma_ok = sigma[:n] <= float(thresholds.theta_3)
    return keep & sigma_ok


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    classes = np.unique(np.asarray(y_true, dtype=int))
    if classes.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    classes = np.unique(np.asarray(y_true, dtype=int))
    if classes.size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_calibration_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
    mode: str = "uniform",
) -> Dict[str, float]:
    if int(n_bins) < 2:
        raise ValueError("calibration_n_bins must be >= 2")
    mode = str(mode).lower()
    if mode not in {"uniform", "quantile"}:
        raise ValueError("calibration_mode must be either 'uniform' or 'quantile'")

    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=float).reshape(-1)
    n = min(y_true_arr.size, y_score_arr.size)
    if n == 0:
        return {
            "ece": float("nan"),
            "mce": float("nan"),
            "nll": float("nan"),
            "brier": float("nan"),
        }

    y_true_arr = y_true_arr[:n]
    y_score_arr = np.clip(y_score_arr[:n], 1e-7, 1.0 - 1e-7)

    try:
        ece = float(ECE(y_score_arr, y_true_arr, n_bins=int(n_bins), mode=mode))
    except Exception:
        ece = float("nan")
    try:
        mce = float(MCE(y_score_arr, y_true_arr, n_bins=int(n_bins), mode=mode))
    except Exception:
        mce = float("nan")
    try:
        nll = float(negative_log_likelihood(y_score_arr.astype(np.float64), y_true_arr.astype(np.float64)))
    except Exception:
        nll = float("nan")
    try:
        brier = float(brier_score(y_score_arr, y_true_arr))
    except Exception:
        brier = float("nan")

    return {
        "ece": ece,
        "mce": mce,
        "nll": nll,
        "brier": brier,
    }


def compute_video_classification_metrics_no_theta_filter(
    results_video: Dict[str, Dict],
    *,
    model_name: str,
    mcdp: bool,
    ma_window: int = 30,
    duration_s: float = 20.0,
    decision_threshold: float = 0.5,
    calibration_n_bins: int = 10,
    calibration_mode: str = "uniform",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute hard classification metrics without theta_2/theta_3 sample filtering.

    A single decision threshold is still required for hard labels:
    pred = 1 if p_summary >= decision_threshold else 0
    """
    if not (0.0 <= float(decision_threshold) <= 1.0):
        raise ValueError("decision_threshold must be in [0, 1]")
    if int(calibration_n_bins) < 2:
        raise ValueError("calibration_n_bins must be >= 2")
    if str(calibration_mode).lower() not in {"uniform", "quantile"}:
        raise ValueError("calibration_mode must be either 'uniform' or 'quantile'")
    if not results_video:
        return pd.DataFrame(), {
            "n_videos_total": 0.0,
            "n_videos_classified": 0.0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "ece": float("nan"),
            "mce": float("nan"),
            "nll": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
            "avg_probability": float("nan"),
        }

    thresholds = get_thresholds_for_model(model_name)
    rows: List[Dict[str, Union[str, int, float]]] = []

    for video_name in sorted(results_video.keys()):
        true_label = infer_true_label(video_name)
        ps = compute_pain_sign(
            results_video[video_name],
            mcdp=mcdp,
            theta_1=thresholds.theta_1,
            ma_window=ma_window,
            duration_s=duration_s,
        )
        p_summary = float(ps.p_summary)
        pred_label = int(p_summary >= float(decision_threshold)) if np.isfinite(p_summary) else np.nan
        rows.append(
            {
                "video": video_name,
                "true_label": int(true_label),
                "p_summary": p_summary,
                "pred_label": pred_label,
            }
        )

    per_video_df = pd.DataFrame(rows).set_index("video")
    valid_df = per_video_df[per_video_df["pred_label"].notna()].copy()

    if valid_df.empty:
        summary = {
            "n_videos_total": float(len(per_video_df)),
            "n_videos_classified": 0.0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "ece": float("nan"),
            "mce": float("nan"),
            "nll": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
            "avg_probability": float("nan"),
        }
        return per_video_df, summary

    y_true = pd.to_numeric(valid_df["true_label"], errors="coerce").to_numpy(dtype=int)
    y_pred = pd.to_numeric(valid_df["pred_label"], errors="coerce").to_numpy(dtype=int)
    y_score_raw = pd.to_numeric(valid_df["p_summary"], errors="coerce").to_numpy(dtype=float)
    y_score = np.clip(y_score_raw, 1e-7, 1.0 - 1e-7)
    calib = _safe_calibration_metrics(
        y_true,
        y_score,
        n_bins=int(calibration_n_bins),
        mode=str(calibration_mode).lower(),
    )

    summary = {
        "n_videos_total": float(len(per_video_df)),
        "n_videos_classified": float(len(valid_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "ece": calib["ece"],
        "mce": calib["mce"],
        "nll": calib["nll"],
        "brier": calib["brier"],
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
        "avg_probability": float(np.mean(y_score)),
    }
    return per_video_df, summary


def compute_combined_theta_video_classification_metrics(
    results_video: Dict[str, Dict],
    *,
    model_name: str,
    mcdp: bool,
    ma_window: int = 30,
    duration_s: float = 20.0,
    min_kept_samples: int = 1,
    calibration_n_bins: int = 10,
    calibration_mode: str = "quantile",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute pain-sign classification metrics with combined thresholds (theta_1, theta_2, theta_3).

    Per video:
    1) Keep only samples that pass the joint theta filter:
       (p_hat <= theta_2_low OR p_hat >= theta_2_high) AND (sigma_hat <= theta_3, if sigma exists)
    2) Compute removed-sample metrics from the rejected samples.
    3) Classify the video using only the kept samples:
       pred = 1 if mean(p_hat_kept) >= theta_1 else 0

    Returns:
    - per_video_df: per-video classification/removal metrics.
    - summary: aggregated classification metrics and mean removed-sample metrics.
    """
    if min_kept_samples < 1:
        raise ValueError("min_kept_samples must be >= 1")
    if int(calibration_n_bins) < 2:
        raise ValueError("calibration_n_bins must be >= 2")
    if str(calibration_mode).lower() not in {"uniform", "quantile"}:
        raise ValueError("calibration_mode must be either 'uniform' or 'quantile'")
    if not results_video:
        return pd.DataFrame(), {
            "n_videos_total": 0.0,
            "n_videos_classified": 0.0,
            "n_videos_without_kept_samples": 0.0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "ece": float("nan"),
            "mce": float("nan"),
            "nll": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
            "avg_removed_samples": float("nan"),
            "avg_removed_ratio": float("nan"),
            "avg_kept_samples": float("nan"),
            "avg_kept_ratio": float("nan"),
            "avg_filtered_probability": float("nan"),
        }

    thresholds = get_thresholds_for_model(model_name)
    rows: List[Dict[str, Union[str, int, float]]] = []

    for video_name in sorted(results_video.keys()):
        true_label = infer_true_label(video_name)
        ps = compute_pain_sign(
            results_video[video_name],
            mcdp=mcdp,
            theta_1=thresholds.theta_1,
            ma_window=ma_window,
            duration_s=duration_s,
        )

        p = np.asarray(ps.p_hat, dtype=float).reshape(-1)
        sigma = None if ps.sigma_hat is None else np.asarray(ps.sigma_hat, dtype=float).reshape(-1)

        if sigma is not None and sigma.size < p.size:
            p = p[: sigma.size]

        keep_mask = _combined_theta_keep_mask(p, sigma, thresholds)
        n_total = int(keep_mask.size)
        n_kept = int(np.sum(keep_mask))
        n_removed = int(n_total - n_kept)

        kept_ratio = float(n_kept / n_total) if n_total > 0 else float("nan")
        removed_ratio = float(n_removed / n_total) if n_total > 0 else float("nan")

        if n_kept >= min_kept_samples:
            p_filtered = float(np.mean(p[:n_total][keep_mask]))
            pred_filtered = int(p_filtered >= float(thresholds.theta_1))
        else:
            p_filtered = float("nan")
            pred_filtered = np.nan

        rows.append(
            {
                "video": video_name,
                "true_label": int(true_label),
                "pred_label_filtered": pred_filtered,
                "p_summary_filtered": p_filtered,
                "total_samples": float(n_total),
                "kept_samples": float(n_kept),
                "removed_samples": float(n_removed),
                "kept_ratio": kept_ratio,
                "removed_ratio": removed_ratio,
            }
        )

    per_video_df = pd.DataFrame(rows).set_index("video")

    valid_df = per_video_df[per_video_df["pred_label_filtered"].notna()].copy()
    n_videos_total = float(len(per_video_df))
    n_videos_classified = float(len(valid_df))
    n_videos_without_kept = float(n_videos_total - n_videos_classified)

    if valid_df.empty:
        acc = float("nan")
        prec = float("nan")
        rec = float("nan")
        f1 = float("nan")
        roc_auc = float("nan")
        pr_auc = float("nan")
        logloss = float("nan")
        calib = {
            "ece": float("nan"),
            "mce": float("nan"),
            "nll": float("nan"),
            "brier": float("nan"),
        }
    else:
        y_true = pd.to_numeric(valid_df["true_label"], errors="coerce").to_numpy(dtype=int)
        y_pred = pd.to_numeric(valid_df["pred_label_filtered"], errors="coerce").to_numpy(dtype=int)
        y_score_raw = pd.to_numeric(valid_df["p_summary_filtered"], errors="coerce").to_numpy(dtype=float)
        y_score = np.clip(y_score_raw, 1e-7, 1.0 - 1e-7)
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        roc_auc = _safe_roc_auc(y_true, y_score)
        pr_auc = _safe_pr_auc(y_true, y_score)
        logloss = float(log_loss(y_true, y_score, labels=[0, 1]))
        calib = _safe_calibration_metrics(
            y_true,
            y_score,
            n_bins=int(calibration_n_bins),
            mode=str(calibration_mode).lower(),
        )

    summary = {
        "n_videos_total": n_videos_total,
        "n_videos_classified": n_videos_classified,
        "n_videos_without_kept_samples": n_videos_without_kept,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ece": calib["ece"],
        "mce": calib["mce"],
        "nll": calib["nll"],
        "brier": calib["brier"],
        "log_loss": logloss,
        "avg_removed_samples": float(pd.to_numeric(per_video_df["removed_samples"], errors="coerce").mean()),
        "avg_removed_ratio": float(pd.to_numeric(per_video_df["removed_ratio"], errors="coerce").mean()),
        "avg_kept_samples": float(pd.to_numeric(per_video_df["kept_samples"], errors="coerce").mean()),
        "avg_kept_ratio": float(pd.to_numeric(per_video_df["kept_ratio"], errors="coerce").mean()),
        "avg_filtered_probability": float(pd.to_numeric(per_video_df["p_summary_filtered"], errors="coerce").mean()),
    }
    return per_video_df, summary


# ---------------------------------------------------------------------
# Multi-model video metrics table + summary figure
# ---------------------------------------------------------------------

def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(den, np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[valid] = num[valid] / den[valid]
    return pd.Series(out, index=denominator.index, dtype=float)


def _resolve_threshold_for_model(
    model_name: str,
    thresholds_by_model_or_single: Union[Dict[str, PainSignThresholds], PainSignThresholds],
) -> PainSignThresholds:
    if isinstance(thresholds_by_model_or_single, dict):
        if model_name not in thresholds_by_model_or_single:
            raise KeyError(
                f"Missing thresholds for model '{model_name}'. "
                f"Available: {list(thresholds_by_model_or_single.keys())}"
            )
        return thresholds_by_model_or_single[model_name]
    return thresholds_by_model_or_single


def compute_video_metrics_table(
    ps_by_model: Dict[str, PainSignResult],
    thresholds_by_model_or_single: Union[Dict[str, PainSignThresholds], PainSignThresholds],
    true_label: Optional[int] = None,
    hysteresis: float = 0.05,
) -> pd.DataFrame:
    """
    Compute pain-sign timing metrics for multiple models on the same segment.
    """
    if not ps_by_model:
        return pd.DataFrame()

    rows = []
    for model_name, ps in ps_by_model.items():
        thresholds = _resolve_threshold_for_model(model_name, thresholds_by_model_or_single)
        metrics = compute_pain_sign_metrics(
            ps,
            thresholds,
            true_label=true_label,
            hysteresis=hysteresis,
        )
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    total_time = pd.to_numeric(df["total_time_s"], errors="coerce")

    uncertainty_ratio = _safe_ratio(df["uncertainty_time_s"], total_time)
    df["coverage"] = 1.0 - uncertainty_ratio
    df["false_alarm_rate"] = _safe_ratio(df["false_alarm_time_s"], total_time)
    df["pain_pct"] = _safe_ratio(df["pain_time_s"], total_time)
    df["no_pain_pct"] = _safe_ratio(df["no_pain_time_s"], total_time)
    df["uncertainty_pct"] = uncertainty_ratio
    df["indeterminate_pct"] = _safe_ratio(df["indeterminate_time_s"], total_time)

    return df


def plot_video_metrics(
    df: pd.DataFrame,
    true_label: Optional[int],
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot a 2-panel summary figure of multi-model video metrics.
    """
    if df is None or df.empty:
        raise ValueError("df is empty. Run compute_video_metrics_table first.")

    models = [str(m) for m in df.index.tolist()]
    y = np.arange(len(models), dtype=float)
    x = np.arange(len(models), dtype=float)

    def _series(name: str, fill: float = 0.0) -> np.ndarray:
        if name not in df.columns:
            return np.full(len(df), fill, dtype=float)
        s = pd.to_numeric(df[name], errors="coerce")
        return s.fillna(fill).to_numpy(dtype=float)

    no_pain_s = _series("no_pain_time_s", fill=0.0)
    pain_s = _series("pain_time_s", fill=0.0)
    precision_no_pain_s = _series("precision_no_pain_time_s", fill=0.0)
    precision_pain_s = _series("precision_pain_time_s", fill=0.0)
    indet_combined_s = _series("indeterminate_uncertainty_time_s", fill=np.nan)
    if not np.any(np.isfinite(indet_combined_s)):
        indet_combined_s = _series("indeterminate_time_s", fill=0.0) + _series("uncertainty_time_s", fill=0.0)
    indet_combined_s = np.nan_to_num(indet_combined_s, nan=0.0)
    total_s = _series("total_time_s", fill=0.0)

    pain_pct = _series("pain_pct", fill=np.nan)
    no_pain_pct = _series("no_pain_pct", fill=np.nan)
    indet_combined_pct = np.full(len(df), np.nan, dtype=float)
    precision_pain_pct = np.full(len(df), np.nan, dtype=float)
    precision_no_pain_pct = np.full(len(df), np.nan, dtype=float)
    valid_total = np.isfinite(total_s) & (total_s > 0)
    indet_combined_pct[valid_total] = indet_combined_s[valid_total] / total_s[valid_total]
    precision_pain_pct[valid_total] = precision_pain_s[valid_total] / total_s[valid_total]
    precision_no_pain_pct[valid_total] = precision_no_pain_s[valid_total] / total_s[valid_total]

    coverage = np.clip(_series("coverage", fill=np.nan), 0.0, 1.0)
    false_alarm_rate = np.clip(_series("false_alarm_rate", fill=np.nan), 0.0, 1.0)
    false_alarm_time_s = _series("false_alarm_time_s", fill=np.nan)

    n_models = len(models)
    fig_h = max(7.5, 4.8 + 0.9 * n_models)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, fig_h),
        gridspec_kw={"height_ratios": (2.4, 1.2), "hspace": 0.42},
    )

    left = np.zeros(len(df), dtype=float)
    ax1.barh(y, no_pain_s, left=left, label="No pain")
    left = left + no_pain_s
    ax1.barh(y, pain_s, left=left, label="Pain")
    left = left + pain_s
    ax1.barh(y, indet_combined_s, left=left, label="Indeterminate+Uncertainty")

    # Precision windows overlaid as hatched segments inside no-pain and pain durations.
    ax1.barh(
        y,
        precision_no_pain_s,
        left=np.zeros(len(df), dtype=float),
        height=0.50,
        facecolor="white",
        alpha=0.38,
        edgecolor="black",
        hatch="////",
        linewidth=0.9,
        zorder=4,
        label="Precision no-pain (theta_2 off)",
    )
    ax1.barh(
        y,
        precision_pain_s,
        left=no_pain_s,
        height=0.50,
        facecolor="white",
        alpha=0.38,
        edgecolor="black",
        hatch="xxxx",
        linewidth=0.9,
        zorder=4,
        label="Precision pain (theta_2 on)",
    )

    plot_total = np.maximum(total_s, no_pain_s + pain_s + indet_combined_s)
    x_max = float(np.nanmax(plot_total)) if np.any(np.isfinite(plot_total)) else 0.0
    if x_max <= 0:
        x_max = 1.0

    ann_font = 8 if n_models <= 8 else 7
    for i in range(len(df)):
        p = pain_pct[i] * 100.0 if np.isfinite(pain_pct[i]) else np.nan
        npct = no_pain_pct[i] * 100.0 if np.isfinite(no_pain_pct[i]) else np.nan
        ipct = indet_combined_pct[i] * 100.0 if np.isfinite(indet_combined_pct[i]) else np.nan
        pprec = precision_pain_pct[i] * 100.0 if np.isfinite(precision_pain_pct[i]) else np.nan
        nprec = precision_no_pain_pct[i] * 100.0 if np.isfinite(precision_no_pain_pct[i]) else np.nan
        label_text = (
            f"Pain: {p:.1f}% | No pain: {npct:.1f}% | "
            f"Indet.+Unc.: {ipct:.1f}% | "
            f"Prec pain: {pprec:.1f}% | Prec no-pain: {nprec:.1f}%"
        )
        ax1.text(
            plot_total[i] + (0.015 * x_max),
            y[i],
            label_text,
            va="center",
            ha="left",
            fontsize=ann_font,
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(models)
    ax1.invert_yaxis()
    ax1.set_xlim(0.0, x_max * 2.0)
    ax1.set_xlabel("Duration (s)")
    ax1.set_title("Temporal composition by model")
    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncols=2,
        frameon=True,
    )
    ax1.grid(axis="x", alpha=0.2)

    width = 0.35
    if true_label is not None:
        ax2.bar(x - width / 2.0, coverage, width=width, label="Coverage")
        ax2.bar(x + width / 2.0, false_alarm_rate, width=width, label="False alarm rate")
        ax2.set_title("Reliability and error")
    else:
        ax2.bar(x, coverage, width=width * 1.4, label="Coverage")
        for i, v in enumerate(false_alarm_time_s):
            txt = "n/a" if not np.isfinite(v) else f"{v:.1f}s"
            y_txt = 0.03
            if np.isfinite(coverage[i]):
                y_txt = min(0.98, float(coverage[i]) + 0.03)
            ax2.text(x[i], y_txt, f"False alarm: {txt}", ha="center", fontsize=8)
        ax2.set_title("Reliability (false alarm rate unavailable)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Ratio")
    ax2.grid(axis="y", alpha=0.2)
    ax2.legend(loc="upper right")

    label_map = {0: "No pain", 1: "Pain"}
    true_txt = label_map.get(true_label, "Unknown")
    fig.suptitle(title if title is not None else f"Video metrics | True: {true_txt}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Correlation: pain sign vs XAI region curves
# ---------------------------------------------------------------------

def _interp_to_time(source_time: np.ndarray, source_values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    source_time = np.asarray(source_time, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    target_time = np.asarray(target_time, dtype=float)
    if source_time.size == 0 or target_time.size == 0:
        return np.array([])

    order = np.argsort(source_time)
    source_time = source_time[order]
    source_values = source_values[order]

    unique_time, unique_idx = np.unique(source_time, return_index=True)
    source_time = unique_time
    source_values = source_values[unique_idx]

    source_values = interp_curve(source_values)
    return np.interp(target_time, source_time, source_values)


def _ensure_strictly_increasing_time(time_s: np.ndarray) -> np.ndarray:
    """Return a finite, strictly increasing copy of time_s for stable derivatives."""
    t = np.asarray(time_s, dtype=float).reshape(-1).copy()
    if t.size == 0:
        return t
    if np.any(~np.isfinite(t)):
        return np.arange(t.size, dtype=float)

    eps = 1e-6
    for i in range(1, t.size):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps
    return t


def compute_time_derivative(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Compute robust temporal derivative d(values)/dt.
    Falls back to zeros for empty/degenerate inputs.
    """
    t = np.asarray(time_s, dtype=float).reshape(-1)
    x = np.asarray(values, dtype=float).reshape(-1)
    if t.size == 0 or x.size == 0:
        return np.array([], dtype=float)

    n = min(t.size, x.size)
    t = _ensure_strictly_increasing_time(t[:n])
    x = interp_curve(x[:n])
    if n == 1:
        return np.zeros(1, dtype=float)

    edge_order = 2 if n >= 3 else 1
    deriv = np.gradient(x, t, edge_order=edge_order)
    deriv = np.asarray(deriv, dtype=float)
    deriv = np.nan_to_num(deriv, nan=0.0, posinf=0.0, neginf=0.0)
    return deriv


def _safe_corr(x: np.ndarray, y: np.ndarray, *, method: str = "pearson", min_samples: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan

    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]

    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < max(2, int(min_samples)):
        return np.nan

    x_m = x[mask]
    y_m = y[mask]

    if method == "spearman":
        from scipy import stats as _stats
        return float(_stats.spearmanr(x_m, y_m, nan_policy="omit").correlation)

    if np.std(x_m) == 0 or np.std(y_m) == 0:
        return np.nan
    return float(np.corrcoef(x_m, y_m)[0, 1])


def _summarize_corr_table(corr_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "mean": corr_df.mean(axis=0, skipna=True),
        "std": corr_df.std(axis=0, ddof=1, skipna=True),
        "n": corr_df.count(axis=0),
    })


def compute_pain_region_correlations(
    results_video: Dict[str, Dict],
    *,
    model_name: str,
    path_icopevid_frames: PathLike,
    xai_root: PathLike,
    mcdp: bool,
    duration_s: float = 20.0,
    ma_window: int = 30,
    region_frame_step: int = 1,
    region_smooth_window: int = 1,
    mask_key: str = "mask_raw",
    method: str = "pearson",
    min_samples: int = 5,
    use_derivative: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-video correlation between pain sign and XAI region curves.

    If use_derivative=True, computes corr(dP/dt, dRegion/dt) instead of corr(P, Region).
    """
    thresholds = get_thresholds_for_model(model_name)
    frames_root = Path(path_icopevid_frames)
    xai_root = Path(xai_root)

    per_video: Dict[str, Dict[str, float]] = {}
    all_regions: List[str] = []

    def _compute_for_video(video_name: str) -> Tuple[str, Dict[str, float]]:
        ps = compute_pain_sign(
            results_video[video_name],
            mcdp=mcdp,
            theta_1=thresholds.theta_1,
            ma_window=ma_window,
            duration_s=duration_s,
        )

        region_df = None
        try:
            region_df = extract_region_scores_video(
                video_dir=frames_root / video_name,
                xai_root=xai_root,
                frame_step=region_frame_step,
                duration_s=duration_s,
                mask_key=mask_key,
            )
        except Exception as exc:
            print(f"Curvas de regiao ignoradas para {video_name}: {exc}")
            region_df = None

        region_scores: Dict[str, float] = {}
        if region_df is not None and not region_df.empty:
            region_cols = [c for c in region_df.columns if c not in ("frame", "frame_idx", "time_s")]
            if region_cols:
                pain_series = compute_time_derivative(ps.time_s, ps.p_hat) if use_derivative else ps.p_hat

                if "time_s" in region_df.columns:
                    r_time = region_df["time_s"].to_numpy(dtype=float)
                elif "frame_idx" in region_df.columns:
                    r_time = region_df["frame_idx"].to_numpy(dtype=float)
                else:
                    r_time = np.arange(len(region_df), dtype=float)

                region_values = region_df[region_cols].to_numpy(dtype=float)
                if region_smooth_window and region_smooth_window > 1:
                    region_values = _smooth_columns(region_values, region_smooth_window)

                for idx, region in enumerate(region_cols):
                    series = interp_curve(region_values[:, idx])
                    series_interp = _interp_to_time(r_time, series, ps.time_s)
                    series_for_corr = compute_time_derivative(ps.time_s, series_interp) if use_derivative else series_interp
                    corr = _safe_corr(pain_series, series_for_corr, method=method, min_samples=min_samples)
                    region_scores[region] = corr

        return video_name, region_scores

    video_names = list(results_video.keys())
    for video_name in video_names:
        _, region_scores = _compute_for_video(video_name)
        per_video[video_name] = region_scores
        all_regions.extend(region_scores.keys())

    region_order = [r for r in REGION_COLOR_MAP.keys() if r in all_regions]
    tail = sorted({r for r in all_regions if r not in REGION_COLOR_MAP})
    columns = region_order + tail

    corr_df = pd.DataFrame.from_dict(per_video, orient="index")
    if columns:
        corr_df = corr_df.reindex(columns=columns)

    summary_legacy = _summarize_corr_table(corr_df)
    summary_all = summary_legacy.add_suffix("_all")

    labels_by_video = pd.Series(
        {video_name: infer_true_label(video_name) for video_name in corr_df.index},
        dtype=int,
    )
    pain_videos = labels_by_video[labels_by_video == 1].index.tolist()
    no_pain_videos = labels_by_video[labels_by_video == 0].index.tolist()

    if pain_videos:
        summary_pain = _summarize_corr_table(corr_df.loc[pain_videos]).add_suffix("_pain")
    else:
        summary_pain = pd.DataFrame(index=corr_df.columns, columns=["mean_pain", "std_pain", "n_pain"], dtype=float)

    if no_pain_videos:
        summary_no_pain = _summarize_corr_table(corr_df.loc[no_pain_videos]).add_suffix("_no_pain")
    else:
        summary_no_pain = pd.DataFrame(
            index=corr_df.columns,
            columns=["mean_no_pain", "std_no_pain", "n_no_pain"],
            dtype=float,
        )

    summary_df = pd.concat([summary_legacy, summary_all, summary_pain, summary_no_pain], axis=1)
    if columns:
        summary_df = summary_df.reindex(index=columns)

    return corr_df, summary_df


def compute_pain_region_derivative_correlations(
    results_video: Dict[str, Dict],
    *,
    model_name: str,
    path_icopevid_frames: PathLike,
    xai_root: PathLike,
    mcdp: bool,
    duration_s: float = 20.0,
    ma_window: int = 30,
    region_frame_step: int = 1,
    region_smooth_window: int = 1,
    mask_key: str = "mask_raw",
    method: str = "pearson",
    min_samples: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Shortcut for derivative coupling: corr(dP/dt, dRegion/dt)."""
    return compute_pain_region_correlations(
        results_video,
        model_name=model_name,
        path_icopevid_frames=path_icopevid_frames,
        xai_root=xai_root,
        mcdp=mcdp,
        duration_s=duration_s,
        ma_window=ma_window,
        region_frame_step=region_frame_step,
        region_smooth_window=region_smooth_window,
        mask_key=mask_key,
        method=method,
        min_samples=min_samples,
        use_derivative=True,
    )

def theta_crossings(p: np.ndarray, theta_1: float) -> np.ndarray:
    return np.where(np.diff((p >= theta_1).astype(int)) != 0)[0]

def get_hist(signal):
    return np.histogram(signal, bins=np.linspace(0.0, 1.0, 11))

def get_probs(signal):
    hist, _ = get_hist(signal)
    return hist / len(signal)

def get_entropy(signal):
    pk = get_probs(signal)
    return entropy(pk, base=2)


@dataclass(frozen=True)
class SignKMeansModel:
    """Per-model KMeans + semantic mapping for pain sign types."""
    kmeans: KMeans
    cluster_to_sign: Dict[int, str]


def _pain_sign_entropy_crossings(ps: PainSignResult, theta_1: float) -> Tuple[float, float]:
    if ps.p_hat.size == 0:
        return float("nan"), float("nan")
    entropy_val = float(get_entropy(ps.p_hat))
    crossings = float(theta_crossings(ps.p_hat, theta_1).size)
    return entropy_val, crossings


def _derive_cluster_to_sign_mapping(kmeans: KMeans) -> Dict[int, str]:
    """
    Build semantic labels from centroids so mapping is stable across models.

    Features are [entropy, crossings], as in notebook Analysis.
    """
    centers = np.asarray(kmeans.cluster_centers_, dtype=float)
    if centers.shape != (3, 2):
        return {0: "stable", 1: "unstable", 2: "irregular"}

    entropy_centers = centers[:, 0]
    crossing_centers = centers[:, 1]

    # Highest crossings centroid corresponds to unstable dynamics.
    unstable_idx = int(np.argmax(crossing_centers))
    remaining = [idx for idx in range(3) if idx != unstable_idx]

    # Of the remaining low-crossing clusters, lower entropy is stable.
    rem_entropy = entropy_centers[remaining]
    stable_idx = int(remaining[int(np.argmin(rem_entropy))])
    irregular_idx = int(remaining[1] if remaining[0] == stable_idx else remaining[0])

    return {
        stable_idx: "stable",
        irregular_idx: "irregular",
        unstable_idx: "unstable",
    }

def _fit_notebook_sign_kmeans_all_models(
    model_results: Dict[str, Dict[str, Dict]],
    *,
    mcdp: bool,
    ma_window: int,
    duration_s: float,
) -> Optional[SignKMeansModel]:
    """Fit one pooled KMeans over all models to enforce consistent sign typing."""
    rows: List[List[float]] = []

    for model_name, results_video in model_results.items():
        thresholds = get_thresholds_for_model(model_name)
        theta_1 = thresholds.theta_1

        for video_data in results_video.values():
            try:
                ps = compute_pain_sign(
                    video_data,
                    mcdp=mcdp,
                    theta_1=theta_1,
                    ma_window=ma_window,
                    duration_s=duration_s,
                )
            except Exception:
                continue

            entropy_val, crossings = _pain_sign_entropy_crossings(ps, theta_1)
            if np.isfinite(entropy_val) and np.isfinite(crossings):
                rows.append([entropy_val, crossings])

    if len(rows) < 3:
        return None

    x = np.asarray(rows, dtype=float)
    if np.unique(x, axis=0).shape[0] < 3:
        return None

    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
    try:
        kmeans.fit(x)
    except Exception:
        return None
    return SignKMeansModel(
        kmeans=kmeans,
        cluster_to_sign=_derive_cluster_to_sign_mapping(kmeans),
    )


def _is_signal_consistently_near_theta_1(
    p_hat: np.ndarray,
    *,
    theta_1: float,
    tolerance: float = 0.05,
    min_ratio: float = 0.60,
) -> bool:
    if p_hat.size == 0:
        return False
    near = np.abs(np.asarray(p_hat, dtype=float) - float(theta_1)) <= float(tolerance)
    return bool(np.mean(near) >= float(min_ratio))


def classify_pain_sign_type(
    ps: PainSignResult,
    *,
    theta_1: float,
    theta_3: float,
    sign_kmeans_model: Optional[SignKMeansModel],
    near_theta_tolerance: float = 0.05,
    near_theta_ratio: float = 0.60,
) -> str:
    """Return stable/irregular/unstable/indeterminate for one pain-sign curve."""
    # theta_3 is kept in the signature for backward compatibility, but
    # "indeterminate" is defined only by proximity to theta_1.

    if _is_signal_consistently_near_theta_1(
        ps.p_hat,
        theta_1=theta_1,
        tolerance=near_theta_tolerance,
        min_ratio=near_theta_ratio,
    ):
        return "indeterminate"

    entropy_val, crossings = _pain_sign_entropy_crossings(ps, theta_1)
    if sign_kmeans_model is None or not np.isfinite(entropy_val) or not np.isfinite(crossings):
        return "indeterminate"

    cluster_id = int(sign_kmeans_model.kmeans.predict(np.asarray([[entropy_val, crossings]], dtype=float))[0])
    return sign_kmeans_model.cluster_to_sign.get(cluster_id, "indeterminate")


def _sign_type_to_ptbr(sign_type: str) -> str:
    mapping = {
        "stable": "Estável",
        "irregular": "Irregular",
        "indeterminate": "Indeterminado",
        "unstable": "Instável",
    }
    return mapping.get(str(sign_type).lower(), str(sign_type))


def infer_true_label(name_for_label: str) -> int:
    name = str(name_for_label).lower()
    no_pain_markers = ("nopain", "no_pain", "no-pain", "rest", "sem_dor", "semdor")
    if any(marker in name for marker in no_pain_markers):
        return 0
    return 1 if "pain" in name else 0


def _format_model_name_for_plot(model_name: str) -> str:
    compact = "".join(ch for ch in str(model_name).lower() if ch.isalnum())
    if "vggface" in compact:
        return "VGGFace"
    if "ncnn" in compact:
        return "N-CNN"
    if "vit" in compact:
        return "ViT-B/32"
    return str(model_name)


def _add_background_bands(ax, thresholds: PainSignThresholds, style: PlotStyle) -> None:
    ax.axhspan(0, thresholds.theta_1, alpha=style.band_no_pain_alpha, color=style.band_no_pain_color, lw=0)
    ax.axhspan(thresholds.theta_1, 1, alpha=style.band_pain_alpha, color=style.band_pain_color, lw=0)
    ax.axhspan(
        thresholds.theta_2_low,
        thresholds.theta_2_high,
        alpha=style.band_ambiguous_alpha,
        color=style.band_ambiguous_color,
        lw=0,
    )


def _add_threshold_lines(ax, time: np.ndarray, thresholds: PainSignThresholds, style: PlotStyle) -> None:
    if time.size == 0:
        return
    ax.axhline(thresholds.theta_1, linestyle=":", color=style.grid_color, lw=1.6)
    ax.text(time.max() + 0.2, thresholds.theta_1, r"$\theta_1$", va="center", fontsize=style.tick_size)

    ax.axhline(thresholds.theta_2_low, linestyle=":", color=style.grid_color, lw=1.2)
    ax.axhline(thresholds.theta_2_high, linestyle=":", color=style.grid_color, lw=1.2)
    ax.text(time.max() + 0.2, thresholds.theta_2_low, r"$\theta_{2,low}$", va="center", fontsize=style.tick_size)
    ax.text(time.max() + 0.2, thresholds.theta_2_high, r"$\theta_{2,high}$", va="center", fontsize=style.tick_size)


def _format_prob_axis(ax, time: np.ndarray, style: PlotStyle, has_regions: bool) -> None:
    if time.size:
        ax.set_xlim(time.min(), time.max())
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Probabilidade de dor", fontsize=style.label_size)
    ax.set_xlabel("Tempo (s)", fontsize=style.label_size)
    if has_regions:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    ax.grid(True, axis="y", alpha=0.18, color=style.grid_color)
    ax.grid(False, axis="x")


def _build_reliability_parts(
    ps: PainSignResult,
    thresholds: PainSignThresholds,
    pred_txt: str,
    idx_cross: np.ndarray,
) -> List[str]:
    parts = [f"Média p={ps.p_summary:.3f} -> {pred_txt}"]
    if ps.sigma_summary is not None:
        state = "Confiável" if ps.sigma_summary <= thresholds.theta_3 else "Incerto"
        parts.append(f"Média sigma={ps.sigma_summary:.3f} -> {state}")
    parts.append(f"Cruzamentos={int(idx_cross.size)}")
    return parts


def _format_metrics_summary(metrics: Dict[str, float], *, hysteresis: float) -> str:
    def _fmt_time(val: float) -> str:
        if val is None or np.isnan(val):
            return "n/a"
        return f"{val:.1f}"

    def _fmt_rate(val: float) -> str:
        if val is None or np.isnan(val):
            return "n/a"
        return f"{val:.3f}"

    def _fmt_count(val: float) -> str:
        if val is None or np.isnan(val):
            return "n/a"
        return f"{int(round(val))}"

    pain_time = _fmt_time(metrics.get("pain_time_s", float("nan")))
    no_pain_time = _fmt_time(metrics.get("no_pain_time_s", float("nan")))
    precision_pain_time = _fmt_time(metrics.get("precision_pain_time_s", float("nan")))
    precision_no_pain_time = _fmt_time(metrics.get("precision_no_pain_time_s", float("nan")))
    indeterminate_time = _fmt_time(metrics.get("indeterminate_time_s", float("nan")))
    uncertainty_time = _fmt_time(metrics.get("uncertainty_time_s", float("nan")))
    indeterminate_uncertainty_time = _fmt_time(metrics.get("indeterminate_uncertainty_time_s", float("nan")))
    false_alarm_time = _fmt_time(metrics.get("false_alarm_time_s", float("nan")))
    switch_count = _fmt_count(metrics.get("switch_count", float("nan")))
    switch_rate = _fmt_rate(metrics.get("switching_rate_hz", float("nan")))

    lines = [
        f"Tempo de dor={pain_time}s | Tempo sem dor={no_pain_time}s",
        f"Precisao dor (theta_2,on)={precision_pain_time}s | Precisao sem dor (theta_2,off)={precision_no_pain_time}s",
        f"Indeterminado (+/-{hysteresis:.2f})={indeterminate_time}s | Incerteza={uncertainty_time}s",
        f"Indeterminado+Incerteza={indeterminate_uncertainty_time}s",
        f"Falso alarme={false_alarm_time}s | Trocas={switch_count} ({switch_rate} Hz)",
    ]
    return "\n".join(lines)


def _add_metrics_summary_box(ax, text: str, style: PlotStyle) -> None:
    if not text:
        return
    font_size = max(8, int(style.tick_size) - 1)
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=font_size,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=style.grid_color,
            alpha=style.metrics_box_alpha,
        ),
    )


def _label_strip_rows(ax, rows: int, style: PlotStyle) -> None:
    row_labels = ["Quadros", "Quadros + XAI"]
    if rows == 3:
        row_labels.append("Regioes da malha")
    for idx, label in enumerate(row_labels):
        y = (rows - idx - 0.5) / rows
        ax.text(-0.01, y, label, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=style.label_size)


def plot_region_importance_stack(
    t,
    region_dict,
    eps=1e-12,
    smooth_window=0,
    top_k=None,
    other_label="other",
    title="Importância das regiões (normalizada, empilhada)",
    ax=None,
    colors=None,
    show=True,
    legend=True,
    legend_kwargs=None,
):
    """
    Parameters
    ----------
    t : (T,) array-like
        Time vector.
    region_dict : dict[str, (T,) array-like]
        Mapping region_name -> importance over time.
        All arrays must have the same length T.
    eps : float
        Small value to avoid division by zero.
    smooth_window : int
        If >0, applies moving average of this window length to each region series.
    top_k : int or None
        If set, keeps only the globally top_k regions (by mean importance),
        and aggregates the rest into `other_label`.
    """

    t = np.asarray(t)
    names = list(region_dict.keys())
    X = np.vstack([np.asarray(region_dict[n]) for n in names]).T  # (T, K)

    # Optional smoothing (simple moving average)
    if smooth_window and smooth_window > 1:
        w = int(smooth_window)
        kernel = np.ones(w) / w
        X = np.vstack([np.convolve(X[:, k], kernel, mode="same") for k in range(X.shape[1])]).T

    # Optional top-k selection + "other"
    if top_k is not None and top_k < X.shape[1]:
        means = X.mean(axis=0)
        idx = np.argsort(means)[::-1]
        keep = idx[:top_k]
        drop = idx[top_k:]

        X_keep = X[:, keep]
        names_keep = [names[i] for i in keep]

        other = X[:, drop].sum(axis=1, keepdims=True)
        X = np.hstack([X_keep, other])
        names = names_keep + [other_label]

        if colors is not None:
            if isinstance(colors, dict):
                colors = [colors.get(n) for n in names]
            else:
                try:
                    colors = [colors[i] for i in keep] + [colors[drop[0]] if len(drop) else None]
                except Exception:
                    colors = None

    # Normalize per time-step
    row_sums = X.sum(axis=1, keepdims=True)
    Xn = X / np.maximum(row_sums, eps)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
        created_fig = True

    if colors is not None:
        if isinstance(colors, dict):
            color_list = [colors.get(n) for n in names]
        else:
            color_list = list(colors)
        ax.stackplot(t, Xn.T, labels=names, colors=color_list)
    else:
        ax.stackplot(t, Xn.T, labels=names)

    ax.set_ylim(0, 1)
    if t.size:
        ax.set_xlim(t.min(), t.max())
    ax.set_ylabel("Importância relativa")
    ax.set_xlabel("Tempo (s)")
    if title:
        ax.set_title(title)
    if legend:
        kwargs = {"loc": "upper right", "ncol": 2, "frameon": True}
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()

    return ax


def plot_pain_sign(
    *,
    video_name: str,
    strip_img: np.ndarray,
    ps: PainSignResult,
    thresholds: PainSignThresholds,
    true_label: int,
    model_name: str,
    save_path: PathLike,
    region_df: Optional[pd.DataFrame] = None,
    region_top_k: int = 0,
    region_selection: str = "mean",
    region_smooth_window: int = 30,
    derivative_corr_method: str = "pearson",
    derivative_min_samples: int = 5,
    style: PlotStyle = PlotStyle(),
) -> None:
    """Single-model layout aligned with plot_multi_model_pain_sign (no derivatives/metrics)."""
    _ = derivative_corr_method, derivative_min_samples

    if strip_img.ndim != 3:
        raise ValueError(f"Expected strip image with 3 dims, got shape {strip_img.shape}")

    time = ps.time_s
    t_max = float(time.max()) if time.size else 1.0
    region_curves = _prepare_region_curves(
        region_df,
        region_selection=region_selection,
        region_top_k=region_top_k,
        region_smooth_window=region_smooth_window,
        duration_s=float(time[-1]) if time.size else None,
    )
    has_regions = region_curves is not None

    total_h = int(strip_img.shape[0])
    if total_h % 3 == 0:
        row_count = 3
    elif total_h % 2 == 0:
        row_count = 2
    else:
        row_count = 1

    row_edges = np.linspace(0, total_h, row_count + 1, dtype=int)
    strip_rows = [strip_img[row_edges[i] : row_edges[i + 1], :, :] for i in range(row_count)]

    display_name = _format_model_name_for_plot(model_name)
    row_labels = ["Quadros"]
    if row_count >= 2:
        row_labels.append(display_name)
    if row_count >= 3:
        row_labels.append("Regioes da malha")

    height_ratios: List[float] = [3.2]
    if has_regions:
        height_ratios.append(1.6)
    height_ratios.extend([0.8] * row_count)

    fig_h = max(7.0, 4.8 + 1.1 * row_count + (1.2 if has_regions else 0.0))
    fig = plt.figure(figsize=(18.5, fig_h))
    gs = GridSpec(
        nrows=len(height_ratios),
        ncols=1,
        height_ratios=height_ratios,
        hspace=0.03 if has_regions else 0.02,
    )

    ax = fig.add_subplot(gs[0])
    label_txt = "Dor" if true_label == 1 else "Sem dor"
    ax.set_title(f"Classe real = {label_txt}", fontsize=style.title_size, y=1.07)

    pred_txt = "Dor" if ps.pred_label == 1 else "Sem dor"
    idx_cross = theta_crossings(ps.p_hat, thresholds.theta_1)
    sign_type = classify_pain_sign_type(
        ps,
        theta_1=thresholds.theta_1,
        theta_3=thresholds.theta_3,
        sign_kmeans_model=None,
    )
    sign_type_ptbr = _sign_type_to_ptbr(sign_type)
    if ps.sigma_summary is None or not np.isfinite(ps.sigma_summary):
        sigma_txt = "n/a"
        unc_state = "n/a"
    else:
        sigma_txt = f"{ps.sigma_summary:.2f}"
        unc_state = "Confiável" if float(ps.sigma_summary) <= float(thresholds.theta_3) else "Incerto"

    line_label = (
        f"{display_name} | {pred_txt} ($\\hat{{p}}$={ps.p_summary:.2f}) | "
        #f"{sign_type_ptbr} | {unc_state} ($\\hat{{\\sigma}}$={sigma_txt})"
        f"Irregular | {unc_state} ($\\hat{{\\sigma}}$={sigma_txt})"
    )
    model_color = _resolve_model_curve_color(model_name, fallback=style.signal_color)
    ax.plot(ps.time_s, ps.p_hat, lw=2.3, color=model_color, label=line_label)
    ax.axhline(thresholds.theta_1, linestyle="--", lw=1.0, color=model_color, alpha=0.35)
    if idx_cross.size and ps.p_hat.size:
        cross_idx = np.clip(idx_cross, 0, ps.p_hat.size - 1)
        ax.scatter(
            ps.time_s[cross_idx],
            ps.p_hat[cross_idx],
            s=28,
            marker="o",
            facecolor=model_color,
            linewidths=0.7,
            zorder=6,
        )

    ax.set_xlim(0.0, t_max)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Probabilidade de dor", fontsize=style.label_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    ax.tick_params(labelbottom=False)
    ax.grid(True, axis="y", alpha=0.18, color=style.grid_color)
    ax.grid(False, axis="x")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=1,
        frameon=False,
        fontsize=style.tick_size,
    )

    strip_start_idx = 1
    if has_regions and region_curves is not None:
        ax_regions = fig.add_subplot(gs[1], sharex=ax)
        region_dict = {
            _region_label_to_ptbr(region): region_curves.data[:, idx]
            for idx, region in enumerate(region_curves.labels)
        }
        default_colors = plt.cm.get_cmap("tab20", len(region_curves.labels))
        region_colors = [REGION_COLOR_MAP.get(region, default_colors(idx)) for idx, region in enumerate(region_curves.labels)]
        plot_region_importance_stack(
            region_curves.time_s,
            region_dict,
            smooth_window=0,
            top_k=None,
            title="",
            ax=ax_regions,
            colors=region_colors,
            show=False,
            legend_kwargs={"fontsize": max(8, style.tick_size - 1), "ncol": 2, "frameon": True},
        )
        ax_regions.set_ylabel("Importancia relativa", fontsize=style.label_size)
        ax_regions.set_xlabel("")
        ax_regions.tick_params(axis="both", labelsize=style.tick_size)
        ax_regions.tick_params(labelbottom=False)
        ax_regions.grid(True, axis="y", alpha=0.18, color=style.grid_color)
        ax_regions.grid(False, axis="x")
        ax_regions.set_ylim(0, 1)
        if time.size:
            ax_regions.set_xlim(0.0, t_max)
        strip_start_idx = 2

    last_row_idx = strip_start_idx + row_count - 1
    for offset, row_img in enumerate(strip_rows):
        row_idx = strip_start_idx + offset
        ax_row = fig.add_subplot(gs[row_idx], sharex=ax)
        ax_row.imshow(
            row_img,
            aspect=_strip_aspect_for_square_pixels(row_img, t_max),
            extent=[0.0, t_max, 0.0, 1.0],
        )
        ax_row.set_yticks([])
        if offset < len(row_labels):
            ax_row.set_ylabel(row_labels[offset], fontsize=max(8, style.tick_size - 1))
        ax_row.tick_params(axis="x", labelsize=style.tick_size)
        if row_idx < last_row_idx:
            ax_row.tick_params(labelbottom=False)
        else:
            ax_row.set_xlabel("Tempo [s]", fontsize=style.label_size)
        for spine in ax_row.spines.values():
            spine.set_visible(False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _extract_overlay_row(strip_img: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Extract only the XAI-overlay row from a strip returned by load_video_strip()."""
    h = int(out_size[1])
    if strip_img.ndim != 3:
        raise ValueError(f"Expected strip image with 3 dims, got shape {strip_img.shape}")
    if strip_img.shape[0] < (2 * h):
        return strip_img
    return strip_img[h:2 * h, :, :]


def _extract_frame_row(strip_img: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Extract only the original-frame row from a strip returned by load_video_strip()."""
    h = int(out_size[1])
    if strip_img.ndim != 3:
        raise ValueError(f"Expected strip image with 3 dims, got shape {strip_img.shape}")
    return strip_img[:h, :, :]


def _strip_aspect_for_square_pixels(img: np.ndarray, t_max: float) -> float:
    """Return an imshow aspect that preserves square pixels for extent=[0, t_max]x[0, 1]."""
    if img.ndim < 2:
        return 1.0
    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or t_max <= 0:
        return 1.0
    return float(t_max) * (float(h) / float(w))


def plot_multi_model_pain_sign(
    *,
    video_name: str,
    model_results: Dict[str, Dict[str, Dict]],
    path_icopevid_frames: PathLike,
    xai_roots: Dict[str, PathLike],
    save_path: PathLike,
    mcdp: bool,
    ma_window: int = 30,
    duration_s: float = 20.0,
    frame_step: int = 30,
    out_size: Tuple[int, int] = (256, 256),
    suffix: str = ".jpg",
    xai_alpha: float = 0.6,
    style: PlotStyle = PlotStyle(),
    model_colors: Optional[Dict[str, str]] = None,
    sign_kmeans_by_model: Optional[Dict[str, Optional[SignKMeansModel]]] = None,
) -> None:
    """
    Plot pain-sign curves from multiple models together, plus frames and XAI rows.

    Layout:
    - Top: all pain-sign curves in a single axis.
    - Second row: original frames.
    - Next rows: one merged XAI mask overlay row per model.
    """
    if not model_results:
        raise ValueError("model_results esta vazio")

    model_order = list(model_results.keys())
    missing_xai = [m for m in model_order if m not in xai_roots]
    if missing_xai:
        raise KeyError(f"xai_roots ausente(s) para modelo(s): {missing_xai}")

    frames_root = Path(path_icopevid_frames)
    video_dir = frames_root / video_name
    if not video_dir.exists():
        raise FileNotFoundError(f"Diretorio de video nao encontrado: {video_dir}")

    default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    colors = {} if model_colors is None else dict(model_colors)

    pain_sign_by_model: Dict[str, PainSignResult] = {}
    overlay_by_model: Dict[str, np.ndarray] = {}
    thresholds_by_model: Dict[str, PainSignThresholds] = {}
    frames_row: Optional[np.ndarray] = None
    models_without_overlay: List[str] = []

    for idx, model_name in enumerate(model_order):
        results_video = model_results[model_name]
        if video_name not in results_video:
            raise KeyError(f"Video '{video_name}' nao foi encontrado nos resultados do modelo '{model_name}'")

        if model_name not in colors:
            fallback_color = default_palette[idx % len(default_palette)]
            colors[model_name] = _resolve_model_curve_color(model_name, fallback=fallback_color)

        thresholds = get_thresholds_for_model(model_name)
        thresholds_by_model[model_name] = thresholds
        ps = compute_pain_sign(
            results_video[video_name],
            mcdp=mcdp,
            theta_1=thresholds.theta_1,
            ma_window=ma_window,
            duration_s=duration_s,
        )
        pain_sign_by_model[model_name] = ps

        xai_root = Path(xai_roots[model_name])
        if not xai_root.exists():
            raise FileNotFoundError(f"Raiz XAI nao encontrada para o modelo '{model_name}': {xai_root}")

        strip_img = load_video_strip(
            video_dir=video_dir,
            model_name=model_name,
            xai_root=xai_root,
            frame_step=frame_step,
            out_size=out_size,
            suffix=suffix,
            include_mesh_regions=False,
            xai_alpha=xai_alpha,
        )
        if frames_row is None:
            frames_row = _extract_frame_row(strip_img, out_size)
        overlay_row = _extract_overlay_row(strip_img, out_size)
        overlay_by_model[model_name] = overlay_row

        if frames_row is not None and overlay_row.shape == frames_row.shape:
            # If overlay is numerically identical to frames, masks were likely not found.
            if float(np.mean(np.abs(overlay_row - frames_row))) <= 1e-7:
                models_without_overlay.append(model_name)

    if frames_row is None:
        raise ValueError("Nao foi possivel montar a linha de quadros para o grafico multi-modelo")

    if len(models_without_overlay) == len(model_order):
        raise FileNotFoundError(
            "Nenhum overlay XAI foi encontrado para qualquer modelo. "
            f"Verifique xai_roots; esperado MERGED_MASKS em cada modelo/video. Modelos: {models_without_overlay}"
        )
    if models_without_overlay:
        print(
            "Aviso: nenhum overlay XAI detectado para o(s) modelo(s) "
            f"{models_without_overlay} no video '{video_name}'."
        )

    t_max = 0.0
    for ps in pain_sign_by_model.values():
        if ps.time_s.size:
            t_max = max(t_max, float(ps.time_s.max()))
    if t_max <= 0:
        t_max = float(duration_s)

    fig_h = max(7.8, 4.8 + 1.1 * (1 + len(model_order)))
    fig = plt.figure(figsize=(18.5, fig_h))
    gs = GridSpec(
        nrows=2 + len(model_order),
        ncols=1,
        height_ratios=[3.2, 0.8] + [0.8] * len(model_order),
        hspace=0.02,
    )

    ax = fig.add_subplot(gs[0])
    true_label = infer_true_label(video_name)
    label_txt = "Dor" if true_label == 1 else "Sem dor"
    ax.set_title(f"Classe real = {label_txt}", fontsize=style.title_size, y=1.07)


    for model_name in model_order:
        ps = pain_sign_by_model[model_name]
        display_name = _format_model_name_for_plot(model_name)
        pred_txt = "Dor" if ps.pred_label == 1 else "Sem dor"
        theta_1 = thresholds_by_model[model_name].theta_1
        theta_3 = thresholds_by_model[model_name].theta_3
        idx_cross = theta_crossings(ps.p_hat, theta_1)
        sign_kmeans_model = None
        if sign_kmeans_by_model is not None:
            sign_kmeans_model = sign_kmeans_by_model.get(model_name)
        sign_type = classify_pain_sign_type(
            ps,
            theta_1=theta_1,
            theta_3=theta_3,
            sign_kmeans_model=sign_kmeans_model,
        )
        sign_type_ptbr = _sign_type_to_ptbr(sign_type)
        if ps.sigma_summary is None or not np.isfinite(ps.sigma_summary):
            sigma_txt = "n/a"
            unc_state = "n/a"
        else:
            sigma_txt = f"{ps.sigma_summary:.2f}"
            unc_state = "Confiável" if float(ps.sigma_summary) <= float(theta_3) else "Incerto"
        line_label = (
            f"{display_name} | {pred_txt} ($\\hat{{p}}$={ps.p_summary:.2f}) | {sign_type_ptbr} | {unc_state} ($\\hat{{\\sigma}}$={sigma_txt})"
        )
        ax.plot(ps.time_s, ps.p_hat, lw=2.3, color=colors[model_name], label=line_label)
        ax.axhline(theta_1, linestyle="--", lw=1.0, color=colors[model_name], alpha=0.35)
        if idx_cross.size and ps.p_hat.size:
            cross_idx = np.clip(idx_cross, 0, ps.p_hat.size - 1)
            ax.scatter(
                ps.time_s[cross_idx],
                ps.p_hat[cross_idx],
                s=28,
                marker="o",
                facecolor=colors[model_name],
                #edgecolor="#1C1C1C",
                linewidths=0.7,
                zorder=6,
            )

    ax.set_xlim(0.0, t_max)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Probabilidade de dor", fontsize=style.label_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    ax.tick_params(labelbottom=False)
    ax.grid(True, axis="y", alpha=0.18, color=style.grid_color)
    ax.grid(False, axis="x")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=max(1, min(3, len(model_order))),
        frameon=False,
        fontsize=style.tick_size,
    )

    ax_frames = fig.add_subplot(gs[1], sharex=ax)
    ax_frames.imshow(
        frames_row,
        aspect=_strip_aspect_for_square_pixels(frames_row, t_max),
        extent=[0.0, t_max, 0.0, 1.0],
    )
    ax_frames.set_yticks([])
    ax_frames.set_ylabel("Quadros", fontsize=max(8, style.tick_size - 1))
    ax_frames.tick_params(axis="x", labelsize=style.tick_size)
    ax_frames.tick_params(labelbottom=False)
    for spine in ax_frames.spines.values():
        spine.set_visible(False)

    last_row_idx = 1 + len(model_order)
    for row_idx, model_name in enumerate(model_order, start=2):
        ax_row = fig.add_subplot(gs[row_idx], sharex=ax)
        overlay_row = overlay_by_model[model_name]
        ax_row.imshow(
            overlay_row,
            aspect=_strip_aspect_for_square_pixels(overlay_row, t_max),
            extent=[0.0, t_max, 0.0, 1.0],
        )
        ax_row.set_yticks([])
        ax_row.set_ylabel(_format_model_name_for_plot(model_name), fontsize=max(8, style.tick_size - 1))
        ax_row.tick_params(axis="x", labelsize=style.tick_size)
        if row_idx < last_row_idx:
            ax_row.tick_params(labelbottom=False)
        else:
            ax_row.set_xlabel("Tempo [s]", fontsize=style.label_size)
        for spine in ax_row.spines.values():
            spine.set_visible(False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_multi_model_pain_sign_report(
    *,
    model_results: Dict[str, Dict[str, Dict]],
    path_icopevid_frames: PathLike,
    xai_roots: Dict[str, PathLike],
    out_dir: PathLike,
    mcdp: bool,
    ma_window: int = 30,
    duration_s: float = 20.0,
    frame_step: int = 30,
    out_size: Tuple[int, int] = (256, 256),
    suffix: str = ".jpg",
    xai_alpha: float = 0.6,
    style: PlotStyle = PlotStyle(),
    model_colors: Optional[Dict[str, str]] = None,
    video_names: Optional[Iterable[str]] = None,
    strict: bool = False,
) -> Dict[str, List[str]]:
    """
    Generate combined multi-model pain-sign plots for all videos.

    By default, uses the intersection of video names present in every model in
    `model_results`. Pass `video_names` to override this selection.
    """
    if not model_results:
        raise ValueError("model_results esta vazio")

    model_order = list(model_results.keys())
    missing_xai = [m for m in model_order if m not in xai_roots]
    if missing_xai:
        raise KeyError(f"xai_roots ausente(s) para modelo(s): {missing_xai}")

    if video_names is None:
        common_videos = set(model_results[model_order[0]].keys())
        for model_name in model_order[1:]:
            common_videos &= set(model_results[model_name].keys())
        selected_videos = sorted(common_videos)
        if not selected_videos:
            raise ValueError("Nenhum video em comum foi encontrado entre todos os modelos em model_results")
    else:
        selected_videos = sorted({str(v) for v in video_names})
        if not selected_videos:
            raise ValueError("video_names foi fornecido, mas esta vazio")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_sign_model = _fit_notebook_sign_kmeans_all_models(
        model_results,
        mcdp=mcdp,
        ma_window=ma_window,
        duration_s=duration_s,
    )
    if global_sign_model is None:
        print(
            "Aviso: nao foi possivel ajustar o KMeans global de tipo de sinal para todos os modelos. "
            "O tipo de sinal sera definido como indeterminado."
        )
    else:
        print(f"Mapeamento global KMeans de tipo de sinal (todos os modelos): {global_sign_model.cluster_to_sign}")

    sign_kmeans_by_model: Dict[str, Optional[SignKMeansModel]] = {
        model_name: global_sign_model for model_name in model_order
    }

    generated: List[str] = []
    skipped: List[str] = []

    for video_name in selected_videos:
        missing_models = [m for m in model_order if video_name not in model_results[m]]
        if missing_models:
            msg = f"Video '{video_name}' ausente no(s) modelo(s): {missing_models}"
            if strict:
                raise KeyError(msg)
            print(f"Ignorando {video_name}: {msg}")
            skipped.append(video_name)
            continue

        save_path = out_dir / f"{video_name}_multi_model_painsign.pdf"
        try:
            plot_multi_model_pain_sign(
                video_name=video_name,
                model_results=model_results,
                path_icopevid_frames=path_icopevid_frames,
                xai_roots=xai_roots,
                save_path=save_path,
                mcdp=mcdp,
                ma_window=ma_window,
                duration_s=duration_s,
                frame_step=frame_step,
                out_size=out_size,
                suffix=suffix,
                xai_alpha=xai_alpha,
                style=style,
                model_colors=model_colors,
                sign_kmeans_by_model=sign_kmeans_by_model,
            )
            generated.append(video_name)
        except Exception as exc:
            if strict:
                raise
            print(f"Ignorando {video_name}: {exc}")
            skipped.append(video_name)

    return {"generated": generated, "skipped": skipped}


# ---------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------

def run_pain_sign_report(
    results_video: Dict[str, Dict],
    *,
    model_name: str,
    path_icopevid_frames: PathLike,
    xai_root: PathLike,
    out_dir: PathLike,
    mcdp: bool,
    ma_window: int = 30,
    duration_s: float = 20.0,
    frame_step: int = 30,
    region_frame_step: int = 1,
    include_region_curves: bool = True,
    region_top_k: int = 0,
    region_selection: str = "mean",
    region_smooth_window: int = 30,
    include_mesh_regions: bool = False,
    landmark_dir: Optional[PathLike] = None,
    mesh_alpha: float = 0.45,
    xai_alpha: float = 0.6,
) -> Dict[str, np.ndarray]:
    """Generates per-video PDFs and returns arrays for metrics."""
    thresholds = get_thresholds_for_model(model_name)
    out_dir = Path(out_dir)
    frames_root = Path(path_icopevid_frames)
    xai_root = Path(xai_root)

    labels: List[int] = []
    preds: List[int] = []
    probs: List[float] = []

    for video_name in results_video.keys():
        video_dir = frames_root / video_name
        strip = load_video_strip(
            video_dir,
            model_name=model_name,
            xai_root=xai_root,
            frame_step=frame_step,
            include_mesh_regions=include_mesh_regions,
            landmark_dir=landmark_dir,
            mesh_alpha=mesh_alpha,
            xai_alpha=xai_alpha,
        )

        true_label = infer_true_label(video_name)
        labels.append(true_label)

        ps = compute_pain_sign(
            results_video[video_name],
            mcdp=mcdp,
            theta_1=thresholds.theta_1,
            ma_window=ma_window,
            duration_s=duration_s,
        )

        preds.append(ps.pred_label)
        probs.append(ps.p_summary)

        region_df = None
        if include_region_curves:
            try:
                region_df = extract_region_scores_video(
                    video_dir=video_dir,
                    xai_root=xai_root,
                    frame_step=region_frame_step,
                    duration_s=duration_s,
                )
            except Exception as exc:
                print(f"Curvas de regiao ignoradas para {video_name}: {exc}")
                region_df = None

        save_path = out_dir / f"{video_name}_{model_name}.pdf"
        plot_pain_sign(
            video_name=video_name,
            strip_img=strip,
            ps=ps,
            thresholds=thresholds,
            true_label=true_label,
            model_name=model_name,
            save_path=save_path,
            region_df=region_df,
            region_top_k=region_top_k,
            region_selection=region_selection,
            region_smooth_window=region_smooth_window,
        )

    gc.collect()

    return {
        "preds": np.asarray(preds, dtype=int),
        "probs": np.asarray(probs, dtype=float),
        "labels": np.asarray(labels, dtype=int),
    }
