from __future__ import annotations

import gc
import pickle
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
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
    ci_alpha: float = 0.18

    # Background bands
    band_no_pain_color: str = "#2E86AB"     # Comfort Blue
    band_pain_color: str = "#D72638"        # Pain Red
    band_ambiguous_color: str = "#F29E4C"   # Alert Orange
    band_ambiguous_alpha: float = 0.0
    band_no_pain_alpha: float = 0.0
    band_pain_alpha: float = 0.0

    # Uncertainty state background
    uncertain_bg_color: str = "#703C3C"
    uncertain_bg_alpha: float = 0.35

    # Typography
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 11


class PainState(IntEnum):
    STABLE_NO_PAIN = 0
    TRANSITION = 1
    TRANSIENT_PAIN = 2
    SUSTAINED_PAIN = 3
    UNCERTAIN = 4


@dataclass(frozen=True)
class StateMachineParams:
    # Thresholds
    theta_1: float                 # pain / no-pain boundary
    theta_3: float          # uncertainty threshold (sigma)
    theta_2_low: Optional[float] = None
    theta_2_high: Optional[float] = None

    # Hysteresis (recommended to avoid flicker around theta_1)
    delta_hyst: float = 0.05       # theta_on = theta_1 + delta, theta_off = theta_1 - delta

    # Durations (in seconds)
    t_confirm: float = 1.5         # time above theta_on to confirm sustained pain
    t_recover: float = 1.0         # time below theta_off to confirm recovery to no-pain
    t_uncertain: float = 1       # time sigma > theta_3 to enter UNCERTAIN (debounce)

    # Transient labeling policy
    transient_enabled: bool = True
    t_transient_max: Optional[float] = None  # if None: t_transient_max = t_confirm (exclusive)


@dataclass
class PainEvent:
    kind: str                       # "transient" or "sustained"
    start_idx: int
    end_idx: int
    peak_idx: int
    peak_p: float
    duration_s: float


STATE_COLORS = {
    0: "#ffffff",   # STABLE_NO_PAIN (white)
    1: "#fff3bf",   # TRANSITION (light yellow)
    2: "#ffd8a8",   # TRANSIENT_PAIN (light orange)
    3: "#ffa8a8",   # SUSTAINED_PAIN (light red)
    4: "#e5dbff",   # UNCERTAIN (light purple/gray)
}

STATE_LABELS = {
    0: "Stable no-pain",
    1: "Transition",
    2: "Transient pain",
    3: "Sustained pain",
    4: "Uncertain",
}


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


def _to_frames(seconds: float, fps: float) -> int:
    # ceil to guarantee minimum dwell time
    return int(np.ceil(seconds * fps))


def _default_state_params(thresholds: PainSignThresholds) -> StateMachineParams:
    return StateMachineParams(
        theta_1=thresholds.theta_1,
        theta_2_low=thresholds.theta_2_low,
        theta_2_high=thresholds.theta_2_high,
        theta_3=thresholds.theta_3,
    )


def run_pain_state_machine(
    p: np.ndarray,
    sigma: np.ndarray,
    fps: float,
    params: StateMachineParams,
    t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[PainEvent], Dict[str, float]]:
    """
    Online-style deterministic state machine.
    Inputs
      p     : (T,) mean pain probability
      sigma : (T,) mean uncertainty (e.g., std from MC dropout)
      fps   : sampling rate of p/sigma (e.g., 30 if per-frame; 1 if per-second)
      params: thresholds and duration settings
      t     : optional timestamps (T,); used only for sanity checks

    Outputs
      states: (T,) int labels (PainState)
      events: list of PainEvent objects
      summary: dict with useful aggregates
    """
    p = np.asarray(p, dtype=float).copy()
    sigma = np.asarray(sigma, dtype=float).copy()
    if p.shape != sigma.shape or p.ndim != 1:
        raise ValueError("p and sigma must be 1D arrays with the same shape.")
    if fps <= 0:
        raise ValueError("fps must be > 0.")
    T = p.shape[0]
    if t is not None:
        t = np.asarray(t)
        if t.shape != (T,):
            raise ValueError("t must have shape (T,) matching p/sigma.")

    theta_on = params.theta_1 + params.delta_hyst
    theta_off = params.theta_1 - params.delta_hyst

    n_confirm = _to_frames(params.t_confirm, fps)
    n_recover = _to_frames(params.t_recover, fps)
    n_uncertain = _to_frames(params.t_uncertain, fps)

    t_transient_max = params.t_transient_max
    if t_transient_max is None:
        t_transient_max = params.t_confirm  # transient if it ends before confirm
    n_transient_max = _to_frames(t_transient_max, fps)

    states = np.full(T, PainState.STABLE_NO_PAIN, dtype=int)

    # Counters for dwell logic
    above_on = 0          # consecutive frames with p >= theta_on (reliable)
    below_off = 0         # consecutive frames with p < theta_off (reliable)
    uncertain_run = 0     # consecutive frames with sigma > theta_3

    # Track last non-UNCERTAIN state for return logic
    last_base_state = PainState.STABLE_NO_PAIN
    current_state = PainState.STABLE_NO_PAIN

    # Episode tracking
    episode_active = False
    episode_start = -1
    episode_peak_idx = -1
    episode_peak_p = -np.inf

    events: List[PainEvent] = []

    def start_episode(i: int):
        nonlocal episode_active, episode_start, episode_peak_idx, episode_peak_p
        episode_active = True
        episode_start = i
        episode_peak_idx = i
        episode_peak_p = p[i]

    def update_episode_peak(i: int):
        nonlocal episode_peak_idx, episode_peak_p
        if p[i] >= episode_peak_p:
            episode_peak_p = p[i]
            episode_peak_idx = i

    def end_episode(i_end: int, kind: str):
        nonlocal episode_active
        if not episode_active:
            return
        dur_s = (i_end - episode_start + 1) / fps
        events.append(
            PainEvent(
                kind=kind,
                start_idx=episode_start,
                end_idx=i_end,
                peak_idx=episode_peak_idx,
                peak_p=float(episode_peak_p),
                duration_s=float(dur_s),
            )
        )
        episode_active = False

    for i in range(T):
        reliable = sigma[i] <= params.theta_3
        if not reliable:
            uncertain_run += 1
        else:
            uncertain_run = 0

        # Enter/Stay UNCERTAIN (debounced)
        if uncertain_run >= n_uncertain:
            current_state = PainState.UNCERTAIN
            states[i] = current_state
            # Do not update pain dwell counters while uncertain
            above_on = 0
            below_off = 0
            # Note: we do not force-end an episode here; uncertainty can "pause" it visually.
            continue

        # If reliable now and we were uncertain previously, return to last base state
        if states[i - 1] == PainState.UNCERTAIN if i > 0 else False:
            current_state = last_base_state

        # Update dwell counters only when reliable
        if reliable and p[i] >= theta_on:
            above_on += 1
            below_off = 0
        elif reliable and p[i] < theta_off:
            below_off += 1
            above_on = 0
        else:
            # inside hysteresis band: don't change counters aggressively
            # keep above_on and below_off as they are (can also decay; leaving stable is fine)
            pass

        # Core state logic (hysteresis + durations)
        if current_state == PainState.STABLE_NO_PAIN:
            if reliable and p[i] >= theta_on:
                # start suspected / transition and start episode
                current_state = PainState.TRANSITION
                last_base_state = current_state
                start_episode(i)
            states[i] = current_state

        elif current_state == PainState.TRANSITION:
            last_base_state = current_state
            if episode_active:
                update_episode_peak(i)

            # Confirm sustained
            if above_on >= n_confirm:
                current_state = PainState.SUSTAINED_PAIN
                last_base_state = current_state
                states[i] = current_state
                continue

            # Recover back to no pain
            if below_off >= n_recover:
                # If it ended before confirm, label as transient (if enabled)
                if params.transient_enabled and episode_active:
                    # Determine episode length; transient if shorter than confirm threshold
                    ep_len = i - episode_start + 1
                    if ep_len < n_confirm and ep_len <= n_transient_max:
                        # Emit transient event and mark current point as transient for visibility
                        end_episode(i, kind="transient")
                        current_state = PainState.TRANSIENT_PAIN
                        states[i] = current_state
                        # Immediately transition back to stable in subsequent frames once recovered
                        # (next iterations will drive it to STABLE via below_off)
                    else:
                        end_episode(i, kind="unknown_short")  # fallback; should be rare
                        current_state = PainState.STABLE_NO_PAIN
                        states[i] = current_state
                else:
                    # No transient labeling: just drop back
                    if episode_active:
                        end_episode(i, kind="unknown_short")
                    current_state = PainState.STABLE_NO_PAIN
                    states[i] = current_state
            else:
                states[i] = current_state

        elif current_state == PainState.TRANSIENT_PAIN:
            # This is a display state to make brief events visible.
            # Once recovery is confirmed, go back to stable.
            if below_off >= n_recover:
                current_state = PainState.STABLE_NO_PAIN
            last_base_state = current_state if current_state != PainState.TRANSIENT_PAIN else PainState.STABLE_NO_PAIN
            states[i] = current_state

        elif current_state == PainState.SUSTAINED_PAIN:
            last_base_state = current_state
            if episode_active:
                update_episode_peak(i)

            # Exit sustained pain only after confirmed recovery
            if below_off >= n_recover:
                end_episode(i, kind="sustained")
                current_state = PainState.STABLE_NO_PAIN
                last_base_state = current_state
            states[i] = current_state

        elif current_state == PainState.UNCERTAIN:
            # handled earlier (debounced), but keep for completeness
            states[i] = current_state

        else:
            raise RuntimeError(f"Unknown state: {current_state}")

    # If episode never closed (signal ends during pain-like activity)
    if episode_active:
        # Classify by current_state
        if current_state == PainState.SUSTAINED_PAIN:
            end_episode(T - 1, kind="sustained")
        else:
            # likely transition / short
            if params.transient_enabled:
                ep_len = (T - 1) - episode_start + 1
                if ep_len < n_confirm and ep_len <= n_transient_max:
                    end_episode(T - 1, kind="transient")
                else:
                    end_episode(T - 1, kind="unknown_short")
            else:
                end_episode(T - 1, kind="unknown_short")

    # Summary metrics (useful for your thesis)
    summary = {
        "time_in_no_pain_s": float(np.sum(states == PainState.STABLE_NO_PAIN) / fps),
        "time_in_transition_s": float(np.sum(states == PainState.TRANSITION) / fps),
        "time_in_transient_s": float(np.sum(states == PainState.TRANSIENT_PAIN) / fps),
        "time_in_sustained_s": float(np.sum(states == PainState.SUSTAINED_PAIN) / fps),
        "time_in_uncertain_s": float(np.sum(states == PainState.UNCERTAIN) / fps),
        "n_events_transient": float(sum(e.kind == "transient" for e in events)),
        "n_events_sustained": float(sum(e.kind == "sustained" for e in events)),
        "theta_on": float(theta_on),
        "theta_off": float(theta_off),
        "n_confirm_frames": float(n_confirm),
        "n_recover_frames": float(n_recover),
        "n_uncertain_frames": float(n_uncertain),
    }

    return states, events, summary


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


def _stack_strip(frame: np.ndarray, overlay: np.ndarray, mesh_overlay: Optional[np.ndarray]) -> np.ndarray:
    if mesh_overlay is None:
        return np.vstack([frame, overlay])
    return np.vstack([frame, overlay, mesh_overlay])


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

    merged_dir = xai_root / video_dir.name / "MERGED_MASKS"

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
    #npz_path = cache_paths.get("npz")
    #if npz_path is not None and npz_path.exists():
    #    with np.load(npz_path, allow_pickle=True) as data:
    ##        return _df_from_npz(data)

    #pkl_path = cache_paths.get("pkl")
    #if pkl_path is not None and pkl_path.exists():
    #    with pkl_path.open("rb") as f:
    #        return pickle.load(f)

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
    merged_dir = xai_root / video_dir.name / "MERGED_MASKS"

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
    for i in tqdm(indices, desc=f"Frames {video_dir.name}"):
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-video correlation between pain sign and XAI region curves."""
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
            print(f"Region curves skipped for {video_name}: {exc}")
            region_df = None

        region_scores: Dict[str, float] = {}
        if region_df is not None and not region_df.empty:
            region_cols = [c for c in region_df.columns if c not in ("frame", "frame_idx", "time_s")]
            if region_cols:
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
                    corr = _safe_corr(ps.p_hat, series_interp, method=method, min_samples=min_samples)
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

    summary_df = pd.DataFrame({
        "mean": corr_df.mean(axis=0, skipna=True),
        "std": corr_df.std(axis=0, ddof=1, skipna=True),
        "n": corr_df.count(axis=0),
    })
    if columns:
        summary_df = summary_df.reindex(index=columns)

    return corr_df, summary_df


# Visualization (doctor-facing)
# ---------------------------------------------------------------------

def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert boolean mask into contiguous [start, end) index segments."""
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    changes = np.diff(m, prepend=m[0])
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if m[0] == 1:
        starts = np.r_[0, starts]
    if m[-1] == 1:
        ends = np.r_[ends, len(m)]
    return list(zip(starts, ends))


def _segments_from_states(states: np.ndarray) -> List[Tuple[int, int, int]]:
    """Convert integer state labels into contiguous [start, end) segments."""
    if states.size == 0:
        return []
    segments: List[Tuple[int, int, int]] = []
    start = 0
    current = int(states[0])
    for idx in range(1, len(states)):
        if int(states[idx]) != current:
            segments.append((start, idx, current))
            start = idx
            current = int(states[idx])
    segments.append((start, len(states), current))
    return segments


def _infer_fps_from_time(time: np.ndarray) -> float:
    if time.size < 2:
        return 1.0
    dt = np.diff(time.astype(float))
    dt = dt[dt > 0]
    if dt.size == 0:
        return 1.0
    return float(1.0 / np.median(dt))


def _theta_crossings(p: np.ndarray, theta_1: float) -> np.ndarray:
    return np.where(np.diff((p >= theta_1).astype(int)) != 0)[0]


def _infer_true_label(name_for_label: str) -> int:
    return 1 if "Pain" in name_for_label else 0


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
    ax.set_ylabel("Pain probability", fontsize=style.label_size)
    ax.set_xlabel("Time (s)", fontsize=style.label_size)
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
    parts = [f"Mean p={ps.p_summary:.3f} -> {pred_txt}"]
    if ps.sigma_summary is not None:
        state = "Certain" if ps.sigma_summary <= thresholds.theta_3 else "Uncertain"
        parts.append(f"Mean sigma={ps.sigma_summary:.3f} -> {state}")
    parts.append(f"Crossings={int(idx_cross.size)}")
    return parts


def _label_strip_rows(ax, rows: int, style: PlotStyle) -> None:
    row_labels = ["Frames", "Frames + XAI"]
    if rows == 3:
        row_labels.append("Mesh regions")
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
    title="Region importance (normalized, stacked)",
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
    ax.set_ylabel("Relative importance")
    ax.set_xlabel("Time (s)")
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


def _plot_state_timeline(ax, time: np.ndarray, states: np.ndarray, style: PlotStyle) -> None:
    if time.size == 0 or states.size == 0:
        return
    for s, e, state in _segments_from_states(states):
        t0 = time[s]
        t1 = time[e - 1] if e - 1 < time.size else time[-1]
        if t1 == t0:
            t1 = t0 + 1e-6
        ax.axvspan(t0, t1, color=STATE_COLORS.get(int(state), "#ffffff"), lw=0)

    if time.size:
        ax.set_xlim(time.min(), time.max())
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("State", fontsize=style.label_size)
    ax.tick_params(axis="x", labelsize=style.tick_size)
    ax.grid(False, axis="both")

    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[k]) for k in STATE_COLORS]
    labels = [STATE_LABELS[k] for k in STATE_COLORS]
    ax.legend(handles=handles, labels=labels, loc="upper center", ncol=3, frameon=True, framealpha=0.9, fontsize=9)


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
    state_params: Optional[StateMachineParams] = None,
    style: PlotStyle = PlotStyle(),
) -> None:
    """Clinical plot with probability curve, region curves, and frame/XAI strip."""
    time = ps.time_s
    p = ps.p_hat
    sigma = ps.sigma_hat
    state_params = state_params or _default_state_params(thresholds)
    fps = _infer_fps_from_time(time)
    sigma_state = sigma if sigma is not None else np.zeros_like(p)
    states, _, _ = run_pain_state_machine(p, sigma_state, fps=fps, params=state_params, t=time if time.size else None)

    region_curves = _prepare_region_curves(
        region_df,
        region_selection=region_selection,
        region_top_k=region_top_k,
        region_smooth_window=region_smooth_window,
        duration_s=float(time[-1]) if time.size else None,
    )
    has_regions = region_curves is not None

    if has_regions:
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(nrows=4, ncols=1, height_ratios=[3.0, 1.8, 0.45, 2.0], hspace=0.12)
    else:
        fig = plt.figure(figsize=(16, 7.5))
        gs = GridSpec(nrows=3, ncols=1, height_ratios=[3.2, 0.45, 2.0], hspace=0.12)

    ax = fig.add_subplot(gs[0])

    _add_background_bands(ax, thresholds, style)

    if sigma is not None:
        upper = np.clip(p + sigma, 0, 1)
        lower = np.clip(p - sigma, 0, 1)
        ax.fill_between(time, lower, upper, color=style.ci_color, alpha=style.ci_alpha, edgecolor="none", label="Uncertainty band (+/- sigma)")

    if sigma is not None:
        certain = sigma <= thresholds.theta_3
    else:
        certain = np.ones_like(p, dtype=bool)

    certain_segs = _segments_from_mask(certain)
    uncertain_segs = _segments_from_mask(~certain)

    for s, e in certain_segs:
        ax.plot(time[s:e], p[s:e], color=style.certain_color, lw=2.2, solid_capstyle="round")
    for s, e in uncertain_segs:
        ax.plot(time[s:e], p[s:e], color=style.uncertain_color, lw=2.6, solid_capstyle="round")

    _add_threshold_lines(ax, time, thresholds, style)

    idx_cross = _theta_crossings(p, thresholds.theta_1)
    if idx_cross.size:
        ax.scatter(time[idx_cross], np.full(idx_cross.shape, thresholds.theta_1), s=40, color=style.uncertain_color, zorder=5, label="Decision crossings")

    _format_prob_axis(ax, time, style, has_regions)

    label_txt = "Pain" if true_label == 1 else "No pain"
    pred_txt = "Pain" if ps.pred_label == 1 else "No pain"
    reliability_parts = _build_reliability_parts(ps, thresholds, pred_txt, idx_cross)

    ax.set_title(
        f"{video_name} | Model: {model_name} | True: {label_txt} | " + " | ".join(reliability_parts),
        fontsize=style.title_size,
        pad=10,
    )

    handles = [
        plt.Line2D([0], [0], color=style.certain_color, lw=2.2, label="Probability (certain)"),
        plt.Line2D([0], [0], color=style.uncertain_color, lw=2.6, label="Probability (uncertain)"),
    ]
    if sigma is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=style.ci_color, alpha=style.ci_alpha, label="+/- sigma band"))
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9)

    if has_regions and region_curves is not None:
        axr = fig.add_subplot(gs[1], sharex=ax)
        region_dict = {region: region_curves.data[:, idx] for idx, region in enumerate(region_curves.labels)}
        default_colors = plt.cm.get_cmap("tab20", len(region_curves.labels))
        region_colors = [REGION_COLOR_MAP.get(region, default_colors(idx)) for idx, region in enumerate(region_curves.labels)]
        plot_region_importance_stack(
            region_curves.time_s,
            region_dict,
            smooth_window=0,
            top_k=None,
            title="",
            ax=axr,
            colors=region_colors,
            show=False,
            legend_kwargs={"fontsize": 9},
        )
        axr.set_ylabel("Relative importance", fontsize=style.label_size)
        axr.set_xlabel("Time (s)", fontsize=style.label_size)
        axr.tick_params(axis="both", labelsize=style.tick_size)
        axr.grid(True, axis="y", alpha=0.18, color=style.grid_color)
        axr.set_ylim(0, 1)
        if time.size:
            axr.set_xlim(time.min(), time.max())
        ax_state = fig.add_subplot(gs[2], sharex=ax)
        ax_strip = fig.add_subplot(gs[3])
    else:
        ax_state = fig.add_subplot(gs[1], sharex=ax)
        ax_strip = fig.add_subplot(gs[2])

    _plot_state_timeline(ax_state, time, states, style)
    ax_state.tick_params(labelbottom=False)
    ax_state.set_xlabel("")

    ax_strip.imshow(strip_img)
    ax_strip.axis("off")

    row_count = 3 if (strip_img.shape[0] % 3 == 0) else 2
    _label_strip_rows(ax_strip, row_count, style)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

        true_label = _infer_true_label(video_name)
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
                print(f"Region curves skipped for {video_name}: {exc}")
                region_df = None

        save_path = out_dir / f"{video_name}_{model_name}.jpg" #CHANGE TO PDF HERE
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

# ---------------------------------------------------------------------
# Animation: frame-by-frame pain sign + XAI overlay
# ---------------------------------------------------------------------

def _resolve_fps_duration(num_frames: int, fps: Optional[float], duration_s: Optional[float], default_fps: float = 30.0) -> Tuple[float, float]:
    if fps is None and duration_s is None:
        fps = float(default_fps)
        duration_s = num_frames / fps if fps > 0 else 0.0
    elif fps is None:
        fps = max(1.0, num_frames / float(duration_s))
    elif duration_s is None:
        duration_s = num_frames / float(fps)
    return float(fps), float(duration_s)


def render_pain_sign_animation(
    *,
    video_name: str,
    video_results: dict,
    model_name: str,
    video_dir: PathLike,
    xai_root: PathLike,
    out_path: PathLike,
    mcdp: bool,
    thresholds: Optional[PainSignThresholds] = None,
    true_label: Optional[int] = None,
    fps: Optional[float] = None,
    duration_s: Optional[float] = 20.0,
    ma_window: int = 30,
    frame_step: int = 1,
    out_size: Tuple[int, int] = (256, 256),
    suffix: str = ".jpg",
    include_region_curves: bool = True,
    region_df: Optional[pd.DataFrame] = None,
    region_frame_step: int = 1,
    region_top_k: int = 0,
    region_selection: str = "mean",
    region_smooth_window: int = 30,
    include_mesh_regions: bool = False,
    landmark_dir: Optional[PathLike] = None,
    mesh_alpha: float = 0.45,
    xai_alpha: float = 0.6,
    mask_key: str = "mask_raw",
    figsize: Tuple[int, int] = (16, 9),
    dpi: int = 110,
    codec: str = "mp4v",
    show_progress: bool = True,
    style: PlotStyle = PlotStyle(),
) -> Path:
    """Create an MP4 animation for a single video."""
    if thresholds is None:
        thresholds = get_thresholds_for_model(model_name)

    video_dir = Path(video_dir)
    xai_root = Path(xai_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh_dir = _resolve_landmark_dir(video_dir, landmark_dir) if include_mesh_regions else None

    img_files = _list_frames(video_dir, suffix)
    if not img_files:
        raise ValueError(f"No frames found in {video_dir} with suffix {suffix}")

    fps, duration_s = _resolve_fps_duration(len(img_files), fps, duration_s)

    ps = compute_pain_sign(
        video_results,
        mcdp=mcdp,
        theta_1=thresholds.theta_1,
        ma_window=ma_window,
        duration_s=duration_s,
    )

    if true_label is None:
        name_for_label = video_name or video_dir.name
        true_label = _infer_true_label(name_for_label)

    if include_region_curves and region_df is None:
        try:
            region_df = extract_region_scores_video(
                video_dir=video_dir,
                xai_root=xai_root,
                frame_step=region_frame_step,
                duration_s=duration_s,
                mask_key=mask_key,
            )
        except Exception as exc:
            print(f"Region curves skipped for {video_name}: {exc}")
            region_df = None

    region_curves = _prepare_region_curves(
        region_df,
        region_selection=region_selection,
        region_top_k=region_top_k,
        region_smooth_window=region_smooth_window,
        duration_s=duration_s,
    )
    has_regions = include_region_curves and region_curves is not None

    # Figure setup
    if has_regions:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(nrows=3, ncols=1, height_ratios=[3.0, 1.8, 2.0], hspace=0.12)
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[3.2, 2.0], hspace=0.12)

    ax = fig.add_subplot(gs[0])

    _add_background_bands(ax, thresholds, style)

    time = ps.time_s
    p = ps.p_hat
    sigma = ps.sigma_hat

    if sigma is not None:
        certain = sigma <= thresholds.theta_3
    else:
        certain = np.ones_like(p, dtype=bool)

    p_certain = np.where(certain, p, np.nan)
    p_uncertain = np.where(~certain, p, np.nan)

    line_certain, = ax.plot([], [], color=style.certain_color, lw=2.2, solid_capstyle="round")
    line_uncertain, = ax.plot([], [], color=style.uncertain_color, lw=2.6, solid_capstyle="round")

    _add_threshold_lines(ax, time, thresholds, style)

    idx_cross = _theta_crossings(p, thresholds.theta_1)
    cross_scatter = ax.scatter([], [], s=40, color=style.uncertain_color, zorder=5)

    _format_prob_axis(ax, time, style, has_regions)

    label_txt = "Pain" if true_label == 1 else "No pain"
    pred_txt = "Pain" if ps.pred_label == 1 else "No pain"
    reliability_parts = _build_reliability_parts(ps, thresholds, pred_txt, idx_cross)

    ax.set_title(
        f"{video_name} | Model: {model_name} | True: {label_txt} | " + " | ".join(reliability_parts),
        fontsize=style.title_size,
        pad=10,
    )

    handles = [
        plt.Line2D([0], [0], color=style.certain_color, lw=2.2, label="Probability (certain)"),
        plt.Line2D([0], [0], color=style.uncertain_color, lw=2.6, label="Probability (uncertain)"),
    ]
    if sigma is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=style.ci_color, alpha=style.ci_alpha, label="+/- sigma band"))
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9)

    # Region curves
    if has_regions and region_curves is not None:
        axr = fig.add_subplot(gs[1], sharex=ax)
        region_dict = {region: region_curves.data[:, idx] for idx, region in enumerate(region_curves.labels)}
        default_colors = plt.cm.get_cmap("tab20", len(region_curves.labels))
        region_colors = [REGION_COLOR_MAP.get(region, default_colors(idx)) for idx, region in enumerate(region_curves.labels)]
        plot_region_importance_stack(
            region_curves.time_s,
            region_dict,
            smooth_window=0,
            top_k=None,
            title="",
            ax=axr,
            colors=region_colors,
            show=False,
            legend_kwargs={"fontsize": 9},
        )
        axr.set_ylabel("Relative importance", fontsize=style.label_size)
        axr.set_xlabel("Time (s)", fontsize=style.label_size)
        axr.tick_params(axis="both", labelsize=style.tick_size)
        axr.grid(True, axis="y", alpha=0.18, color=style.grid_color)
        axr.set_ylim(0, 1)
        if time.size:
            axr.set_xlim(time.min(), time.max())
    else:
        axr = None

    # Frame + XAI panel
    if has_regions:
        ax_strip = fig.add_subplot(gs[2])
    else:
        ax_strip = fig.add_subplot(gs[1])

    h, w = out_size[1], out_size[0]
    strip_rows = 3 if include_mesh_regions else 2
    strip_init = np.zeros((h * strip_rows, w, 3), dtype=np.float32)
    strip_im = ax_strip.imshow(strip_init)
    ax_strip.axis("off")
    _label_strip_rows(ax_strip, strip_rows, style)

    # Video writer setup
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    pad_w = width % 2
    pad_h = height % 2
    fps_out = max(1.0, float(fps) / float(frame_step))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps_out), (width + pad_w, height + pad_h))

    frame_indices = np.arange(0, len(img_files), frame_step)
    curve_indices = np.searchsorted(time, frame_indices / float(fps), side="right") if time.size else np.zeros_like(frame_indices)

    if show_progress:
        iterator = tqdm(list(enumerate(frame_indices)), desc=f"Animating {video_name}")
    else:
        iterator = list(enumerate(frame_indices))

    merged_dir = xai_root / video_dir.name / "MERGED_MASKS"
    fill_between = None

    cross_offsets = None
    cross_times = None
    if idx_cross.size:
        cross_times = time[idx_cross]
        cross_offsets = np.c_[cross_times, np.full_like(cross_times, thresholds.theta_1)]

    def _set_line(line, x, y):
        n = min(len(x), len(y))
        if n <= 0:
            line.set_data([], [])
        else:
            line.set_data(x[:n], y[:n])

    for pos, frame_idx in iterator:
        frame_path = img_files[frame_idx]

        frame = _safe_read_rgb(frame_path, out_size)
        mask_path = merged_dir / f"{frame_path.stem}.npz"
        mask = _safe_read_xai_mask(mask_path, out_size, mask_key=mask_key)
        overlay = _blend_xai_overlay(frame, mask, xai_alpha)
        if include_mesh_regions and mesh_dir is not None:
            mesh_overlay = _safe_read_region_overlay(
                frame_path=frame_path,
                frame=frame,
                landmark_dir=mesh_dir,
                size=out_size,
                alpha=mesh_alpha,
            )
        else:
            mesh_overlay = None
        strip_im.set_data(_stack_strip(frame, overlay, mesh_overlay))

        curve_idx = int(curve_indices[pos])

        _set_line(line_certain, time[:curve_idx], p_certain[:curve_idx])
        _set_line(line_uncertain, time[:curve_idx], p_uncertain[:curve_idx])

        if sigma is not None:
            if fill_between is not None:
                fill_between.remove()
            upper = np.clip(p[:curve_idx] + sigma[:curve_idx], 0, 1)
            lower = np.clip(p[:curve_idx] - sigma[:curve_idx], 0, 1)
            fill_between = ax.fill_between(time[:curve_idx], lower, upper, color=style.ci_color, alpha=style.ci_alpha, edgecolor="none")

        if cross_offsets is not None and cross_times is not None:
            k = np.searchsorted(cross_times, frame_idx / float(fps), side="right")
            if k:
                cross_scatter.set_offsets(cross_offsets[:k])
            else:
                cross_scatter.set_offsets(np.empty((0, 2)))

        fig.canvas.draw()
        # Matplotlib compatibility: newer backends expose buffer_rgba() instead of tostring_rgb()
        if hasattr(fig.canvas, "tostring_rgb"):
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape((height, width, 3))
        else:
            buf = np.asarray(fig.canvas.buffer_rgba())
            img = buf[..., :3].copy()
        if pad_w or pad_h:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img_bgr)

    writer.release()
    plt.close(fig)

    return out_path
