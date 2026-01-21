import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import os
import logging
import time

PERTURBATION_LEVELS = np.arange(0, 100, 1)
PERTURB_TYPE = ["impute"]

def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"

for xx in ["NCNN", "VGGFace", "ViT_B_32"]:
    MODEL_NAME_AUX = xx        # "NCNN", "VGGFace", "ViT_B_32"

    if MODEL_NAME_AUX == "NCNN":
        MODEL_NAME_2 = "NCNN_FINAL"
    elif MODEL_NAME_AUX == "VGGFace":
        MODEL_NAME_2 = "VGGFace_FINAL"
    elif MODEL_NAME_AUX == "ViT_B_32":
        MODEL_NAME_2 = "ViT_B_32_ENSEMBLE_FINAL"

    ALIGN = False                  # True or False
    LOAD_TYPE = "trained"          # all | trained | random | baselines
    IMPORTANCE = "MoRF"
    MASK_KEY = "mask_raw"
    MASK_DTYPE = np.float32

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    TARGET_EXPLAINERS = {
        "Saliency", "IntegratedGradients", "DeepLift", "DeepLiftShap",
        "GradientShap", "GradCAM", "GuidedGradCAM", "Deconvolution",
        "Occlusion", "Lime",
    }


    def load_masks(xai_dirs, mask_key="mask_raw", aligned=True, method_filter=None, dtype=np.float32):
        """Return {sample_id: {method_name: normalized 2D mask}} for every .npz in the folders."""
        samples = {}
        for xai_dir in map(Path, xai_dirs):
            method = xai_dir.name
            if method_filter is not None and method not in method_filter:
                continue
            pattern = "aligned/*.npz" if aligned else "*.npz"
            for npz_file in tqdm(xai_dir.glob(pattern), desc=f"Loading {method}", leave=False):
                with np.load(npz_file) as archive:
                    if mask_key not in archive:
                        raise KeyError(f"{npz_file} missing '{mask_key}' array.")
                    mask = archive[mask_key]
                collapsed = mask.sum(axis=2, dtype=dtype)
                norm_mask = normalize_mask_0_1(collapsed, dtype=dtype)
                samples.setdefault(npz_file.stem, {})[method] = norm_mask
        return samples

    def compute_aopc(explainers, return_steps=False):
        summary = {}
        step_stats = {} if return_steps else None
        for name, payload in explainers.items():
            original = np.asarray(payload["original"])
            perturbed = np.asarray(payload["perturbed"])

            if original.shape != perturbed.shape:
                raise ValueError(
                    f"{name}: original and perturbed shapes differ "
                    f"{original.shape} vs {perturbed.shape}"
                )

            per_step = (original[:, [0]] - perturbed).mean(axis=0)
            summary[name] = float(per_step.mean())
            if return_steps:
                step_stats[name] = per_step
        return summary, step_stats


    def normalize_0_1(values, eps=1e-12):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        vmin = float(values.min())
        vmax = float(values.max())
        denom = vmax - vmin
        if denom < eps:
            return np.ones_like(values)
        return (values - vmin) / (denom + eps)


    def normalize_mask_0_1(mask, eps=1e-12, dtype=np.float32):
        mask = np.asarray(mask, dtype=dtype)
        finite = np.isfinite(mask)
        if not finite.any():
            return np.zeros_like(mask)
        vmin = float(mask[finite].min())
        vmax = float(mask[finite].max())
        denom = vmax - vmin
        if denom < eps:
            return np.zeros_like(mask)
        np.nan_to_num(mask, copy=False, nan=vmin, posinf=vmax, neginf=vmin)
        mask -= vmin
        mask /= (denom + eps)
        return mask

    logger.info("Model %s (%s)", MODEL_NAME_AUX, MODEL_NAME_2)
    logger.info("Settings: align=%s load=%s importance=%s", ALIGN, LOAD_TYPE, IMPORTANCE)

    video_dir = Path("experiments") / MODEL_NAME_2 / "icopevid"
    video_list = list(os.listdir(video_dir))
    total_videos = len(video_list)
    model_start = time.perf_counter()
    logger.info("Videos: %d in %s", total_videos, video_dir)
    for video_idx, video in enumerate(video_list, start=1):
        video_start = time.perf_counter()
        HEATMAP_DIR = video_dir / video

        logger.info("Video %s (%d/%d)", video, video_idx, total_videos)
        xai_dirs = [p for p in HEATMAP_DIR.iterdir() if p.is_dir()]
        samples = load_masks(
            xai_dirs,
            mask_key=MASK_KEY,
            aligned=ALIGN,
            method_filter=TARGET_EXPLAINERS,
            dtype=MASK_DTYPE,
        )
        total_masks = sum(len(m) for m in samples.values())
        logger.info("Loaded %d samples (%d masks)", len(samples), total_masks)


        for perturb in PERTURB_TYPE:
            logger.info("Perturbation: %s", perturb)
            curves_path = (
                Path("sdumont_scripts")
                / f"perturb_pixel_curves_{MODEL_NAME_AUX}_{perturb}_{IMPORTANCE}_{LOAD_TYPE}.pkl"
            )
            random_path = (
                Path("sdumont_scripts")
                / f"perturb_pixel_curves_{MODEL_NAME_AUX}_{perturb}_random_{LOAD_TYPE}.pkl"
            )

            if not curves_path.exists():
                raise FileNotFoundError(f"Explainability curves missing: {curves_path}")
            if not random_path.exists():
                raise FileNotFoundError(f"Random baseline curves missing: {random_path}")

            method_curves = pickle.loads(curves_path.read_bytes())
            random_curves = pickle.loads(random_path.read_bytes())
            logger.info("Loaded curves and random baseline")

            summary, _ = compute_aopc(method_curves)
            random_summary, _ = compute_aopc(random_curves)

            random_value = next(iter(random_summary.values()))
            aopc_dicts = {name: value - random_value for name, value in summary.items()}
            aopc_values = np.array(list(aopc_dicts.values()), dtype=np.float64)
            aopc_norm_values = normalize_0_1(aopc_values)
            aopc_norm = dict(zip(aopc_dicts.keys(), aopc_norm_values))
            #print(aopc_dicts)
            #aopc_dicts["random"] = 0.0  # keep baseline in the dict (not used in merging unless you add it to TARGET_EXPLAINERS)

            output_root = Path("icopevid_XAI") / MODEL_NAME_AUX / video / "MERGED_MASKS"
            output_root.mkdir(parents=True, exist_ok=True)
            logger.info("Output: %s", output_root)

            for ID, masks in tqdm(samples.items(), desc=f"Merging {perturb}"):
                if not masks:
                    continue
                merged_mask = np.zeros_like(next(iter(masks.values())), dtype=MASK_DTYPE)

                weights = np.array([aopc_norm.get(m, 0.0) for m in masks], dtype=MASK_DTYPE)
                denom = weights.sum()
                if denom < 1e-12:
                    weights = np.ones_like(weights)
                    denom = weights.sum()

                inv_denom = 1.0 / denom
                scratch = np.empty_like(merged_mask)
                for (method, norm_mask), w in zip(masks.items(), weights):
                    weight = w * inv_denom
                    np.multiply(norm_mask, weight, out=scratch)
                    merged_mask += scratch

                merged_mask = normalize_mask_0_1(merged_mask, dtype=merged_mask.dtype)

                np.savez_compressed(output_root / f"{ID}.npz", mask_raw=merged_mask)

        video_elapsed = time.perf_counter() - video_start
        model_elapsed = time.perf_counter() - model_start
        avg_per_video = model_elapsed / video_idx
        remaining = avg_per_video * (total_videos - video_idx)
        logger.info(
            "Done %s in %s | ETA %s",
            video,
            format_duration(video_elapsed),
            format_duration(remaining),
        )

    logger.info("Model %s done in %s", MODEL_NAME_AUX, format_duration(time.perf_counter() - model_start))
