import warnings
warnings.filterwarnings("ignore")

import os
from collections import defaultdict

from tqdm import tqdm
from PIL import Image, ImageFilter

import torch
import numpy as np
import pandas as pd

from captum.attr import (
    IntegratedGradients,
    Saliency,
    DeepLift,
    Occlusion,
    LayerGradCam,
    GuidedGradCam,
    Deconvolution,
    GradientShap,
    DeepLiftShap,
    Lime,
    LayerAttribution,
)
from captum.metrics import sensitivity_max, infidelity

from skimage.segmentation import slic

from utils.utils import create_folder, load_config
from dataloaders.presets import PresetTransform
from models import *

def perturb_fn(inputs):
    noise = torch.tensor(np.random.normal(0, 0.1, inputs.shape)).float().to(inputs.device)
    return noise, torch.clip(inputs - noise, 0, 1)

def rgb_to_gray_and_scale(x):
    x = np.asarray(x)
    # Shape must be in (H, W, C)
    x_combined = np.sum(x, axis=2)
    x_combined = (x_combined > 0) * x_combined

    sorted_vals = np.sort(np.abs(x_combined).flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id: int = np.where(cum_sums >= cum_sums[-1] * 0.01 * 100)[0][0]
    threshold = sorted_vals[threshold_id]

    attr_norm = x_combined / threshold

    return np.clip(attr_norm, -1, 1)

# --- Create superpixel feature mask for Captum ---
def make_feature_mask(img_tensor, n_segments=100):
    x = img_tensor.detach().cpu().squeeze(0)  # 3 x H x W
    x_np = x.numpy()
    x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
    x_np = np.transpose(x_np, (1, 2, 0))  # H, W, 3

    seg = slic(x_np, n_segments=n_segments, compactness=10.0, sigma=0.0,
               start_label=0, channel_axis=2)

    seg_t = torch.from_numpy(seg).long().unsqueeze(0).unsqueeze(0)
    return seg_t, seg

# ------------------------------------------------------------------------------

device = 'cuda'


# ------------------------------------------------------------------------------

def resolve_experiment(exp_name: str, device: torch.device):
    if "NCNN" in exp_name:
        model = NCNN().to(device)
        return {
            "model": model,
            "img_size": 120,
            "transform": PresetTransform("NCNN").transforms,
            "layer": model.merge_branch[0],
            "model_name": exp_name,
        }
    if "VGGFace" in exp_name:
        model = VGGFace().to(device)
        return {
            "model": model,
            "img_size": 224,
            "transform": PresetTransform("VGGFace").transforms,
            "layer": model.VGGFace.features.conv5_3,
            "model_name": exp_name,
        }
    if "ViT" in exp_name:
        model = ViT().to(device)
        return {
            "model": model,
            "img_size": 224,
            "transform": PresetTransform("ViT").transforms,
            "layer": model.ViT.conv_proj,
            "model_name": exp_name,
        }
    return None


def ensure_feature_mask(ctx: dict, n_segments: int = 100):
    if "feature_mask" not in ctx:
        mask, _ = make_feature_mask(ctx["input"], n_segments=n_segments)
        ctx["feature_mask"] = mask.to(ctx["device"]).contiguous()
    return ctx["feature_mask"]


# explainer catalogue ----------------------------------------------------------

EXPLAINER_SPECS = [
    (
        "IntegratedGradients",
        {
            "factory": lambda model, layer: IntegratedGradients(model),
            "prepare": lambda ctx: {"attribute": {"internal_batch_size": 10}},
        },
    ),
    (
        "Saliency",
        {
            "factory": lambda model, layer: Saliency(model),
        },
    ),
    (
        "DeepLift",
        {
            "factory": lambda model, layer: DeepLift(model),
            "prepare": lambda ctx: {"attribute": {"baselines": ctx["blurred"]}},
        },
    ),
    (
        "Occlusion",
        {
            "factory": lambda model, layer: Occlusion(model),
            "prepare": lambda ctx: {
                "attribute": {
                    "sliding_window_shapes": (3, 20, 20),
                    "strides": (3, 5, 5),
                }
            },
        },
    ),
    (
        "GradCAM",
        {
            "factory": lambda model, layer: LayerGradCam(model, layer),
            "postprocess": lambda attr, ctx: LayerAttribution.interpolate(
                attr, ctx["target_shape"], interpolate_mode="bilinear"
            ).repeat(1, 3, 1, 1),
            "prepare": lambda ctx: {
                "relu_attributions": True,
            }
        },
    ),
    (
        "GuidedGradCAM",
        {
            "factory": lambda model, layer: GuidedGradCam(model, layer),
        },
    ),
    (
        "Deconvolution",
        {
            "factory": lambda model, layer: Deconvolution(model),
        },
    ),
    (
        "GradientShap",
        {
            "factory": lambda model, layer: GradientShap(model),
            "prepare": lambda ctx: {
                "attribute": {
                    "baselines": torch.zeros_like(ctx["input"]),
                    "n_samples": 5,
                    "stdevs": 0.0,
                }
            },
        },
    ),
    (
        "DeepLiftShap",
        {
            "factory": lambda model, layer: DeepLiftShap(model),
            "prepare": lambda ctx: {
                "attribute": {"baselines": ctx["blurred"].repeat(5, 1, 1, 1)}
            },
        },
    ),
    (
        "Lime",
        {
            "factory": lambda model, layer: Lime(model),
            "prepare": lambda ctx: {
                "attribute": {
                    "baselines": torch.zeros_like(ctx["input"]),
                    "feature_mask": ensure_feature_mask(ctx),
                    "n_samples": 500,
                    "perturbations_per_eval": 64,
                    "show_progress": False,
                }
            },
        },
    ),
]


# main pipeline ----------------------------------------------------------------

for model_name in ["NCNN", "VGGFace", "ViT_B_32"]:

    path_experiments = os.path.join('experiments', model_name)

    all_data = defaultdict(list)

    for exp in os.listdir(path_experiments):
        if any(ext in exp for ext in (".pkl", "masks", ".png", ".pdf")):
            continue

        experiment_cfg = resolve_experiment(exp, device)
        if experiment_cfg is None:
            continue

        model = experiment_cfg["model"]
        img_size = experiment_cfg["img_size"]
        transform = experiment_cfg["transform"]
        layer = experiment_cfg["layer"]

        path_model = os.path.join(path_experiments, exp, "Model", "best_model.pt")
        path_yaml = os.path.join(path_experiments, exp, "Model", "config.yaml")
        config = load_config(path_yaml)
        test_path = config["path_test"].replace("\\", "/")

        state_dict = torch.load(path_model, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        explainers = {name: spec["factory"](model, layer) for name, spec in EXPLAINER_SPECS}

        image_files = [f for f in os.listdir(test_path) if f.lower().endswith(".jpg")]
        for image_file in tqdm(image_files):
            full_img_path = os.path.join(test_path, image_file)

            img_rgb = Image.open(full_img_path).convert("RGB")
            img_rgb = img_rgb.resize((img_size, img_size))
            img_name = os.path.splitext(image_file)[0]
            label = 1 if img_name.split("_")[3] == "pain" else 0

            if "VGGFace" in exp:
                img_input = Image.fromarray(np.array(img_rgb)[:, :, ::-1])
            else:
                img_input = img_rgb

            blurred_image = img_input.filter(ImageFilter.GaussianBlur(radius=5))

            transformed = transform(img_input)
            transformed_blurred = transform(blurred_image)

            base_input = transformed.unsqueeze(0).to(device)
            base_blurred = transformed_blurred.unsqueeze(0).to(device)

            ctx_base = {
                "device": device,
                "target_shape": (img_size, img_size),
                "input_base": base_input,
                "blurred": base_blurred,
            }

            for XAI_name, spec in EXPLAINER_SPECS:
                explainer = explainers[XAI_name]

                method_ctx = dict(ctx_base)
                method_ctx["input"] = (
                    ctx_base["input_base"].clone().detach().requires_grad_(True)
                )

                if XAI_name == "Lime" or XAI_name == "DeepLift":
                    method_ctx["input"] = (
                        ctx_base["input_base"].clone().detach().requires_grad_(True).contiguous()
                )

                spec_kwargs = spec.get("prepare", lambda ctx: {})(method_ctx)
                attr_kwargs = spec_kwargs.get("attribute", {})
                sensitivity_kwargs = dict(attr_kwargs)
                sensitivity_kwargs.update(spec_kwargs.get("sensitivity", {}))
                infidelity_kwargs = spec_kwargs.get("infidelity", {}).copy()

                attributions = explainer.attribute(method_ctx["input"], **attr_kwargs)

                if "postprocess" in spec:
                    attributions = spec["postprocess"](attributions, method_ctx)

                sens = sensitivity_max(
                    explainer.attribute, method_ctx["input"], **sensitivity_kwargs
                )
                infid = infidelity(
                    model,
                    perturb_fn,
                    method_ctx["input"],
                    attributions,
                    n_perturb_samples=30,
                    **infidelity_kwargs,
                )

                attributions_np = (
                    attributions.squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )
                attributions_normalized = rgb_to_gray_and_scale(attributions_np)

                output_dir = os.path.join("RGU", model_name, XAI_name)
                create_folder(output_dir)
                output_path = os.path.join(output_dir, f"{img_name}.npz")
                np.savez_compressed(output_path, mask=attributions_normalized)

                all_data["img_path"].append(full_img_path)
                all_data["fold"].append(os.path.basename(os.path.dirname(test_path)))
                all_data["label"].append(label)
                all_data["sensitivity"].append(float(sens))
                all_data["infidelity"].append(float(infid))
                all_data["mask_path"].append(output_path)

                with torch.no_grad():
                    probs = model.predict(ctx_base["input_base"])
                pred = (probs >= 0.5).int()
                all_data["probability"].append(float(probs))
                all_data["prediction"].append(int(pred))
                all_data["XAI_name"].append(XAI_name)

    dataframe = pd.DataFrame(all_data)
    create_folder(os.path.join("RGU", model_name))
    dataframe.to_csv(os.path.join("RGU", model_name, "explainers.csv"), index=False)