# AGENTS.md

Instructions for AI coding agents (OpenAI Codex) working in this repository.

## Scope and goals

This repository contains research code to replicate results for **AI-based neonatal pain assessment from facial images**, including:
- Dataset preparation and face cropping
- Train/validate pipelines (cross-validation / leave-some-subject-out style)
- Uncertainty + calibration utilities
- XAI methods (e.g., Integrated Gradients, GradCAM-like workflows)

Primary goals for agent contributions:
1. Keep experiments **reproducible** (configs, deterministic seeds where applicable, documented CLI).
2. Avoid breaking existing scripts (backwards-compatible CLI; stable imports).
3. Keep code **research-friendly**: readable, minimal hidden magic, clear assumptions.

## Repository map (high level)

Top-level scripts (entry points):
- `create_dataset.py` – builds standardized dataset layout from original sources
- `face_detection.py` – crops faces (uses InsightFace/RetinaFace per README)
- `leave_some_subject_out.py` – creates folds
- `data_augmentation.py` – augments training data + landmarks
- `train.py` – training entry point (uses YAML configs under `models/configs/`)
- `validate.py` – validation/inference entry point

Main packages:
- `models/` – model definitions + YAML configs
- `dataloaders/` – dataset loading / batching
- `XAI/` – explainability methods
- `uncertainty/` – uncertainty estimation tools
- `calibration/` – calibration metrics/methods
- `utils/` – shared helpers

## Environment setup

Preferred options:
- `conda env create -f environment.yml` (if maintained)
- or `pip install -r requirements.txt`

Agent rule:
- If adding a new dependency, prefer standard scientific Python stack.
- Update `requirements.txt` (and `environment.yml` if used) and explain why.

## Data and privacy constraints (hard rules)

This repo expects iCOPE and UNIFESP datasets locally and they are permissioned.
- DO NOT add any dataset files to git.
- DO NOT include real patient images in issues/PRs/logs/tests.
- Avoid adding sample outputs that could leak sensitive information.

Expected local folder layout (as referenced by existing scripts/README):
- `Datasets/Originais/` contains original datasets with original filenames.
- Generated folders may include: `Datasets/NewDataset`, `Datasets/Faces`, `Datasets/Folds`, etc.

Agent rule:
- Any code you add must fail gracefully if data is missing:
  - clear error message
  - no silent partial processing

## How to run (common commands)

Dataset pipeline (typical order):
1. `python create_dataset.py`
2. `python face_detection.py`
3. `python leave_some_subject_out.py`
4. `python data_augmentation.py`

Training:
- `python train.py --config models/configs/<CONFIG>.yaml`

Validation:
- `python validate.py --config models/configs/<CONFIG>.yaml` (if supported by script)
  - If `validate.py` uses different CLI flags, keep it consistent with `train.py` when possible.

Agent rule:
- If you modify CLI flags, preserve old flags or provide deprecation warnings.

## Coding conventions

Language: Python.

Style (pragmatic):
- Prefer type hints for public functions and core utilities.
- Keep modules small and cohesive.
- Avoid global state; prefer passing config/args explicitly.
- Determinism: set seeds where feasible (torch, numpy, python).

I/O conventions:
- Do not hardcode absolute paths.
- Prefer `pathlib.Path`.
- Centralize default paths in one place (e.g., a `utils/paths.py`) if you need to extend path logic.

Logging:
- Prefer `logging` over `print` for long-running scripts.
- Keep logs concise; never print patient-identifying info.

## Configuration rules (YAML)

Models are configured through YAML in `models/configs/`.

Agent rule:
- New experiment knobs belong in YAML first, not hardcoded in scripts.
- Provide safe defaults.
- Validate config fields early (raise a helpful exception with missing/invalid keys).

## Testing and validation expectations

There may be no formal test suite today.

Minimum bar for PRs:
- `python -m compileall .` passes
- Run at least one “no-data” dry-path test:
  - scripts should exit with clear “dataset not found” message rather than stack traces deep inside libraries
- If you add logic that can be unit-tested without datasets:
  - add lightweight tests (e.g., `pytest`) only if it doesn’t disrupt the repo
  - otherwise add a `--dry-run` path or small self-check functions

## Making changes safely

Before editing:
- Identify whether the change impacts published results or metrics.
- Avoid altering default hyperparameters, preprocessing, or thresholds without documenting it.

When changing behavior:
- Add a short note to the PR describing:
  - what changed
  - why it changed
  - how to reproduce/verify

If you refactor:
- Keep imports stable (`from models import ...`, `from XAI import ...`) where possible.

## Output artifacts

Hard rule:
- Do not commit large binaries (models, checkpoints, datasets).
- If needed, document where to download checkpoints and how to place them locally.

Prefer output directories:
- `runs/` or `outputs/` (ignored by git) for experiment artifacts
- Use timestamped subfolders to avoid overwriting results.

## Security and responsible AI notes

This is a medical-adjacent domain.
- Avoid claims that the model is diagnostic.
- Keep “clinical use” language aligned with research context.
- Ensure uncertainty/calibration outputs are described as decision-support signals, not ground truth.

## What to do when uncertain

If something is ambiguous:
1. Inspect the README and existing scripts for precedent.
2. Prefer minimal, reversible changes.
3. Add documentation in README or inline docstrings.
4. If you must choose a default, choose the safest option (no data leakage, no silent success).