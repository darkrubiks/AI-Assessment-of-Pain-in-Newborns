# AGENTS.md

Repository-wide operating instructions for AI coding agents, including OpenAI Codex. Read this file before working and inspect applicable nested instructions. Keep this manual discoverable at the repository root.

## Project context

This is scientific research software for automatic neonatal pain assessment from facial images using machine learning and computer vision. It includes dataset preparation, facial-region analysis, neural-network training, validation, predictive uncertainty, probability calibration, explainable AI (XAI), selective prediction and temporal analysis of video predictions.

Implementation changes can affect reproducibility and published findings. Models, attribution maps, uncertainty estimates and temporal signals are experimental research outputs, not diagnostic tools or independently established clinical ground truth. Do not claim clinical effectiveness or readiness without supporting evidence.

## Agent operating principles

- **Understand before modifying:** inspect relevant code, documentation, dependencies, configurations and Git history before choosing an approach.
- **Make minimal changes:** prefer focused, reversible modifications; avoid unrelated refactoring, dependencies or formatting.
- **Preserve scientific integrity:** changes to experimental behavior, preprocessing, thresholds, mathematical definitions, evaluation protocols or dataset partitions require explicit task authorization and appropriate validation.
- **Preserve reproducibility:** retain existing seed behavior, configurations, checkpoint handling and evaluation procedures. Record the commit, environment, data revision, splits and checkpoints needed to reproduce an experiment.
- **Use evidence:** never invent methodology, results, references or performance claims. Distinguish implemented functionality from proposals and historical notebook outputs.
- **Protect privacy:** restricted data must remain local unless the user explicitly authorizes a permitted transfer. An ignored directory is not a privacy boundary.
- **Verify changes:** perform relevant static checks, targeted tests and diff review before completing a task. Report checks that failed or could not be performed.
- **Preserve user work:** inspect the initial Git state and do not overwrite, revert, delete or move unrelated changes or files with uncertain ownership.

## Repository architecture and supported commands

Run documented commands from the repository root. See [README.md](README.md) for prerequisites, environments, analysis dependencies and known limitations.

| Location | Implemented purpose |
|---|---|
| `create_dataset.py` | Standardize and copy iCOPE/UNIFESP images and UNIFESP perception heatmaps using the metadata CSVs. |
| `face_detection.py` | Detect and crop faces with InsightFace, save landmarks and update image metadata. |
| `leave_some_subject_out.py` | Create subject-grouped folds and the separate `TrainAll` split. |
| `data_augmentation.py` | Resize images and augment training images and landmarks. |
| `crop_facial_regions.py` | Crop semantic facial regions using precomputed landmarks. |
| `train.py` | Train and evaluate a configured classifier and save the best checkpoint, logs and plots. |
| `validate.py` | Shared classification/calibration metrics and plotting functions; no standalone validation CLI. |
| `ttt_optimized.py` | Temporal pain-sign calculations, operational thresholds, metrics, facial-region curves and report/video generation. |
| `models/`, `models/configs/` | NCNN, VGGFace and ViT implementations and experiment YAMLs. |
| `dataloaders/` | Still-image/video-frame datasets and transformation presets. |
| `uncertainty/` | MC-dropout implementation; ensemble analyses also appear in notebooks. |
| `calibration/` | Calibration transformations and probability-calibration metrics. |
| `XAI/` | Integrated Gradients, Grad-CAM, RISE, feature maps, attention rollout, facial-region masks, explanation metrics and postprocessing. |
| `corruption/` | Image perturbations used in explanation analyses. |
| `facial_features/` | Eye and mouth aspect-ratio calculations from landmarks. |
| `utils/` | Configuration, CSV, directory, landmark and plotting helpers and plotting styles. |
| `tests/test_pain_sign_layout.py` | Synthetic, dataset-free temporal-report/layout and missing-data checks using `unittest`. |
| `iCOPE+UNIFESP_data.csv`, `UNIFESP_percep_heatmaps.csv` | Existing dataset metadata; preserve provenance and review privacy before exposing contents or adding metadata. |

Root notebooks cover training (`0_train.ipynb`), prediction/calibration exports (`1_create_data.ipynb`), classification/calibration metrics (`2_validation_metrics.ipynb`), uncertainty (`3_uncertainty.ipynb`), calibration comparisons (`4_calibration.ipynb`), XAI (`5_XAI.ipynb`), selective prediction (`6_limits.ipynb`) and video/temporal analysis (`7_icopevid.ipynb`). `RGU_XAI_PAPER.ipynb` and `RGU_notebook.ipynb` contain additional explanation analyses. `graficos axuliares.ipynb`, `lancet.ipynb` and `teste_cassia.ipynb` include auxiliary or historical explorations. They are not a guaranteed executable sequence; inspect each notebook's prerequisites and compatibility notes.

The prepared-data pipeline is:

```bash
python create_dataset.py
python face_detection.py
python leave_some_subject_out.py
python data_augmentation.py
```

Original still-image resources are expected under `Datasets/Originais/`; preparation produces `Datasets/NewDataset`, `Datasets/DatasetFaces` and `Datasets/Folds`. Video paths are configured separately in the video notebook. These preparation scripts can recreate directories, overwrite images and alter metadata. Do not execute or import them as smoke tests. Check inputs and protect existing prepared data before an explicitly requested preparation run.

Training accepts an existing configuration:

```bash
python train.py --config models/configs/config_NCNN.yaml
```

Review paths, device, split and workload before running. Training defaults to `experiments/` when `save_dir` is absent. Saved-model analysis uses the prediction/metrics notebooks and shared functions, not a `validate.py --config` command.

## AI agent workspace

`.agents/` is the exclusive in-repository workspace for temporary agent-generated content. It is local and untracked, excluded by the root `.gitignore`. Create subdirectories only when needed; do not add placeholders to empty directories.

```text
.agents/
├── plans/       # Task plans, implementation notes and checklists
├── scratch/     # Temporary working files and intermediate calculations
├── scripts/     # Disposable task-support scripts
├── reports/     # Agent audits, validation reports and summaries
├── logs/        # Diagnostic logs and execution traces
├── cache/       # Temporary cached content
├── artifacts/   # Intermediate generated files
├── backups/     # Temporary pre-edit backups
└── sessions/    # Resumable task state and working notes
```

## Workspace isolation rules

1. Store all temporary files created by agents in `.agents/`.
2. Store disposable scripts in `.agents/scripts/`.
3. Store task-plan files in `.agents/plans/`.
4. Store agent-generated audits, validation reports and summaries in `.agents/reports/`.
5. Store diagnostic logs and traces in `.agents/logs/`.
6. Store intermediate outputs in `.agents/artifacts/` and temporary calculations in `.agents/scratch/`.
7. Store temporary backups in `.agents/backups/`, caches in `.agents/cache/` and resumable notes in `.agents/sessions/`.
8. Never create disposable files in the repository root.
9. Never leave temporary scripts beside production modules.
10. Never stage or commit `.agents/` contents, including with forced Git additions.

Configure external tools to write temporary files into `.agents/` when supported. Examples include `PYTHONPYCACHEPREFIX` for Python bytecode, `MPLCONFIGDIR` for Matplotlib and explicit cache/output arguments. Limit environment overrides to the task and restore previous values afterward.

If a tool cannot redirect an output, document the location and reason in the local task report and clean up only files created by the agent's run. For example, existing tests use self-cleaning temporary directories beneath `tests/`; invoking them is a documented tool-location exception, not permission to add disposable files there. Preserve all pre-existing user files and caches.

Keep substantial working plans and useful local reports in the workspace; remove agent-owned disposable materials when no longer useful. Do not recursively delete an existing workspace without checking ownership and the resolved target path.

## Temporary files versus intentional deliverables

Temporary analysis scripts, trial implementations, debug logs, intermediate CSVs/figures, notes, plans, generated audits/summaries, backups and disposable notebook copies belong in `.agents/`.

Intentional deliverables belong in the appropriate tracked locations: production modules, permanent tests, explicitly requested documentation, README edits, notebook edits, reusable utilities and authorized experiment configurations. Do not relocate legitimate source into `.agents/` because an agent created it, or hide permanent reusable code in the workspace.

Authorized research runs retain their configured dataset, checkpoint and final-output locations. Temporary task-support materials still belong in `.agents/`. Preserve established output paths and use timestamped task/run subdirectories where compatible to prevent overwriting. Verify ignore behavior for new output directories rather than assuming they are ignored; `experiments/` and `outputs/` are excluded today, but `runs/` is not.

## Jupyter notebook protection

- Do not execute notebooks unless required by the explicitly authorized task; never regenerate outputs during documentation-only work.
- For notebook documentation tasks, edit Markdown cells only. Preserve executable cells, cell identifiers, metadata and existing outputs.
- Avoid unrelated JSON reformatting or output clearing. Keep temporary notebook copies inside `.agents/`.
- After edits, parse the notebook JSON and validate its format with `nbformat` when available. Compare against the Git baseline to verify only intended cells/fields changed; report unavailable schema validation rather than claiming it passed.
- Avoid committing large generated outputs and inspect changed outputs for patient information before sharing.

## Documentation rules

- Write professional scientific English and base technical statements on actual code.
- Preserve accurate citations and research references; verify Markdown links and any newly introduced external references.
- Distinguish implemented methods, proposed research, historical results and known limitations. Document assumptions and avoid unsupported clinical claims.
- Keep documentation synchronized with actual commands, paths and behavior; do not invent missing flags or utilities.
- For README-only tasks, do not modify source code. For notebook documentation tasks, do not modify executable code.

## Coding standards and configuration

- Use clear Python functions, consistent naming and cohesive modules. Prefer type annotations for public functions and core utilities and explicit arguments over global state.
- Prefer existing utilities and `pathlib.Path`; do not add hardcoded absolute paths. Centralize new path logic only when needed, without unrelated refactors.
- Preserve stable imports and backward-compatible CLI behavior. Retain existing flags or supply documented deprecation warnings when authorized changes require replacements.
- New experiment settings belong in YAML under `models/configs/`. Use safe defaults and validate missing/invalid fields early with helpful exceptions.
- Prefer concise `logging` for long-running scripts. Never log patient-identifying details, credentials or sensitive local paths.
- New or modified I/O paths must fail clearly when required data is absent, rather than silently producing partial results. Existing deficiencies should be reported without expanding the task to fix them.
- Preserve current random seed behavior for Python, NumPy, PyTorch and CUDA. Add determinism where feasible for new functionality; changes to existing experimental randomness require scientific authorization.
- Use the recorded environment instructions in README and inspect `requirements.txt` and `environment.yml` before adding dependencies. Prefer the existing scientific Python stack; add dependencies only when necessary, update the relevant dependency manifests and explain why.

## Scientific computation rules

Preserve model architectures, preprocessing, hyperparameters, experimental configurations, checkpoint selection, subject-level partitions, metric definitions, uncertainty/calibration calculations, attribution methods, statistical evaluation and existing seed behavior. Any change affecting these requires explicit authorization and validation appropriate to the affected findings.

Do not silently fix apparent scientific inconsistencies. Report the evidence and propose a targeted correction. Subject-grouped folds require membership checks; the separate image-level `TrainAll` split must not be described as guaranteeing subject independence. Preserve the distinction between calibration fitting data, checkpoint-selection data and independent evaluation data.

Record sufficient reproducibility details for authorized experiments without exposing identifying data. Public code alone does not establish reproducibility when permissioned data, compatible checkpoints or historical assets are unavailable.

## Dataset privacy and security

UNIFESP, iCOPE and iCOPEvid resources may be permissioned. Local availability does not authorize external transmission. Agents must not:

- Commit patient images, videos, restricted datasets, large checkpoints or sensitive generated outputs.
- Copy clinical data into reports, publish identifiable metadata or expose sensitive paths or credentials in commits.
- Upload restricted data to external services without explicit authorization and applicable permission.
- Generate examples that reproduce private patient information.
- Use real patient data in debugging reports without explicit authorization and appropriate handling; never include patient-identifying material in shared logs, tests, issues or PRs.

Prefer synthetic, anonymized or metadata-only inputs for checks, and assess re-identification risk before sharing. Apply privacy rules to ignored workspace files as well as tracked files. Document legitimate checkpoint acquisition and local placement instead of committing binaries. Respect dataset permissions and third-party asset terms described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Agent instruction discovery

Inspect applicable nested `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `.cursor/rules/`, `.github/copilot-instructions.md` and similar tool configuration before editing. Persistent instructions needed for automatic discovery must remain in their tool-supported locations; temporary working files belong in `.agents/`.

Use this root manual as the canonical repository-wide guidance. Do not create duplicate instruction files without a demonstrated need, or move/delete tool instructions automatically. Preserve valid scoped rules and resolve conflicts against task instructions and applicable instruction precedence.

## Mandatory agent workflow and verification

### 1. Inspect

Read applicable instructions; check the branch, commit, Git status and staged changes; inspect relevant source, documentation, dependencies and configurations. Identify task scope, user-owned files, data/privacy constraints and potential scientific effects. Inventory temporary-file candidates without reading restricted data unnecessarily.

### 2. Plan

Define intended behavior and affected files, verification steps and scientific risks. Save substantial working plans in `.agents/plans/`. Distinguish permanent deliverables from temporary support files before creating them.

### 3. Execute

Apply focused changes within the authorized scope, use `.agents/` for temporary work and preserve unrelated user changes. Avoid unnecessary dependencies, notebook execution or refactoring. Follow the user's branch/commit instructions; do not merge or publish work without authorization.

### 4. Validate

- Run appropriate static checks, available targeted tests and `git diff --check`. Review the complete task diff and staged diff, inspect modified files and verify that unrelated paths were not changed.
- For Python changes, compile relevant repository source with `python -m compileall` and run a safe synthetic/missing-data check. Redirect bytecode to `.agents/cache/`; exclude datasets, environments and temporary workspace code from source checks. A tracked-source compilation also provides a safe syntax baseline for governance tasks without importing preparation scripts.
- Use existing `unittest` checks where relevant. The dataset-free check can be selected with `python -m unittest discover -s tests -p test_pain_sign_layout.py -k test_missing_dataset_has_clear_error`. Do not launch destructive data preparation or invent an unsupported `--dry-run` flag.
- For new dataset-independent logic, add lightweight tests using existing tooling; introduce another framework only when justified. Documentation-only tasks require documentation/scope checks and do not automatically require scientific execution.
- For workspace/ignore changes, run `git check-ignore -v` on `.agents/logs/example.log` and representative paths in `.agents/scripts/`, `.agents/reports/`, `.agents/artifacts/` and `.agents/backups/`. Confirm the shared `.gitignore` supplies the rule, not just `.git/info/exclude` or a global exclusion. Test new temporary-file patterns and representative intentional source/configuration/notebook paths.
- Run `git ls-files .agents/` and expect no tracked files. If any are tracked, evaluate and report their role; do not automatically untrack or move them. Ignore rules do not remove already tracked files.
- Review instruction consistency, link targets, actual commands and architecture. Report missing dependencies or pre-existing failures accurately; do not install packages or fix scientific code merely to make an unrelated task's checks pass.

### 5. Clean up

Remove only disposable agent-owned files that are no longer needed. Preserve useful local reports under `.agents/` and pre-existing user files. For candidates elsewhere, preserve permanent files, migrate only clearly identified untracked disposable agent files, and report uncertain ownership. Propose removal of tracked temporary files instead of silently deleting or moving them.

Check `git status --short`, workspace tracking and staged paths. Stage explicit intended deliverables only; never force-add ignored workspace contents. Confirm temporary files are absent from the commit and tool-location exceptions were cleaned up.

### 6. Report

Summarize changes, modified files, validation results, limitations and unresolved issues. For housekeeping tasks, list candidates identified, moved, preserved or flagged, and confirm whether scientific code, notebooks, datasets, configurations or results changed. For behavior changes, explain what changed, why and how to reproduce/verify it. Do not claim successful validation for unperformed checks.

These instructions and `.gitignore` establish workflow conventions and default Git exclusion, not runtime access controls or CI enforcement. When uncertain, inspect README and existing code, choose minimal reversible changes, document assumptions and seek clarification only for decisions that cannot be resolved from evidence or existing authorization.
