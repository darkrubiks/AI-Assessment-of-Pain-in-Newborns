# AI Assessment of Pain in Newborns

Research code for assessing neonatal pain from facial images, developed at **Centro Universitário FEI in partnership with UNIFESP**. The repository combines deep learning classifiers with uncertainty estimation, probability calibration, explainable AI and video analysis.

**Research use:** The models and temporal signals are experimental and have not been validated or approved for clinical use.

## Contents

- [Getting started](#getting-started)
- [Datasets and preparation](#datasets-and-preparation)
- [Training and evaluation](#training-and-evaluation)
- [Notebooks and analyses](#notebooks-and-analyses)
- [Repository guide](#repository-guide)
- [Reproducibility](#reproducibility)
- [Results and publications](#results-and-publications)
- [Citation and contact](#citation-and-contact)
- [License](#license)

## Getting started

Run commands from the repository root. The recorded environment uses **Python 3.10.16** and **PyTorch 2.6 with CUDA 12.6** on Windows. A compatible NVIDIA GPU is recommended for training and model-based analyses; supplied configurations select CUDA. Other platforms require adapted dependencies and paths.

```bash
git clone --branch develop https://github.com/darkrubiks/AI-Assessment-of-Pain-in-Newborns.git
cd AI-Assessment-of-Pain-in-Newborns

conda create --name neonatal-pain python=3.10.16 pip
conda activate neonatal-pain
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
```

The recorded Conda environment is also available in [environment.yml](environment.yml). For XAI and statistical notebooks using the pip setup, install:

```bash
python -m pip install captum==0.8.0 statsmodels==0.14.4
```

Open notebooks in your IDE or install Jupyter:

```bash
python -m pip install notebook
python -m notebook
```

Check the environment before preparing data:

```bash
python -m pip check
python train.py --help
```

Some notebooks need additional packages or local analysis files; read their introductory notes before running cells. VGGFace requires `models/weights/VGG_face_original_model.pt`, obtained from the authors. ViT initialization requires torchvision pretrained weights. Trained checkpoints are local resources and are not included in the repository.

## Datasets and preparation

The experiments use **iCOPE and UNIFESP facial images**, UNIFESP perception heatmaps and, for temporal analysis, **iCOPEvid**. Obtain permission and data from the respective owners. Do not publish patient images, videos, identifying metadata or notebook outputs containing them.

Place the original still-image resources in the layout referenced by the metadata:

```text
Datasets/
└── Originais/
    ├── iCOPE/pilot/
    ├── UNIFESP/
    └── PERCEP_HEATMAPS/
```

Keep original filenames and check the source paths in [iCOPE+UNIFESP_data.csv](iCOPE%2BUNIFESP_data.csv) and [UNIFESP_percep_heatmaps.csv](UNIFESP_percep_heatmaps.csv). Video paths are configured separately in [7_icopevid.ipynb](7_icopevid.ipynb).

Once the files and face-detector assets are available, run:

```bash
python create_dataset.py
python face_detection.py
python leave_some_subject_out.py
python data_augmentation.py
```

These steps standardize filenames, crop faces, create folds and augment training images. Generated data is stored under `Datasets/NewDataset`, `Datasets/DatasetFaces` and `Datasets/Folds`.

**Back up prepared data before rerunning:** these scripts recreate output directories, overwrite resized images and update the image metadata CSV. Confirm all required inputs before starting.

## Training and evaluation

Available models are **NCNN**, **VGGFace** and **ViT**. Choose an existing YAML from [models/configs/](models/configs) and review its model, dataset paths, device, training settings and output directory.

```bash
python train.py --config models/configs/config_NCNN.yaml
```

Each command trains one configuration on one split. The supplied NCNN example uses `TrainAll`; select the intended fold paths in your configuration for cross-validation. [0_train.ipynb](0_train.ipynb) schedules multiple training runs and ensemble members, so review its workload before executing it.

Training saves the best checkpoint, a copy of the configuration, logs and evaluation plots under the configured `save_dir` (default: `experiments/`).

For saved-model evaluation, use [1_create_data.ipynb](1_create_data.ipynb) to export predictions, then [2_validation_metrics.ipynb](2_validation_metrics.ipynb) for classification and calibration metrics. [validate.py](validate.py) provides shared evaluation functions and has no standalone validation CLI.

## Notebooks and analyses

Select the notebooks needed for your research question. They depend on prepared data, compatible checkpoints and saved predictions; they are not all intended to run in sequence. Read each notebook's Markdown notes for paths, prerequisites and known compatibility issues.

| Notebook | Purpose |
|---|---|
| [0_train.ipynb](0_train.ipynb) | Schedule training, target comparisons and ensemble runs. |
| [1_create_data.ipynb](1_create_data.ipynb) | Export predictions and fit calibration transformations. |
| [2_validation_metrics.ipynb](2_validation_metrics.ipynb) | Evaluate classification and probability calibration. |
| [3_uncertainty.ipynb](3_uncertainty.ipynb) | Study MC-dropout and ensemble uncertainty. |
| [4_calibration.ipynb](4_calibration.ipynb) | Compare saved calibration variants; some comparisons use outputs from `6_limits.ipynb`. |
| [5_XAI.ipynb](5_XAI.ipynb) | Generate attribution maps and compare facial regions. |
| [6_limits.ipynb](6_limits.ipynb) | Evaluate confidence/uncertainty thresholds and prediction coverage. |
| [7_icopevid.ipynb](7_icopevid.ipynb) | Analyse frame predictions and temporal pain-related signals. |
| [RGU_XAI_PAPER.ipynb](RGU_XAI_PAPER.ipynb) | Compare XAI methods, explanation agreement and perturbation responses. |
| [RGU_notebook.ipynb](RGU_notebook.ipynb) | Explore explanation-quality methods and concept analysis. |
| [graficos axuliares.ipynb](graficos%20axuliares.ipynb) | Create auxiliary research illustrations. |
| [lancet.ipynb](lancet.ipynb) | Explore historical thresholds and video analyses. |
| [teste_cassia.ipynb](teste_cassia.ipynb) | Explore historical attribution comparisons and face-recognition experiments. |

Uncertainty measures prediction variability, while calibration assesses whether probabilities agree with observed labels. Attribution maps help inspect model behavior; agreement between explanations does not establish their faithfulness or clinical significance. Temporal signals describe model predictions and are not independent pain annotations.

## Repository guide

| Location | Contents |
|---|---|
| [models/](models) and [models/configs/](models/configs) | Classifiers and experiment configurations. |
| [dataloaders/](dataloaders) | Image and video loading. |
| [uncertainty/](uncertainty) and [calibration/](calibration) | Prediction uncertainty and probability calibration. |
| [XAI/](XAI) | Explainability and facial-region analysis. |
| [utils/](utils) | Shared data, plotting and configuration helpers. |
| [train.py](train.py) and [validate.py](validate.py) | Training and evaluation functions. |
| [AGENTS.md](AGENTS.md) | Contribution, reproducibility and privacy guidance. |

## Reproducibility

Record the repository commit, environment, configuration, dataset revision, exact splits and checkpoints with each experiment. Public code alone is insufficient to reproduce every analysis: permissioned data, pretrained assets and some historical artifacts must be obtained separately.

- **Evaluation splits:** The fold script creates subject-grouped partitions, but their membership and stratification need checking. `TrainAll` splits images and does not guarantee subject independence. The training evaluation set also selects the checkpoint, so it is not an untouched final test set.
- **Calibration and thresholds:** Check which data was used to fit them. The calibration exporter currently fits on training predictions; independent evaluation is needed to assess generalization.
- **Historical notebooks:** Some cells depend on older models, helpers or artifact formats. Saved outputs do not establish that a notebook runs from start to finish with the current code.
- **Image versus video experiments:** Match preprocessing and model revisions to the publication. Still-image performance does not establish video performance or generalization to another population or hospital.

Keep datasets, checkpoints and generated outputs local. Review notebook outputs before sharing, even when their source files are tracked by Git.

## Results and publications

These papers describe the research associated with the project. Match the code revision, data and checkpoints to the paper you want to reproduce. Some methods belong to related work and are not implemented in this checkout.

1. **From pixels to pain scales: Robust evaluation of explainable AI in neonatal pain detection.** L. A. Ferreira, C. Pirie, G. A. S. Coutrin, L. P. Carlini, C. F. Moreno-Garcia, M. C. M. Barros, R. Guinsburg and C. E. Thomaz. *4th World Conference on eXplainable Artificial Intelligence (XAI 2026)*, Fortaleza, Brazil, 1–3 July **2026**, 22 pp. **Forthcoming:** the [author's bibliography](https://fei.edu.br/~cet/) lists this work as “to appear”; a final proceedings record and DOI have not been confirmed.

2. **Is this neonate feeling pain? Leveraging clinical knowledge towards high-precision Large Language Model-based neonatal pain assessment.** Lucas Pereira Carlini, Leonardo Antunes Ferreira, Gabriel de Almeida Sá Coutrin, Tatiany Marcondes Heiderich, Rita de Cássia Xavier Balda, Marina Carvalho de Moraes Barros, Ruth Guinsburg and Carlos Eduardo Thomaz. *Pediatric Research*, **100**(1), pp. 431–438, **2026** (first online **11 December 2025**). DOI: [10.1038/s41390-025-04669-8](https://doi.org/10.1038/s41390-025-04669-8). [Publisher record](https://www.nature.com/articles/s41390-025-04669-8); [bibliographic record](https://pubmed.ncbi.nlm.nih.gov/41381876/).

3. **Barriers and pathways to clinical translation of AI in automated neonatal pain assessment.** L. A. Ferreira, L. P. Carlini, G. A. S. Coutrin, L. F. Buzuti, R. G. Magalhaes Jr, R. N. Orsi, T. M. Heiderich, G. Cianciarullo, S. P. Gimenez, C. Pirie, C. F. Moreno-Garcia, M. C. M. Barros, R. Guinsburg and C. E. Thomaz. *I Workshop de Grandes Desafios da Computação Aplicada à Saúde, 26th Simpósio Brasileiro de Computação Aplicada à Saúde (SBCAS 2026)*, pp. 86–97, Ouro Preto, Brazil, 1–4 June **2026**. [Author's bibliography](https://fei.edu.br/~cet/); [conference programme](https://www.sbcas2026.com/programacao). A DOI has not been confirmed.

4. **Understanding Disagreement Between Humans and Machines in XAI: Robustness, Fidelity, and Region-Based Explanations in Automatic Neonatal Pain Assessment.** Craig Pirie, Leonardo Antunes Ferreira, Gabriel de Almeida Sá Coutrin, Lucas Pereira Carlini, Carlos Francisco Moreno-García, Marina Carvalho de Moraes Barros, Ruth Guinsburg, Carlos Eduardo Thomaz, Rafael Nobre and Nirmalie Wiratunga. *Explainable Artificial Intelligence (xAI 2025)*, CCIS **2580**, pp. 274–298, Springer, **2026** (first online **19 October 2025**). DOI: [10.1007/978-3-032-08333-3_13](https://doi.org/10.1007/978-3-032-08333-3_13). [Publisher record](https://link.springer.com/chapter/10.1007/978-3-032-08333-3_13). The citation year and author names follow the publisher record.

5. **Disclosing neonatal pain in real-time: AI-derived pain sign from continuous assessment of facial expressions.** Leonardo Antunes Ferreira, Lucas Pereira Carlini, Gabriel de Almeida Sá Coutrin, Tatiany Marcondes Heiderich, Rita de Cássia Xavier Balda, Marina Carvalho de Moraes Barros, Ruth Guinsburg and Carlos Eduardo Thomaz. *Computers in Biology and Medicine*, **189**, 109908, **2025**. DOI: [10.1016/j.compbiomed.2025.109908](https://doi.org/10.1016/j.compbiomed.2025.109908). [Bibliographic record](https://pubmed.ncbi.nlm.nih.gov/40024188/).

6. **Human vs machine towards neonatal pain assessment: A comprehensive analysis of the facial features extracted by health professionals, parents, and convolutional neural networks.** Lucas Pereira Carlini, Gabriel de Almeida Sá Coutrin, Leonardo Antunes Ferreira, Juliana do Carmo Azevedo Soares, Giselle Valério Teixeira Silva, Tatiany Marcondes Heiderich, Rita de Cássia Xavier Balda, Marina Carvalho de Moraes Barros, Ruth Guinsburg and Carlos Eduardo Thomaz. *Artificial Intelligence in Medicine*, **147**, 102724, **2024** (online November 2023). DOI: [10.1016/j.artmed.2023.102724](https://doi.org/10.1016/j.artmed.2023.102724). [Bibliographic record](https://pubmed.ncbi.nlm.nih.gov/38184347/).

7. **Convolutional Neural Networks for Newborn Pain Assessment Using Face Images: A Quantitative and Qualitative Comparison.** Gabriel A. S. Coutrin, Lucas P. Carlini, Leonardo A. Ferreira, Tatiany M. Heiderich, Rita C. X. Balda, Marina C. M. Barros, Ruth Guinsburg and Carlos E. Thomaz. *Medical Imaging and Computer-Aided Diagnosis (MICAD 2022)*, LNEE **810**, pp. 503–513, Springer; first online **20 December 2023**. DOI and [publisher record](https://link.springer.com/chapter/10.1007/978-981-16-6775-6_41): **10.1007/978-981-16-6775-6_41**. The event name contains 2022; it is not the online publication date.

8. **Revisiting N-CNN for Clinical Practice.** Leonardo Antunes Ferreira, Lucas Pereira Carlini, Gabriel de Almeida Sá Coutrin, Tatiany Marcondes Heideirich, Marina Carvalho de Moraes Barros, Ruth Guinsburg and Carlos Eduardo Thomaz. *Predictive Intelligence in Medicine (PRIME 2023)*, LNCS **14277**, pp. 231–240, Springer, **2023**. DOI and [publisher record](https://link.springer.com/chapter/10.1007/978-3-031-46005-0_20): **10.1007/978-3-031-46005-0_20**. Author spelling follows this publisher record.

9. **A Convolutional Neural Network-based Mobile Application to Bedside Neonatal Pain Assessment.** Lucas P. Carlini, Leonardo A. Ferreira, Gabriel A. S. Coutrin, Victor V. Varoto, Tatiany M. Heiderich, Rita C. X. Balda, Marina C. M. Barros, Ruth Guinsburg and Carlos E. Thomaz. *2021 34th SIBGRAPI Conference on Graphics, Patterns and Images*, IEEE, pp. 394–401, **2021**. DOI: [10.1109/SIBGRAPI54419.2021.00060](https://doi.org/10.1109/SIBGRAPI54419.2021.00060). [Proceedings contents](https://www.proceedings.com/content/061/061787webtoc.pdf); also referenced in [models/VGGFace.py](models/VGGFace.py).

10. **Mobile Convolutional Neural Network for Neonatal Pain Assessment.** Lucas P. Carlini, Leonardo A. Ferreira, Gabriel A. S. Coutrin, Victor V. Varoto, Tatiany M. Heiderich, Rita C. X. Balda, Marina C. M. Barros, Ruth Guinsburg and Carlos E. Thomaz. *LatinX in AI Research Workshop at CVPR*, **2021**, 4 pp. [Workshop paper](https://research.latinxinai.org/papers/cvpr/2021/pdf/9_CameraReady_09.pdf).

### Foundational references in the implementation

Modules identify their methodological sources: [N-CNN](https://doi.org/10.1109/IJCNN.2019.8851879), [Integrated Gradients](https://doi.org/10.48550/arXiv.1703.01365), [Grad-CAM](https://doi.org/10.48550/arXiv.1610.02391), and [Neonatal Face Mosaic](https://doi.org/10.5753/wvc.2021.18914). See the corresponding modules for adaptations.

## Citation and contact

Cite the publications relevant to the methods and results you use, together with the repository URL and exact commit. There is no tracked `CITATION.cff` or repository release DOI.

The project is developed and researched by [Leonardo Antunes Ferreira](https://www.linkedin.com/in/leonardoantunesferreira/), [Lucas Pereira Carlini](https://br.linkedin.com/in/lucas-pereira-carlini-947409161), [Gabriel de Almeida Sá Coutrin](https://www.linkedin.com/in/gabriel-coutrin/) and [Carlos Eduardo Thomaz](https://fei.edu.br/~cet/) at FEI, in partnership with UNIFESP. The publications credit the additional collaborators.

For questions, use the [issue tracker](https://github.com/darkrubiks/AI-Assessment-of-Pain-in-Newborns/issues) or the profiles above. Do not include patient data.

### Historical recognitions

These credits are retained as reported in the previous repository README:

- G. A. S. Coutrin, L. P. Carlini, L. A. Ferreira, V. V. Varoto and C. E. Thomaz: *Development of a Mobile Application for Automatic Recognition of Facial Expression of Pain in Newborns*, Municipal Week of Science, Technology and Innovation of Santo André, 8th edition, October 2022.
- L. P. Carlini, L. A. Ferreira, G. A. S. Coutrin, V. V. Varoto, R. Guinsburg and C. E. Thomaz: *Development of a Mobile Convolutional Neural Network for Neonatal Pain Assessment*, H-INNOVA Health Innovation Award, finalist group, 2nd edition, 2021.

## License

The repository's original software and associated documentation are licensed under the [MIT License](LICENSE), copyright (c) 2026 Leonardo Antunes Ferreira. Academic and commercial reuse are permitted under its terms, including retention of the copyright and license notices.

Permissioned datasets, patient materials, third-party pretrained weights and third-party components retain their separate terms and are not relicensed by this repository. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and asset guidance, and obtain any required permissions from the respective owners.

InsightFace code is MIT-licensed, but its supplied pretrained models have separate non-commercial research terms unless separately licensed. This repository's MIT license does not grant commercial rights to those models. Consult the [InsightFace license policy](https://github.com/deepinsight/insightface#license).
