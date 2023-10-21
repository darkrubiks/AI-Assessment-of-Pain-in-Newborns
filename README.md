# **AI Assessment of Pain in Newborns**

## **Introduction**

Inside this repository you will find the code needed to replicate most of our research findings in ***Pain Assessment in Newborns with AI***. The main contributions of our work are:

* Development of AI models to automatically classify facial expressions of pain in newborns

* Understanding  of perceived facial areas by humans and machines during the task of analyzing pain in newborns

* The use of XAI (eXplainable AI) methods in the clinical practice

## **Table of Contents**

* [How to use this repository](#how-to-use-this-repository)
  * [Requirements](#requirements)
  * [Repository Structure](#repository-structure)
  * [Running the code](#running-the-code)
* [Models](#models)
* [Publications](#publications)
* [Awards](#awards)
* [Authors](#authors)

## **How to use this repository**

### **Requirements**

First make sure to install all requirements from `requirements.txt`.

To run most of the codes the iCOPE e UNIFESP datasets are required to be inside a folder `Datasets\Originais` with the original filenames from the dataset authors. To get access to these datasets you must ask permission from its own creators.

```bash
MESTRADO
├── ...
├── Datasets             # All of the datasets are going to be saved here
│   ├── COPE             # iCOPE images
│   ├── UNIFESP          # UNIFESP images
│   ├── PERCEP_HEATMAPS  # UNIFESP visual perception heatmaps
```

The file `iCOPE+UNIFESP_data.csv` contains all the necessary information to associate the iCOPE and UNIFESP original filenames to a new and standardized name, it also contains other information like the subjects number, face location, and facial landmarks location.

Like `iCOPE+UNIFESP_data.csv` the `UNIFESP_percep_heatmaps.csv` contains information about the UNIFESP visual perception heatmaps and its corresponding image.

### **Repository Structure**

All code inside folders like `dataloaders`, `models` and `XAI` can be called using Python imports:

```python
from models import VGGNB, NCNN
from XAI import IntegratedGradients
from dataloader import *
```

The main codes are on the root directory and can run from the command line or your favorite programming software.

### **Running the code**

**Follow the instructions in order!**

#### **1. [`create_dataset.py`](create_dataset.py)**

To run this code you will need the original datasets on `Datasets\Originais` and the `iCOPE+UNIFESP_data.csv` and `UNIFESP_percep_heatmaps.csv` files.
It will create a new folder `Datasets\NewDataset` with the subdirectories `Images` and `Heatmaps`. All images and heatmaps will be renamed to a standardized name from the `.csv` files mentioned before.

#### **2. [`face_detection.py`](face_detection.py)**

To run this code you will need the newly created dataset on `Datasets\NewDataset` and the `iCOPE+UNIFESP_data.csv`. You will also need to follow the instructions on [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package) to download and install the RetinaFace model. It will create a new folder `Datasets\Faces` with the face cropped from each image.

#### **3. [`leave_one_subject_out.py`](leave_some_subject_out.py)**

To run this code you will need the the `Datasets\Faces` folder and the `iCOPE+UNIFESP_data.csv`. It will create 10 folds with Train and Test sets using the leave-some-subject-out method, each fold will be stored on `Datasets\Folds\{fold_number}`

#### **4. [`data_augmentation.py`](data_augmentation.py)**

To run this code you will need the `Datasets\Folds` folder and the `iCOPE+UNIFESP_data.csv`. For each training image 20 new images will be created using data augmentation techniques. All images are resized to `512 x 512` and the facial landmarks are augmented as well and saved inside a folder `Datasets\Folds\0\Train\Landmarks` in the [pickle](https://docs.python.org/3/library/pickle.html) format.

#### **5. [`train.py`](train.py)**

After the above steps, you can run the training code. First make sure to create or use one of the existing `.yaml` configuration files inside `models\configs`.

Inside the `.yaml` file you can choose any optimizer from PyTorch and set its hyperparameters on the `optimizer_hyp` field. The same applies to Learning Rate Schedulers. See `models\configs\config_Tuned_NCNN.yaml` for a full example.

```text
python train.py --config models\configs\config_NCNN.yaml
```

## **Models**

| **Model**                | **Accuracy** | **Precision** | **Recall** | **F1**     | **Trained On**|
| :---:                    | :---:        | :---:         | :---:      | :---:      | :---:         |
| VGGNB                    | 85.2% ± 6%   | 87.1% ± 7%    | 79.6% ± 9% | 88.9% ± 6% | iCOPE+UNIFESP |
| NCNN-Tuned               | 82.9% ± 5%   | 85.6% ± 8%    | 75.4% ± 9% | 79.7% ± 6% | iCOPE+UNIFESP |
| NCNN                     | 78.7% ± 5%   | 83.2% ± 8%    | 69.0% ± 9% | 74.1% ± 6% | iCOPE+UNIFESP |

## **Publications**

* Ferreira, Leonardo A., et al. "Revisiting N-CNN for Clinical Practice." International Workshop on PRedictive Intelligence In MEdicine. Cham: Springer Nature Switzerland, 2023.

* Coutrin, Gabriel AS et al. "Convolutional neural networks for newborn pain assessment using face images: A quantitative and qualitative comparison." Proceedings of the 3rd International Conference on Medical Imaging and Computer-Aided Diagnosis, MICAD 2022. Springer LNEE, 2022.

* Carlini, Lucas P., et al. "A Convolutional Neural Network-based Mobile Application to Bedside Neonatal Pain Assessment." 2021 34th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI). IEEE, 2021.

## **Awards**

* G. A. S. Coutrin, L. P. Carlini, L. A. Ferreira, V. V. Varoto and C. E. Thomaz. Development of a Mobile Application for Automatic Recognition of Facial Expression of Pain in Newborns, Municipal Week of Science, Technology and Innovation of Santo André, 8th. edition, October 2022.

* L. P. Carlini, L. A. Ferreira, G. A. S. Coutrin, V. V. Varoto, R. Guinsburg and C. E. Thomaz. Developtment of a Mobile Convolutional Neural Network for Neonatal Pain Assessment, H-INNOVA Health Innovation Award (finalist group), 2nd edition, 2021.

## **Authors**

Developed and researched by:

* [Leonardo Antunes Ferreia](https://www.linkedin.com/in/leonardoantunesferreira/) - FEI - Repository Author
* [Lucas Pereira Carlini](https://br.linkedin.com/in/lucas-pereira-carlini-947409161) - FEI
* [Gabriel de Almeida Sá Coutrin](https://www.linkedin.com/in/gabriel-coutrin/) - FEI
* [Carlos Eduardo Thomaz](https://fei.edu.br/~cet/) - FEI

In partnership with UNIFESP
