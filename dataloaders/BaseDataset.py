"""
Module for a unified Dataset that supports optional soft labels and caching.

This Dataset applies preset transformations based on the provided model name and
loads images from a directory. Optionally, it reads a CSV file to compute soft labels.
"""

import os
import glob
import gc
import re
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from dataloaders.presets import PresetTransform, SoftLabel


def safe_glob(path_pattern):
    # Escape special glob characters ([, ]) by wrapping them in brackets
    escaped_pattern = re.sub(r'([\[\]])', r'[\1]', path_pattern)
    return glob.glob(escaped_pattern)

class BaseDataset(Dataset):
    """
    Dataset for loading image datasets with optional soft labels and caching.

    Parameters
    ----------
    model_name : str
        The name of the model. Determines the image transformations applied.
    img_dir : str
        The directory where the images are located.
    soft : str, optional
        Soft label method to use (e.g., 'SIGMOID', 'STEP', 'LINEAR'). Use 'None'
        (case insensitive) for binary labels. Default is 'None'.
    cache : bool, optional
        If True, caches all images and labels in RAM for faster training. Default is False.
    """

    def __init__(self, model_name: str, img_dir: str, soft: str = 'None', cache: bool = False) -> None:
        self.model_name = model_name.upper()
        self.img_dir = img_dir
        self.cache = cache
        self.soft = soft.upper()

        # Determine file pattern based on soft labeling option.
        pattern = '*_UNIFESP_*.jpg' if self.soft != 'NONE' else '*.jpg'
        self.img_paths = sorted(safe_glob(os.path.join(self.img_dir, pattern)))
        if not self.img_paths:
            raise ValueError(f"No images found in {self.img_dir} with pattern {pattern}")

        # Get the transformation preset for the given model.
        self.transform = PresetTransform(self.model_name).transforms

        if self.soft != "NONE":
            self.nfcs_df = pd.read_csv('iCOPE+UNIFESP_data.csv', usecols=['new_file_name', 'NFCS'])
            self.soft_labeler = SoftLabel(self.soft)

        # Initialize caches.
        self._images_cached = []
        self._labels_cached = []

        # Cache images and labels if requested.
        if self.cache:
            print('Caching images and labels, please wait...')
            for idx in range(len(self.img_paths)):
                self._images_cached.append(self._load_image(idx))
                self._labels_cached.append(self._load_label(idx))

        

    def _load_image(self, idx: int):
        """
        Load and transform the image at the given index.
        """
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        return self.transform(image)

    def _load_label(self, idx: int):
        """
        Load the label for the image at the given index.
        """
        filename = os.path.basename(self.img_paths[idx])

        # Remove augmented prefix if present.
        if 'AUG' in filename:
            filename = '_'.join(filename.split('_')[2:])

        # Use soft labeling if specified.
        if self.soft != 'NONE':
            row = self.nfcs_df[self.nfcs_df['new_file_name'] == filename]
            if row is None:
                raise ValueError(f"Label not found for {filename}")
            nfcs = row['NFCS'].values[0]
            # TODO: Handle cases where NFCS is NaN or not found.
            classe = filename.split('_')[-1].split('.')[0].lower()

            if classe == 'pain' and nfcs < 3:
                nfcs = 5

            elif classe == 'nopain' and nfcs >= 3:
                nfcs = 0
           
            label = self.soft_labeler.get_soft_label(nfcs)
        else:
            # Binary labeling: 1 for 'pain', 0 otherwise.
            label = 0 if 'nopain' in filename.split('_')[-1] else 1

        return label

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        if self.cache:
            image = self._images_cached[idx]
            label = self._labels_cached[idx]
        else:
            image = self._load_image(idx)
            label = self._load_label(idx)

        return {'image': image, 'label': label}

    def __del__(self):
        del self._labels_cached
        del self._images_cached
        gc.collect()