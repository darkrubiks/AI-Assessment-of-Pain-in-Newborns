"""
create_dataset.py

Author: Leonardo Antunes Ferreira
Date: 19/12/2022

This code is responsible for creating a new merged iCOPE and UNIFESP database
together with the Perception Heatmaps from UNIFESP. The files are renamed to the
following pattern: {ID}_{DATASET}_{SUBJECT}_{CLASSIFICATION}.
"""
import os
from shutil import copyfile, rmtree

import pandas as pd
from tqdm import tqdm

from utils.utils import create_folder

iCOPE_UNIFESP_data = pd.read_csv("iCOPE+UNIFESP_data.csv")
UNIFESP_percep_data = pd.read_csv("UNIFESP_percep_heatmaps.csv")

path_new_dataset = os.path.join("Datasets", "NewDataset")

try:
    rmtree(path_new_dataset)
except:
    pass

print("Creating directory...")
create_folder(path_new_dataset)
create_folder(os.path.join(path_new_dataset, "Images"))
create_folder(os.path.join(path_new_dataset, "Heatmaps"))

print("Creating images directory...")
for _, row in tqdm(iCOPE_UNIFESP_data.iterrows()):
    src_file = os.path.join(row["file_name"])
    dst_file = os.path.join(path_new_dataset, "Images", row["new_file_name"])
    copyfile(src_file, dst_file)

print("Creating heatmaps directory...")
for _, row in tqdm(UNIFESP_percep_data.iterrows()):
    src_file = os.path.join(row["heatmap_file_name"])
    dst_file = os.path.join(path_new_dataset, "Heatmaps", row["new_percep_file_name"])
    copyfile(src_file, dst_file)
