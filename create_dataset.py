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
import logging
import pandas as pd

from utils.utils import create_folder

# Configure native logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Read CSV files
iCOPE_UNIFESP_data = pd.read_csv("iCOPE+UNIFESP_data.csv")
UNIFESP_percep_data = pd.read_csv("UNIFESP_percep_heatmaps.csv")

path_new_dataset = os.path.join("Datasets", "NewDataset")

# Remove existing dataset directory if it exists
try:
    rmtree(path_new_dataset)
    logger.info(f"Removed existing directory: {path_new_dataset}")
except Exception as e:
    logger.warning(f"Could not remove directory {path_new_dataset}: {e}")

logger.info("Creating dataset directories...")
create_folder(path_new_dataset)
create_folder(os.path.join(path_new_dataset, "Images"))
create_folder(os.path.join(path_new_dataset, "Heatmaps"))
logger.info(f"Created directories: {path_new_dataset}, Images, Heatmaps")

logger.info("Copying image files...")
for idx, row in iCOPE_UNIFESP_data.iterrows():
    src_file = os.path.join(row["file_name"])
    dst_file = os.path.join(path_new_dataset, "Images", row["new_file_name"])
    copyfile(src_file, dst_file)
logger.info("Completed copying image files.")

logger.info("Copying heatmap files...")
for idx, row in UNIFESP_percep_data.iterrows():
    src_file = os.path.join(row["heatmap_file_name"])
    dst_file = os.path.join(path_new_dataset, "Heatmaps", row["new_percep_file_name"])
    copyfile(src_file, dst_file)
logger.info("Completed copying heatmap files.")
