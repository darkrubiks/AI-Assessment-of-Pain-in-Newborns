"""
create_dataset.py

Author: Leonardo Antunes Ferreira
Date:19/12/2022

This code is responsible for creating a new merged iCOPE and UNIFESP database
together with the Perception Heatmaps from UNIFESP. The files are renamed to the
following pattern: {ID}_{DATASET}_{SUBJECT}_{CLASSIFICATION}.
"""
import os
import pandas as pd
from shutil import copyfile, rmtree
from tqdm import tqdm


iCOPE_UNIFESP_data = pd.read_csv('iCOPE+UNIFESP_data.csv')
UNIFESP_percep_data = pd.read_csv('UNIFESP_percep_heatmaps.csv')

path_original_dataset = os.path.join('Datasets','Originais')
path_new_dataset = os.path.join('Datasets','NewDataset')

try:
    rmtree(path_new_dataset)
except:
    pass

print('Criando diretorio')
os.makedirs(path_new_dataset)
os.makedirs(os.path.join(path_new_dataset, 'Images'))
os.makedirs(os.path.join(path_new_dataset, 'Heatmaps'))

print('Criando diretorio de imagens')
for _, row in tqdm(iCOPE_UNIFESP_data.iterrows()):
    src_file = os.path.join(path_original_dataset, row['dataset'], row['file_name'])
    dst_file = os.path.join(path_new_dataset, 'Images', row['new_file_name'])

    copyfile(src_file, dst_file)

print('Criando diretorio de heatmaps')
for _, row in tqdm(UNIFESP_percep_data.iterrows()):
    src_file = os.path.join(path_original_dataset, 'PERCEP_HEATMAPS', row['heatmap_file_name'])
    dst_file = os.path.join(path_new_dataset, 'Heatmaps', row['new_file_name'])

    copyfile(src_file, dst_file)