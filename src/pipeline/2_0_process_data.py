# get_adjacency_matrix():
# get_edge_list_with_timestamps(raw_data):
# get_path_features(raw_data):
# %%

import numpy as np
import yaml
import sys
sys.path.append('/home/milanpetrovic/my_module/src')
import my_module as mm
import pandas as pd

CONFIG = '/home/milanpetrovic/my_module/configs/main.yaml'
PREPROC_DATA = '/home/milanpetrovic/my_module/data/preproc/1_0_preproc_data/25_03_2022'
SAVE_PATH = '/home/milanpetrovic/my_module/data/preproc/2_0_process_data/'

with open(CONFIG) as f:
    config = yaml.safe_load(f)

x_col_name = config['x_col_name']
y_col_name = config['y_col_name']
file_extension = config['file_extension']

# distances_in_time = mm.distance_in_time_for_individuals(
#     PREPROC_DATA, x_col_name, y_col_name, file_extension)

# distances_in_time = distances_in_time.round(decimals=2)
# distances_in_time.to_csv(SAVE_PATH + 'distances_in_time.csv')


