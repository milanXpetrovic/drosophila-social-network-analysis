#%%
from re import X
import yaml
import numpy as np
import pandas as pd 
import package_functions as pf

CONFIG = '../config.yaml'

with open(CONFIG) as f:
    config = yaml.safe_load(f)

INPUT_DATA = '../data/preproc/0_0_preproc_data'
SAVE_PATH = '../data/preproc/0_2_distances_between_flies_matrix'

experiments = pf.load_multiple_folders(INPUT_DATA)
for pop_name, path in experiments.items():  

    list_of_df = pf.load_dfs_to_list(path)
    dist_between_all = pf.distances_between_all_flies(list_of_df).round()

    dist_between_all.to_csv(SAVE_PATH + '/' + pop_name + '.csv')

    import sys
    sys.exit()
