#%%
from distutils.command.config import config
import os
import yaml
import pandas as pd

import package_functions as pf

CONFIG = '../config.yaml'

with open(CONFIG) as f:
    config = yaml.safe_load(f)

WINDOW_SIZE = config['WINDOW_SIZE']

DATA_PATH = '../data/preproc/0_0_preproc_data'
SAVE_PATH = '../data/preproc/0_1_get_path_features'

if not (os.path.exists(SAVE_PATH + '/window_size_' + str(WINDOW_SIZE))):
    os.mkdir(SAVE_PATH + '/window_size_'+ str(WINDOW_SIZE))

experiments = pf.load_multiple_folders(DATA_PATH)

for pop_name, path in experiments.items():  
    fly_dict = pf.load_files_from_folder(path)
    
    if not os.path.exists(SAVE_PATH + '/window_size_' + str(WINDOW_SIZE) + '/' + pop_name):
        os.mkdir(SAVE_PATH + '/window_size_' + str(WINDOW_SIZE) + '/' + pop_name)
    
    for fly_name, path in fly_dict.items(): 
        df = pd.read_csv(path, index_col=0)
        df = pf.get_path_values(df, WINDOW_SIZE)
        
        values = ['str_index', 'vel', 'ang_vel', 'dist_to_wall', 'ori', 
              'step', 'spl', 'rpl', 'abs_change_x', 'abs_change_y']
        
        for value in values:   
            pf.df_descriptor(df, value, WINDOW_SIZE)     
        
        df = df.iloc[WINDOW_SIZE:,44:].reset_index(drop=True)
        
        df.to_csv(SAVE_PATH + '/window_size_' + str(WINDOW_SIZE) + '/' +
                  pop_name + '/' + fly_name)

    import sys
    sys.exit()

