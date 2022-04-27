import os
import pandas as pd

import package_functions as hf

WINDOW_SIZE = 3

DATA_PATH = '../2_pipeline/0_0_preproc_data/out'
SAVE_PATH = '../2_pipeline/0_1_get_path_features/out'

if not (os.path.exists(SAVE_PATH + '/window_size_' + str(WINDOW_SIZE))):
    os.mkdir(SAVE_PATH + '/window_size_'+ str(WINDOW_SIZE))

experiments = hf.load_multiple_folders(DATA_PATH)

for pop_name, path in experiments.items():  
    fly_dict = hf.load_files_from_folder(path)
    
    if not os.path.exists(SAVE_PATH + '/window_size_' + str(WINDOW_SIZE) + '/' + pop_name):
        os.mkdir(SAVE_PATH + '/window_size_' + str(WINDOW_SIZE) + '/' + pop_name)
    
    for fly_name, path in fly_dict.items(): 
        df = pd.read_csv(path, index_col=0)
        df = hf.get_path_values(df, WINDOW_SIZE)
        
        values = ['str_index', 'vel', 'ang_vel', 'dist_to_wall', 'ori', 
              'step', 'spl', 'rpl', 'abs_change_x', 'abs_change_y']
        
        for value in values:   
            hf.df_descriptor(df, value, WINDOW_SIZE)     
        
        df = df.iloc[WINDOW_SIZE:,44:].reset_index(drop=True)
        
        df.to_csv(SAVE_PATH + '/window_size_' + str(WINDOW_SIZE) + '/' +
                  pop_name + '/' + fly_name)


