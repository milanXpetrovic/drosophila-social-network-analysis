#%%
import os
import yaml
import pandas as pd
import package_functions as pf

CONFIG = '../config.yaml'

with open(CONFIG) as f:
    config = yaml.safe_load(f)

EXPERIMENT_DURATION = config['EXPERIMENT_DURATION'] #experiment duration time must be in seconds
FPS = config['FPS']
DATAFRAME_LEN = EXPERIMENT_DURATION * FPS

RAW_DATA = config['RAW_DATA']

SAVE_PATH = '../data/preproc/0_0_preproc_data'

experiments = pf.load_multiple_folders(RAW_DATA)

for pop_name, path in experiments.items():  

    if not pf.check_data(path):
        continue    
    
    if not os.path.exists(SAVE_PATH + '/' + pop_name):
        os.mkdir(SAVE_PATH + '/' + pop_name)
        
    fly_dict = pf.load_files_from_folder(path)
    
    #pf.inspect_population_coordinates(path, pop_name)
    
    min_x, min_y = pf.find_pop_mins(path)
    
    for fly_name, path in fly_dict.items(): 
        
        df = pd.read_csv(path, index_col=0, nrows=DATAFRAME_LEN)
        df = pf.prepproc(df, min_x, min_y)
        #df = pf.round_coordinates(df, decimal_places=0)
        
        df.to_csv(SAVE_PATH + '/' + pop_name + '/' + fly_name)
    
