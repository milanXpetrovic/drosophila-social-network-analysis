import pandas as pd 
import package_functions as hf

DATA_PATH = '../2_pipeline/0_1_get_path_features/out/window_size_3'
SAVE_PATH = '../2_pipeline/2_1_get_fly_distance_traveled/out'

experiments = hf.load_multiple_folders(DATA_PATH)

for pop_name, path in experiments.items():  
    fly_dict = hf.load_files_from_folder(path)
    distances = {}
    
    for fly_name, path in fly_dict.items(): 
        df = pd.read_csv(path, usecols=['step'])
        walked_path = int(df['step'].sum())
        distances.update({fly_name.replace('.csv', '') : walked_path})
    
    df = pd.DataFrame(distances, ['distance'])
    df = df.T
    
    df.to_csv(SAVE_PATH + '/' + pop_name + '.csv')
    


