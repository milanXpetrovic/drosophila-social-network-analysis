# get_adjacency_matrix():
# get_edge_list_with_timestamps(raw_data):
# get_path_features(raw_data):
# %%

sys.path.append('/home/milanpetrovic/my_module/src')
import pandas as pd
import my_module as mm
import sys
import yaml
import numpy as np 

CONFIG = '/home/milanpetrovic/my_module/configs/main.yaml'

with open(CONFIG) as f:
    config = yaml.safe_load(f)

files_to_check = mm.load_files_from_folder(
    config['raw_data_path'], config['file_extension'])


files = []
for n,p in files_to_check.items():
    df = pd.read_csv(p)
    files.append(df)


final_df = pd.DataFrame()

for i in range(len(files)):

    df1 = files[i]

    next_flie = i + 1

    if next_flie <= len(files):
        for j in range(next_flie, len(files)):
            df2 = files[j]

            df = pd.concat([df1['X#wcentroid (cm)'], df1['Y#wcentroid (cm)'],
                            df2['X#wcentroid (cm)'], df2['Y#wcentroid (cm)']], axis=1)

            df.columns = ['pos_x1', 'pos_y1', 'pos_x2', 'pos_y2']

            df['x_axis_dif'] = (df['pos_x1'] - df['pos_x2']).abs()
            df['y_axis_dif'] = (df['pos_y1'] - df['pos_y2']).abs()

            name = str(i+1) + ' ' + str(j+1)
            final_df[name] = np.sqrt(np.square(df['x_axis_dif']) +
                                        np.square(df['y_axis_dif']))







