
# %%
import math
import numpy as np
import yaml
import sys
import my_module as mm
import pandas as pd
sys.path.append('/home/milanpetrovic/my_module/src')


CONFIG = '/home/milanpetrovic/my_module/configs/main.yaml'
SAVE_PATH = '../../data/preproc/1_0_preproc_data'


with open(CONFIG) as f:
    config = yaml.safe_load(f)

files_to_check = mm.load_files_from_folder(
    config['raw_data_path'], config['file_extension'])

for file_name, file_path in files_to_check.items():
    df = pd.read_csv(file_path)
    df = mm.interpolate_inf_values(df, interpolation_method='linear')

    df = mm.convert_radians_to_degrees(df, config['angle_column'])

    name = SAVE_PATH + '/' + config['pop_name'] + '/' + file_name
    df.to_csv(name)

# %%

PATH = '/home/milanpetrovic/my_module/data/preproc/1_0_preproc_data/25_03_2022/2022-03-25_11-46_fly_1.csv'

df = pd.read_csv(PATH).round(decimals=2)
df = df[['ANGLE', 'X (cm)', 'Y (cm)']]
df.reset_index(inplace=True)


df = df.iloc[:-1, :]
df = df[['X (cm)', 'Y (cm)']]
df.columns = ['x', 'y']
# df1.reset_index()

res = df

res['x_axis_dif'] = (0 - res['x']).abs()
res['y_axis_dif'] = (0 - res['y']).abs()

res['distance'] = np.sqrt(np.square(res['x_axis_dif']) +
                          np.square(res['y_axis_dif']))

res['angle'] = np.sin(res['y_axis_dif']/res['distance'])

res = mm.convert_radians_to_degrees(res, 'angle')

chck = res.iloc[::100,:]
# %%
sys.path.append('/home/milanpetrovic/my_module/src')


# ax = angles.plot.hist()
# preproc_data(raw_data):
# inspec_raw_data():
# FOOS FOR DATA PRE PROCESING
# NORMALIZE, ROUND VALUES, ETC.

# %%
