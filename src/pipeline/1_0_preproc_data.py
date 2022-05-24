
# %%
import pandas as pd
import my_module as mm
import sys
import yaml

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

    name = SAVE_PATH + '/' + config['pop_name'] + '/' + file_name
    df.to_csv(name)


# preproc_data(raw_data):
# inspec_raw_data():
# FOOS FOR DATA PRE PROCESING
# NORMALIZE, ROUND VALUES, ETC.
