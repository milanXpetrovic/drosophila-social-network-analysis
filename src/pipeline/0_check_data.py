# TO START ENTER PATH OR POPULATION ID
# %%
import yaml
import sys

sys.path.append('/home/milanpetrovic/my_module/src')

import my_module as mm


CONFIG = '/home/milanpetrovic/my_module/configs/main.yaml'

with open(CONFIG) as f:
    config = yaml.safe_load(f)


# check_nans()


valid_rows_and_columns = mm.check_if_valid_rows_and_columns(
    config['raw_data_path'], config['file_extension'],
    config['validation_columns'], config['video_fps'],
    config['video_length_sec'])


# main()
# """
# In yaml sets value of valid data to TRUE if all above foos work
# """
# RETURNS valid_data = True
