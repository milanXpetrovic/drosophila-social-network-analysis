import yaml
import sys
import my_module as mm

sys.path.append('/home/milanpetrovic/my_module/src')

CONFIG = '/home/milanpetrovic/my_module/configs/main.yaml'

with open(CONFIG) as f:
    config = yaml.safe_load(f)

valid_rows_and_columns = mm.check_if_valid_rows_and_columns(
    config['raw_data_path'], config['file_extension'],
    config['validation_columns'], config['video_fps'],
    config['video_length_sec'])
