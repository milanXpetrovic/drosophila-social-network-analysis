
# %%
import pandas as pd
import my_module as mm
import sys
import yaml
sys.path.append('/home/milanpetrovic/my_module/src')

import matplotlib.pyplot as plt

CONFIG = '/home/milanpetrovic/my_module/configs/main.yaml'

DATA_PATH = '/home/milanpetrovic/my_module/data/preproc/1_0_preproc_data/25_03_2022'

with open(CONFIG) as f:
    config = yaml.safe_load(f)


# distribution of column values for individuals and group
files_to_check = mm.load_files_from_folder(
    DATA_PATH, config['file_extension'])

for file_name, file_path in files_to_check.items():
    df = pd.read_csv(file_path)

    for col in df.columns:
        print(col)
        df.hist(column=col)
        plt.show()

    sys.exit()
