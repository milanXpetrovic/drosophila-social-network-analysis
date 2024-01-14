#%%
import math
import os
import sys

import pandas as pd
import toml

from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]

CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    config = toml.load(file)

INPUT_DIR = f"/srv/milky/matlab/recordings/{TREATMENT}"
treatment = fileio.load_multiple_folders(INPUT_DIR)

OUTPUT_DIR = os.path.join(settings.INPUT_DIR, TREATMENT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for group_name, group_path in treatment.items():
    group_xlsx = fileio.load_files_from_folder(group_path, file_format=".xls")
    if not group_xlsx:
        continue
    else:
        OUTPUT_DIR = os.path.join(settings.INPUT_DIR, TREATMENT, group_name)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for xls_group_name, xls_group_path in  group_xlsx.items():
            print(xls_group_path)
            df = pd.read_excel(xls_group_path, sheet_name=None)
            
            for sheet_name, sheet_data in df.items():
                if sheet_name == "Sheet1":
                    continue
                
                else:
                    save_name = os.path.join(OUTPUT_DIR, f"{sheet_name}.csv")
                    sheet_data.to_csv(save_name)


# %%
