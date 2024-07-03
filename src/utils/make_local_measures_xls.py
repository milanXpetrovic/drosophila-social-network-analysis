#%%
import os

import pandas as pd
import toml

from src import settings
from src.utils import fileio

INPUT_PATH = os.path.join(settings.RESULTS_DIR, "local_measures")
CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "measures_xlsx")

with open(CONFIG_PATH, "r") as file:
    config = toml.load(file)
   
all_treatments = fileio.load_multiple_folders(INPUT_PATH)
all_treatments = {key: value for key, value in all_treatments.items() if key in config["TREATMENTS"]}

treatment_dataframes = []
for treatment_name, treatment_path in all_treatments.items():
    all_groups = fileio.load_files_from_folder(treatment_path)
    group_dataframes = []
    for group_name, df_path in all_groups.items():
        df = pd.read_csv(df_path, index_col=0)  
        df['Group'] = group_name.replace(".csv","")
        group_dataframes.append(df)

    df = pd.concat(group_dataframes)
    df.set_index('Group', append=True, inplace=True)
    df['Treatment'] = treatment_name
    df.set_index('Treatment', append=True, inplace=True)
    treatment_dataframes.append(df)

combined_data = pd.concat(treatment_dataframes)

SAVE_PATH = os.path.join(SCRIPT_OUTPUT, "local_measures.xlsx")
with pd.ExcelWriter(SAVE_PATH) as writer:
    for treatment, data in combined_data.groupby(level='Treatment'):
        data.to_excel(writer, sheet_name=treatment)
        