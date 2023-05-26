#%%
import os
import sys
import pandas as pd

from src import settings
from src.utils import fileio, data_utils

TREATMENT = os.environ["TREATMENT"]

START_TIME = int(os.environ["START_TIME"]) * 60 
END_TIME = int(os.environ["END_TIME"]) * 60

START = settings.FPS * START_TIME
END = settings.FPS * END_TIME

SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, TREATMENT, "0_0_preproc_data")
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

INPUT_DIR = os.path.join(settings.INPUT_DIR, TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

for group_name, group_path in treatment.items():
    os.makedirs(os.path.join(SCRIPT_OUTPUT, group_name), exist_ok=True)

    fly_dict = fileio.load_files_from_folder(group_path)
    min_x, min_y = data_utils.find_group_mins(group_path)

    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path)
        df = df. iloc[START:END, :]
        df = data_utils.prepproc(df, min_x, min_y)
        df = data_utils.round_coordinates(df, decimal_places=0)

        df.to_csv(os.path.join(SCRIPT_OUTPUT, group_name, fly_name))
