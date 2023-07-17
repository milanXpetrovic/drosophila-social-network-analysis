# %%
import os
import toml
import pandas as pd

from src import settings
from src.utils import data_utils, fileio

TREATMENT = os.environ["TREATMENT"]

START = int(os.environ["START_TIME"]) * 60 * int(os.environ["FPS"])
END = int(os.environ["END_TIME"]) * 60 * int(os.environ["FPS"])


SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "0_0_preproc_data", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

INPUT_DIR = os.path.join(settings.INPUT_DIR, TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

for group_name, group_path in treatment.items():
    os.makedirs(os.path.join(SCRIPT_OUTPUT, group_name), exist_ok=True)

    fly_dict = fileio.load_files_from_folder(group_path)
    min_x, min_y = data_utils.find_group_mins(group_path)

    group_norm_path = os.path.join(settings.NORMALIZATION_DIR, TREATMENT, f"{group_name.replace('.csv', '')}.toml")

    with open(group_norm_path, "r") as group_norm:
        group_norm = toml.load(group_norm)

    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path)
        df = df.iloc[START:END, :]

        df = df.fillna(method="ffill")

        df["pos x"] = df["pos x"].subtract(group_norm.get("min_x"))
        df["pos y"] = df["pos y"].subtract(group_norm.get("min_y"))

        df["pos x"] = df["pos x"] / group_norm.get("x_px_ratio")
        df["pos y"] = df["pos y"] / group_norm.get("y_px_ratio")

        mean_ratio = (group_norm.get("x_px_ratio") + group_norm.get("y_px_ratio")) / 2
        df["major axis len"] = df["major axis len"] / mean_ratio

        # df = data_utils.round_coordinates(df, decimal_places=0)

        df.to_csv(os.path.join(SCRIPT_OUTPUT, group_name, fly_name))
