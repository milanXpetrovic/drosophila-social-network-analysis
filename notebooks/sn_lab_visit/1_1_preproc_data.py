#%%
import os

import pandas as pd
import toml

from src import settings
from src.utils import fileio

INPUT_DIR = './data/CsCh/proc'
treatment = fileio.load_multiple_folders(INPUT_DIR)

START = 0 * 60 * 22.8
START = int(START)
END = 30 * 60 * 22.8
END = int(END)

SCRIPT_OUTPUT = "./data/CsCh/postproc"
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

for group_name, group_path in treatment.items():
    os.makedirs(os.path.join(SCRIPT_OUTPUT, group_name), exist_ok=True)

    fly_dict = fileio.load_files_from_folder(group_path)
    group_norm_path = os.path.join("./data/CsCh/norm", f"{group_name.replace('.csv', '')}.toml")

    with open(group_norm_path, "r") as group_norm:
        group_norm = toml.load(group_norm)

    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path) #, index_col=0
        df = df.iloc[START:END, :]
        df = df.fillna(method="ffill")

        df["pos x"] = df["pos x"].subtract(group_norm.get("min_x"))
        df["pos y"] = df["pos y"].subtract(group_norm.get("min_y"))

        # df["pos x"] = df["pos x"] / group_norm.get("x_px_ratio")
        # df["pos y"] = df["pos y"] / group_norm.get("y_px_ratio")

        mean_ratio = (group_norm.get("x_px_ratio") + group_norm.get("y_px_ratio")) / 2
        # df["major axis len"] = df["a"] / mean_ratio
        # df["major axis len"] = df["a"]

        df.reset_index(drop=True, inplace=True)

        df = df[["pos x", "pos y", "ori", "major axis len"]]
        df.to_csv(os.path.join(SCRIPT_OUTPUT, group_name, fly_name))
