import os

import numpy as np
import pandas as pd

from src import settings
from src.utils import fileio, plotting

TREATMENT = os.environ["TREATMENT"]

START_TIME, END_TIME = int(os.environ["START_TIME"]) * 60, int(os.environ["END_TIME"]) * 60
START, END = settings.FPS * START_TIME, settings.FPS * END_TIME

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "0_0_preproc_data", TREATMENT)
trials = fileio.load_multiple_folders(INPUT_DIR)

SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "retention_heatmaps", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

pos_x_treatment, pos_y_treatment = np.array([]), np.array([])
for group_name, group_path in trials.items():
    pos_x_group, pos_y_group = np.array([]), np.array([])

    fly_dict = fileio.load_files_from_folder(group_path)
    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path, usecols=["pos x", "pos y"])
        df = df.iloc[START : END + 1, :]
        pos_x, pos_y = df["pos x"].to_numpy(), df["pos y"].to_numpy()
        pos_x_group = np.concatenate((pos_x_group, pos_x))
        pos_y_group = np.concatenate((pos_y_group, pos_y))
        pos_x_treatment = np.concatenate((pos_x_treatment, pos_x))
        pos_y_treatment = np.concatenate((pos_y_treatment, pos_y))

    save_name = os.path.join(SCRIPT_OUTPUT, f"{group_name}")
    pos_x_group = pos_x_group[~np.isnan(pos_x_group)]
    pos_y_group = pos_y_group[~np.isnan(pos_y_group)]
    plotting.plot_histogram(
        pos_x_group,
        pos_y_group,
        f"{TREATMENT}: {group_name} with gaussian filter",
        save_name,
    )

pos_x_treatment = pos_x_treatment[~np.isnan(pos_x_treatment)]
pos_y_treatment = pos_y_treatment[~np.isnan(pos_y_treatment)]
save_name = os.path.join(SCRIPT_OUTPUT, f"{TREATMENT}_all")
plotting.plot_histogram(
    pos_x_treatment,
    pos_y_treatment,
    f"{TREATMENT} all groups with gaussian filter",
    save_name,
)
