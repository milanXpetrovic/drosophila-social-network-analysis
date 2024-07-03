# %%
import os

import numpy as np
import pandas as pd

from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_0_preproc_data", TREATMENT)
trials = fileio.load_multiple_folders(INPUT_DIR)
SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "distances_traveled", TREATMENT)

os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

for group_name, group_path in trials.items():
    group_distances = {}
    fly_dict = fileio.load_files_from_folder(group_path)

    res = pd.DataFrame()
    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path, usecols=["pos x", "pos y"])
        pos_x, pos_y = df["pos x"].to_numpy(), df["pos y"].to_numpy()

        distances = np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2)
        total_distance = np.sum(distances)

        fly_name = fly_name.replace(".csv", "")
        distances = pd.Series(distances, name=fly_name)
        res = pd.concat([res, distances], axis=1)

    res.to_csv(
        os.path.join(
            SCRIPT_OUTPUT,
            f"{group_name}.csv",
        )
    )
