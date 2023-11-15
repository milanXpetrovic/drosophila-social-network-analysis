import math
import os

import pandas as pd
import toml
from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]

CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    config = toml.load(file)

INPUT_DIR = os.path.join(settings.INPUT_DIR, TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

SCRIPT_OUTPUT = os.path.join(settings.NORMALIZATION_DIR, TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

for group_name, group_path in treatment.items():
    fly_dict = fileio.load_files_from_folder(group_path)
    min_x, min_y, max_x, max_y = (
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
    )

    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path)
        min_x = min(min_x, df["pos x"].min())
        max_x = max(max_x, df["pos x"].max())
        min_y = min(min_y, df["pos y"].min())
        max_y = max(max_y, df["pos y"].max())

    north = ( ( (max_x - min_x) / 2 ) + min_x, max_y )
    south = ( ( (max_x - min_x) / 2 ) + min_x, min_y )
    west = ( max_x, ((max_y - min_y) / 2) + min_y )
    east = ( min_x, ((max_y - min_y) / 2) + min_y )

    dist_south_north = math.dist(south, north)
    dist_east_west = math.dist(east, west)

    toml_data = {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "x_px_ratio": dist_east_west / config["ARENA_DIAMETER"],
        "y_px_ratio": dist_south_north / config["ARENA_DIAMETER"],
    }

    toml_file_path = os.path.join(SCRIPT_OUTPUT, f"{group_name.replace('.csv', '')}.toml")
    with open(toml_file_path, "w") as toml_file:
        toml.dump(toml_data, toml_file)
