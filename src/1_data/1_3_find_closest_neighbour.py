# %%
import os

import pandas as pd
import toml

from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]

CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    main_config = toml.load(file)

DISTANCES_DIR = os.path.join(settings.OUTPUT_DIR, "1_1_1_distances_matrix", TREATMENT)
distances = fileio.load_files_from_folder(DISTANCES_DIR)

SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "1_3_find_closest_neighbour", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

for distances_name, distances_path in distances.items():
    df = pd.read_csv(distances_path, index_col=0)
    all_flies = set([c.split(" ")[0] for c in df.columns])

    res = {}
    for fly in all_flies:
        df_selected = df[[col for col in df.columns if col.startswith(fly)]]
        res[fly] = df_selected.min(axis=1)

    df_res = pd.DataFrame(res)
    df_res.to_csv(os.path.join(SCRIPT_OUTPUT, distances_name))