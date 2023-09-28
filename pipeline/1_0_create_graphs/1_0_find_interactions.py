# %%
import os
import sys
import toml

import numpy as np
import pandas as pd

from src import settings
from src.utils import fileio

# TREATMENT = os.environ["TREATMENT"]
TREATMENT = "LDA_5DIZ"

CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    main_config = toml.load(file)

SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "1_0_find_interactions", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

ANGLES_DIR = os.path.join(settings.OUTPUT_DIR, "0_2_angles_between_flies_matrix", TREATMENT)
DISTANCES_DIR = os.path.join(settings.OUTPUT_DIR, "0_1_distances_between_flies_matrix", TREATMENT)

angles = fileio.load_files_from_folder(ANGLES_DIR)
distances = fileio.load_files_from_folder(DISTANCES_DIR)

TREATMENT_CONFIG = os.path.join(settings.CONFIG_DIR, "interaction_criteria", f"{TREATMENT}.toml")

with open(TREATMENT_CONFIG) as f:
    treatment_config = toml.load(f)
    ANGLE = treatment_config["ANGLE"]
    DISTANCE = treatment_config["DISTANCE"]
    TIME = treatment_config["TIME"]

for angles_tuple, distances_tuple in zip(angles.items(), distances.items()):
    angles_name, angles_path = angles_tuple
    distances_name, distances_path = distances_tuple

    if angles_name != distances_name:
        sys.exit()

    df_angles = pd.read_csv(angles_path, index_col=0)
    df_distances = pd.read_csv(distances_path, index_col=0)

    edgelist = pd.DataFrame(
        columns=[
            "node_1",
            "node_2",
            "start_of_interaction",
            "end_of_interaction",
            "duration",
            "distance",
            "angle",
        ]
    )

    for angles_col, distances_col in zip(df_angles.columns, df_distances.columns):
        if angles_col != distances_col:
            sys.exit()

        df = pd.concat([df_angles[angles_col], df_distances[distances_col]], axis=1)
        df.columns = ["angle", "distance"]

        distance_mask = df["distance"] <= DISTANCE
        angle_mask = (df["angle"] >= ANGLE[0]) & (df["angle"] <= ANGLE[1])
        df = df[distance_mask & angle_mask]

        timecut = int(TIME[0] * main_config["FPS"])
        clear_list_of_df = [d for _, d in df.groupby(df.index - np.arange(len(df))) if len(d) >= timecut]

        node_1, node_2 = angles_col.split(" ")
        node_1, node_2 = node_1.replace(".csv", ""), node_2.replace(".csv", "")

        for interaction in clear_list_of_df:
            duration = len(interaction)
            start_of_interaction = interaction.index[0]
            end_of_interaction = interaction.index[-1]

            data = {
                "node_1": node_1,
                "node_2": node_2,
                "start_of_interaction": int(start_of_interaction),
                "end_of_interaction": int(end_of_interaction),
                "duration": int(duration),
                "distance": interaction.distance.to_list(),
                "angle": interaction.angle.to_list(),
            }

            row = pd.DataFrame.from_dict(data, orient="index").T
            edgelist = pd.concat([edgelist, row], ignore_index=True)

    edgelist = edgelist.sort_values("start_of_interaction")
    edgelist.to_csv(os.path.join(SCRIPT_OUTPUT, angles_name))
