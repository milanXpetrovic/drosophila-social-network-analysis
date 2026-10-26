# %%
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Define the Analysis Root (2 levels up) to find 'src'
#    (src/2_networks/ -> src/ -> 5_drosophila_social_network_analysis/)
analysis_root = os.path.dirname(os.path.dirname(current_dir))

# 2. Define the Project Root (3 levels up) to find 'settings'
#    (5_drosophila.../ -> droso-pipe/)
project_root = os.path.dirname(analysis_root)

# Add both to sys.path to ensure Python finds 'src' AND 'settings'
if analysis_root not in sys.path:
    sys.path.append(analysis_root)

if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import toml
from src.utils import fileio

from settings import main_settings
from settings import social_network_analysis as settings

# TREATMENT = main_settings.TREATMENT

# TREATMENT = "CS_old8-10d_24group"
# TREATMENT = "CS_isolated_24group"
TREATMENT = "CS_old3-5d_24group"


SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "2_0_find_interactions", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

ANGLES_DIR = os.path.join(settings.OUTPUT_DIR, "1_1_2_angles_matrix", TREATMENT)
DISTANCES_DIR = os.path.join(settings.OUTPUT_DIR, "1_1_1_distances_matrix", TREATMENT)

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
            "duration"
        ])

    for angles_col, distances_col in zip(df_angles.columns, df_distances.columns):
        if angles_col != distances_col:
            sys.exit()

        df = pd.concat([df_angles[angles_col], df_distances[distances_col]], axis=1)
        df.columns = ["angle", "distance"]

        distance_mask = df["distance"] <= DISTANCE
        angle_mask = (df["angle"] >= ANGLE[0]) & (df["angle"] <= ANGLE[1])
        df = df[distance_mask & angle_mask]

        timecut_frames = int(TIME * main_settings.VIDEO_FPS)
        clear_list_of_df = [d for _, d in df.groupby(df.index - np.arange(len(df))) if len(d) >= timecut_frames]

        node_1, node_2 = angles_col.split(" ")
        node_1, node_2 = node_1.replace(".csv", ""), node_2.replace(".csv", "")

        for interaction in clear_list_of_df:
            data = {
                "node_1": node_1,
                "node_2": node_2,
                "start_of_interaction": int(interaction.index[0]),
                "end_of_interaction": int(interaction.index[-1]),
                "duration": int(len(interaction))
            }

            row = pd.DataFrame.from_dict(data, orient="index").T
            edgelist = pd.concat([edgelist, row], ignore_index=True)

    edgelist = edgelist.sort_values("start_of_interaction")
    edgelist.to_csv(os.path.join(SCRIPT_OUTPUT, angles_name))
