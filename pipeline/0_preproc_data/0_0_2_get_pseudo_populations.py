# %%

# Generates pseudo populations by random sampling N flies from given treatment,
# each fly from different group

import os
import sys
import toml
import pandas as pd

from src import settings
from src.utils import data_utils, fileio


def find_interactions(df_angles, df_distances, treatment_config):
    ANGLE = treatment_config["ANGLE"]
    DISTANCE = treatment_config["DISTANCE"]
    TIME = treatment_config["TIME"]

    edgelist = pd.DataFrame(
        columns=[
            "node_1",
            "node_2",
            "start_of_interaction",
            "end_of_interaction",
            "duration"
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
        min_soc_duration = int(TIME[0] * int(os.environ["FPS"]))
        max_soc_duration = int((TIME[0]) * int(os.environ["FPS"]))

        clear_list_of_df = [
            d
            for _, d in df.groupby(df.index - np.arange(len(df)))
            if len(d) >= min_soc_duration and len(d) <= max_soc_duration
        ]

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
                "duration": int(duration)
            }

            row = pd.DataFrame.from_dict(data, orient="index").T
            edgelist = pd.concat([edgelist, row], ignore_index=True)

    return edgelist


TREATMENT = os.environ["TREATMENT"]

CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    config = toml.load(file)

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "0_0_preproc_data", TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

TREATMENT_CONFIG = os.path.join(settings.CONFIG_DIR, "trackings", f"{TREATMENT}.toml")
with open(TREATMENT_CONFIG) as f:
    treatment_config = toml.load(f)


# make random population


find_interactions(df_angles, df_distances, treatment_config)
