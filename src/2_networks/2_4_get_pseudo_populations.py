#%%
import multiprocessing
import os
import random
import sys

import numpy as np
import pandas as pd
import toml

from src import settings
from src.utils import data_utils, fileio


def process_iteration(iteration_data):
    i, treatment_all, main_config, treatment_config, SCRIPT_OUTPUT = iteration_data
    
    temp_ind = random.sample(range(len(treatment_all)), main_config['N_OF_SAMPLES'])
    pick_random_groups = {list(treatment_all.keys())[i]: list(treatment_all.values())[i] for i in temp_ind}

    pseudo_fly_dict = {}
    for group_name, _ in pick_random_groups.items():
        group = treatment_all[group_name]

        random_ind = random.randint(0, len(group) - 1)

        pick_fly_name = list(group.keys())[random_ind]
        pick_fly_path = list(group.values())[random_ind]

        pseudo_fly_dict[f"{pick_fly_name}_{group_name}"] = pick_fly_path

    distances = data_utils.distances_between_all_flies(pseudo_fly_dict)
    angles = data_utils.angles_between_all_flies(pseudo_fly_dict)
    edgelist2 = find_interactions2(angles, distances, main_config, treatment_config)
    edgelist2.to_csv(os.path.join(SCRIPT_OUTPUT, f"{i}.csv"))


def find_interactions2(df_angles, df_distances, main_config, treatment_config):
    ANGLE = treatment_config["ANGLE"]
    DISTANCE = treatment_config["DISTANCE"]
    TIME = treatment_config["TIME"]
    max_soc_duration = 1

    edgelist = []
    columns = ["node_1", "node_2", "start_of_interaction", "end_of_interaction", "duration"]

    for angles_col, distances_col in zip(df_angles.columns, df_distances.columns):
        if angles_col != distances_col:
            sys.exit()

        df = pd.concat([df_angles[angles_col], df_distances[distances_col]], axis=1)
        df.columns = ["angle", "distance"]

        distance_mask = df["distance"] <= DISTANCE
        angle_mask = (df["angle"] >= ANGLE[0]) & (df["angle"] <= ANGLE[1])
        df = df[distance_mask & angle_mask]

        clear_list_of_df = [
            d
            for _, d in df.groupby(df.index - np.arange(len(df)))
            if len(d) >= max_soc_duration
        ]

        node_1, node_2 = angles_col.split(" ")
        node_1, node_2 = node_1.replace(".csv", ""), node_2.replace(".csv", "")

        for interaction in clear_list_of_df:
            duration = len(interaction)
            start_of_interaction = interaction.index[0]
            end_of_interaction = interaction.index[-1]

            edgelist.append([node_1, node_2, int(start_of_interaction), int(end_of_interaction), int(duration)])

    return pd.DataFrame(edgelist, columns=columns)


TREATMENT = os.environ["TREATMENT"]

main_config = fileio.get_config(settings.CONFIG_NAME)

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_1_preproc_data", TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

treatment_all = {}
for group_name, group_path in treatment.items():
    group = fileio.load_files_from_folder(group_path, file_format=".csv")
    treatment_all.update({group_name: group})

TREATMENT_CONFIG = os.path.join(settings.CONFIG_DIR, "interaction_criteria", f"{TREATMENT}.toml")
with open(TREATMENT_CONFIG) as f:
    treatment_config = toml.load(f)

SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "2_0_find_interactions", f"pseudo_{TREATMENT}")
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

existing_files = [f for f in os.listdir(SCRIPT_OUTPUT) if os.path.isfile(os.path.join(SCRIPT_OUTPUT, f))]
existing_files_count = len(existing_files)
start_iteration = existing_files_count + 1 if existing_files else 1

num_processes = multiprocessing.cpu_count()//2
pool = multiprocessing.Pool(processes=num_processes)
iteration_data = [(i, treatment_all, main_config, treatment_config, SCRIPT_OUTPUT) for i in range(start_iteration, main_config['N_RANDOM_GROUPS'] + 1)]

pool.map(process_iteration, iteration_data)
pool.close()
pool.join()
