# %%

import multiprocessing
import os
import random
import shutil
import sys

import numpy as np
import pandas as pd
import toml
# from settings import main_settings
# --- IMPORTS ---
from src import settings
from src.utils import data_utils, fileio

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_TREATMENT = "Cs_5DIZ"  # Use Control flies to build the null model
NEW_PIPELINE_OUTPUT = f'./drosophila-data/Pseudo{SOURCE_TREATMENT}'
NUM_PSEUDO_GROUPS = 200      # 200 Files for strong statistics
FLIES_PER_GROUP = 12

# CSV COLUMN NAMES (CHECK THIS!)
X_COL = "pos x"
Y_COL = "pos y"

# Temp folder for rotated trajectories
TEMP_DIR = os.path.join(settings.OUTPUT_DIR, "temp_rotated_trajectories")


def rotate_trajectory(df):
    """
    Rotates x,y coordinates by a random angle theta around (0,0).
    Preserves distance from center (Thigmotaxis) but randomizes wall position.
    """
    theta = random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)

    # Assuming data is centered at 0,0.
    # If arena center is (W/2, H/2), subtract center first, rotate, then add back.
    # Here we assume pre-processed data is centered.

    x = df[X_COL].values
    y = df[Y_COL].values

    # Rotation Matrix
    x_new = x * c - y * s
    y_new = x * s + y * c

    df_new = df.copy()
    df_new[X_COL] = x_new
    df_new[Y_COL] = y_new

    return df_new


def process_iteration(iteration_data):
    i, treatment_all, treatment_config, output_dir = iteration_data

    # Create specific temp subfolder for this worker process to avoid collisions
    worker_temp_dir = os.path.join(TEMP_DIR, f"worker_{i}")
    os.makedirs(worker_temp_dir, exist_ok=True)

    # 1. Flatten and Sample
    all_available_flies = []
    for group_name, files_dict in treatment_all.items():
        for fly_name, file_path in files_dict.items():
            all_available_flies.append((fly_name, group_name, file_path))

    selected_flies = random.sample(all_available_flies, FLIES_PER_GROUP)

    pseudo_fly_dict = {}

    # 2. Rotate and Save Temp Files
    for fly_name, group_name, file_path in selected_flies:
        try:
            df = pd.read_csv(file_path)

            df_rotated = rotate_trajectory(df)

            unique_name = f"{fly_name}_{group_name}.csv"
            temp_path = os.path.join(worker_temp_dir, unique_name)
            df_rotated.to_csv(temp_path, index=False)

            unique_key = f"{fly_name}_{group_name}"
            pseudo_fly_dict[unique_key] = temp_path

        except Exception as e:
            print(f"Skipping corrupt file {file_path}: {e}")

    try:
        distances = data_utils.distances_between_all_flies(pseudo_fly_dict)
        angles = data_utils.angles_between_all_flies(pseudo_fly_dict)

        edgelist_df = find_interactions2(angles, distances, treatment_config)

        output_filename = f"pseudo_group_{i}.csv"
        edgelist_df.to_csv(os.path.join(output_dir, output_filename), index=False)

    except Exception as e:
        print(f"Error in iteration {i}: {e}")

    shutil.rmtree(worker_temp_dir, ignore_errors=True)


def find_interactions2(df_angles, df_distances, treatment_config):
    ANGLE = treatment_config["ANGLE"]
    DISTANCE = treatment_config["DISTANCE"]
    max_soc_duration = 1

    edgelist = []
    columns = ["node_1", "node_2", "start_of_interaction", "end_of_interaction", "duration"]

    for angles_col, distances_col in zip(df_angles.columns, df_distances.columns):
        if angles_col != distances_col:
            continue

        dist_vals = df_distances[distances_col].values
        angle_vals = df_angles[angles_col].values

        mask = (dist_vals <= DISTANCE) & (angle_vals >= ANGLE[0]) & (angle_vals <= ANGLE[1])

        if not np.any(mask):
            continue

        padded = np.concatenate(([0], mask.astype(int), [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        durations = ends - starts
        valid = durations >= max_soc_duration

        starts = starts[valid]
        ends = ends[valid]
        durations = durations[valid]

        if len(starts) == 0:
            continue

        node_1, node_2 = angles_col.split(" ")
        node_1 = node_1.replace(".csv", "")
        node_2 = node_2.replace(".csv", "")

        rows = [[node_1, node_2, int(s), int(e), int(d)] for s, e, d in zip(starts, ends-1, durations)]
        edgelist.extend(rows)

    return pd.DataFrame(edgelist, columns=columns)


print("--- Generating Rotated Pseudo-Populations ---")

main_config = fileio.get_config(settings.CONFIG_NAME)
config_path = os.path.join(settings.CONFIG_DIR, "interaction_criteria", f"{SOURCE_TREATMENT}.toml")

if not os.path.exists(config_path):
    print(f"Error: Config not found {config_path}")


with open(config_path) as f:
    treatment_config = toml.load(f)

input_dir = os.path.join(settings.OUTPUT_DIR, "1_1_preproc_data", SOURCE_TREATMENT)
treatment_dict = fileio.load_multiple_folders(input_dir)

print(treatment_dict)

treatment_all = {}
for group_name, group_path in treatment_dict.items():
    treatment_all.update({group_name: fileio.load_files_from_folder(group_path, file_format=".csv")})

os.makedirs(NEW_PIPELINE_OUTPUT, exist_ok=True)

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

num_processes = max(1, multiprocessing.cpu_count() // 2)
pool = multiprocessing.Pool(processes=num_processes)

iteration_data = [(i, treatment_all, treatment_config, NEW_PIPELINE_OUTPUT)
                  for i in range(1, NUM_PSEUDO_GROUPS + 1)]

pool.map(process_iteration, iteration_data)
pool.close()
pool.join()

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
print("\nGeneration Complete.")
