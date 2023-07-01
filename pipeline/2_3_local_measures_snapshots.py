# %%
import sys
import os

import networkx as nx
import pandas as pd

from src import settings
from src.utils import fileio, graph_utils

TREATMENT = os.environ["TREATMENT"]
TIME_WINDOW = os.environ["TIME_WINDOW"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_1_create_snapshots", f"{TIME_WINDOW}_sec_window", TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

DISTANCE_TRAVELED = os.path.join(settings.RESULTS_DIR, "distances_traveled", TREATMENT)
treatment_distances = fileio.load_files_from_folder(DISTANCE_TRAVELED)

SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "local_measures_snapshots", f"{TIME_WINDOW}_sec_window", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

graph_functions = graph_utils.local_measures_functions()

for group_name, group_path in treatment.items():
    # Fly traveled distances part
    group_distances_path = treatment_distances[group_name+".csv"]
    df_distances = pd.read_csv(group_distances_path, index_col=0)
    df_distances = df_distances.groupby(df_distances.index // (int(TIME_WINDOW) * int(settings.FPS))).sum()
    df_distances.index = df_distances.index+1
    df_distances = df_distances.T

    # Graph measures part
    all_snapshopts = fileio.load_files_from_folder(group_path, '.gml')

    os.makedirs(os.path.join(SCRIPT_OUTPUT, group_name), exist_ok=True)

    for snapshot_name, snapshot_path in all_snapshopts.items():
        G = nx.read_gml(snapshot_path)
        data = {}
        for function_name, function_defintion in graph_functions:
            try:
                data[function_name] = function_defintion(G)
            except:
                data[function_name] = 0

        df = pd.DataFrame(data)
        # print(df)

        snapshot_name = snapshot_name.replace('.gml', '')
        # print(df.index)
        # print(df_distances[int(snapshot_name)].index)
        # sys.exit()
        df["Distance traveled"] = df_distances[int(snapshot_name)]

        df.to_csv(os.path.join(SCRIPT_OUTPUT, group_name, f"{snapshot_name}.csv"))
