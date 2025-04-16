#%%
import os

import networkx as nx
import pandas as pd

from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]
INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_0_find_interactions", TREATMENT)
main_config = fileio.get_config(settings.CONFIG_NAME)
TIME_WINDOW = main_config["TIME_WINDOW"]
TIME_WINDOW_FPS = TIME_WINDOW * main_config["FPS"]
SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "2_1_create_snapshots", f"{TIME_WINDOW}_sec_window", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)
treatment = fileio.load_files_from_folder(INPUT_DIR)

for group_name, group_path in treatment.items():
    df_interactions = pd.read_csv(group_path, index_col=0)
    df_interactions = df_interactions.sort_values("start_of_interaction")
    df_interactions["snapshot"] = df_interactions["start_of_interaction"] // TIME_WINDOW_FPS
    df_interactions["snapshot"] = df_interactions["snapshot"] + 1

    SAVE_GROUP_PATH = os.path.join(SCRIPT_OUTPUT, group_name.replace(".csv", ""))
    os.makedirs(SAVE_GROUP_PATH, exist_ok=True)

    for i in range(1, df_interactions["snapshot"].max() + 1):
        df_snapshot = df_interactions[df_interactions["snapshot"] == i]
        G = nx.DiGraph()

        for _, row in df_snapshot.iterrows():
            node_1, node_2 = row["node_1"], row["node_2"]
            duration = row["duration"]

            if G.has_edge(node_1, node_2):
                G[node_1][node_2]["count"] += 1
                G[node_1][node_2]["interaction_times_list"].append(duration)
                G[node_1][node_2]["total_interaction_times"] += duration

            else:
                G.add_edge(node_1, node_2, count=1, total_interaction_times=duration,
                           interaction_times_list=[duration])

        nx.write_gml(G, os.path.join(SAVE_GROUP_PATH, f"{i}.gml"))
