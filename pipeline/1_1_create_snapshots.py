import os

import networkx as nx
import pandas as pd

from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]
INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_0_find_interactions", TREATMENT)

TIME_WINDOW = int(os.environ["TIME_WINDOW"])
TIME_WINDOW_FPS = TIME_WINDOW * int(os.environ["FPS"])

SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "1_1_create_snapshots", f"{TIME_WINDOW}_sec_window", TREATMENT)
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

        # nodes = ['fly' + str(i) for i in range(1, 13)]
        # G.add_nodes_from(nodes)

        for _, row in df_snapshot.iterrows():
            node_1 = row["node_1"]
            node_2 = row["node_2"]
            duration = row["duration"]

            duration_list = [duration]
            count = 1

            if G.has_edge(node_1, node_2):
                count += G[node_1][node_2]["count"]
                duration_list.append(duration)

            else:
                G.add_edge(node_1, node_2, count=count, duration=duration_list)

        nx.write_gml(G, os.path.join(SAVE_GROUP_PATH, f"{i}.gml"))
