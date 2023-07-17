# %%
import os

import networkx as nx
import pandas as pd

from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]
INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_0_find_interactions", TREATMENT)
SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "1_2_create_graph", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

treatment = fileio.load_files_from_folder(INPUT_DIR)
for group_name, group_path in treatment.items():
    df_interactions = pd.read_csv(group_path, index_col=0)
    df_interactions = df_interactions.sort_values("start_of_interaction")
    G = nx.DiGraph()
    for _, row in df_interactions.iterrows():
        node_1, node_2 = row["node_1"], row["node_2"]
        duration = row["duration"]
        duration_list = [duration]
        count = 1

        if G.has_edge(node_1, node_2):
            count += G[node_1][node_2]["count"]
            duration_list.append(duration)
        else:
            G.add_edge(node_1, node_2, count=count, duration=duration_list)

    nx.write_gml(G, os.path.join(SCRIPT_OUTPUT, f"{group_name.replace('.csv', '')}.gml"))
