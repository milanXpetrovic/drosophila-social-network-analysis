# %%
import os

import networkx as nx
import pandas as pd
import toml
from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_0_find_interactions", TREATMENT)
SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "1_2_create_total_graph", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    config = toml.load(file)

treatment = fileio.load_files_from_folder(INPUT_DIR)
for group_name, group_path in treatment.items():
    df_interactions = pd.read_csv(group_path, index_col=0)
    df_interactions = df_interactions.sort_values("start_of_interaction")
    G = nx.DiGraph()

    for _, row in df_interactions.iterrows():
        node_1, node_2 = row["node_1"], row["node_2"]
        duration = row["duration"] / config["FPS"]

        if G.has_edge(node_1, node_2):
            G[node_1][node_2]["count"] += 1
            G[node_1][node_2]["interaction_times_list"].append(duration)
            G[node_1][node_2]["total_interaction_times"] += duration

        else:
            G.add_edge(
                node_1,
                node_2,
                count=1,
                total_interaction_times=duration,
                interaction_times_list=[duration],
            )

    nx.write_gml(G, os.path.join(SCRIPT_OUTPUT, f"{group_name.replace('.csv', '')}.gml"))
