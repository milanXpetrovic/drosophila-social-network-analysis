# %%
import os
import random

import networkx as nx

from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "2_2_create_total_graph", TREATMENT)
os.makedirs(os.path.join(settings.OUTPUT_DIR, "2_5_get_shuffled_networks"), exist_ok=True)

SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "2_5_get_shuffled_networks", f"{TREATMENT}")
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

graphs = fileio.load_files_from_folder(INPUT_DIR, file_format=".gml")
for group_name, group_path in graphs.items():
    GRAPHS_OUTPUT = os.path.join(SCRIPT_OUTPUT, group_name.replace('.gml', ''))
    os.makedirs(GRAPHS_OUTPUT, exist_ok=True)

    G = nx.read_gml(group_path)
    A_count = nx.to_numpy_array(G, weight="count", dtype=int)
    A_time = nx.to_numpy_array(G, weight="total_interaction_times", dtype=float)
    num_nodes = len(G.nodes())

    for g_i in range(1000):
        G_shuffled = nx.DiGraph()
        G_shuffled.add_nodes_from([x for x in range(num_nodes)])
        weights_counts = A_count[A_count != 0].tolist()
        weights_time = A_time[A_time != 0].tolist()
        random.shuffle(weights_counts)
        random.shuffle(weights_time)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                elif len(weights_counts) and len(weights_time):
                    G_shuffled.add_edge(i, j, count=weights_counts.pop(), total_interaction_times=weights_time.pop())

        nx.write_gml(G_shuffled, f"{GRAPHS_OUTPUT}/{g_i}.gml")
