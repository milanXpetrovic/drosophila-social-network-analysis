#%%
import os

import networkx as nx
import pandas as pd

from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "2_2_create_total_graph", TREATMENT)
SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "2_3_create_adj_matrix", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

graphs = fileio.load_files_from_folder(INPUT_DIR, file_format=".gml")

for group_name, group_path in graphs.items():
    G = nx.read_gml(group_path)

    adj_matrix_count = nx.to_numpy_array(G, weight="count", dtype=int)
    adj_matrix_time = nx.to_numpy_array(G, weight="total_interaction_times", dtype=int)

    SAVE_PATH = os.path.join(SCRIPT_OUTPUT, 'count')
    os.makedirs(SAVE_PATH, exist_ok=True)
    pd.DataFrame(adj_matrix_count).to_csv(os.path.join(SAVE_PATH, f"{group_name}.csv"), index=False, header=False)

    SAVE_PATH = os.path.join(SCRIPT_OUTPUT, 'total_time')
    os.makedirs(SAVE_PATH, exist_ok=True)
    pd.DataFrame(adj_matrix_time).to_csv(os.path.join(SAVE_PATH, f"{group_name}.csv"), index=False, header=False)
