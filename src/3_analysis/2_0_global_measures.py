# %%
import os

import networkx as nx
import pandas as pd
from src import settings
from src.utils import fileio, graph_utils

TREATMENT = os.environ["TREATMENT"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_2_create_total_graph", TREATMENT)
SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "global_measures")
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

graphs = fileio.load_files_from_folder(INPUT_DIR, file_format=".gml")

total = pd.DataFrame()
for group_name, group_path in graphs.items():
    G = nx.read_gml(group_path)
    df = graph_utils.graph_global_measures(G, group_name)
    total = pd.concat([total, df], axis=1)

SAVE_PATH = os.path.join(SCRIPT_OUTPUT, f"{TREATMENT}.csv")
total = total.T
total.to_csv(SAVE_PATH)
