#%%
import os
import sys

import pandas as pd
import networkx as nx

from src import settings
from src.utils import fileio, graph_utils


TREATMENT = "CsCh"
TIME_WINDOW = 120

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, TREATMENT, "1_1_create_snapshots", f"{TIME_WINDOW}_sec_window")

SCRIPT_OUTPUT = os.path.join(
    settings.RESULTS_DIR, TREATMENT, "global_measures", f"{TIME_WINDOW}_sec_window"
)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

treatment = fileio.load_multiple_folders(INPUT_DIR)

for group_name, group_path in treatment.items():
    snapshot_graphs = fileio.load_files_from_folder(group_path, n_sort=True, file_format='.gml')
    total = pd.DataFrame()
    for i, graph_path in enumerate(snapshot_graphs.values()):
        G = nx.read_gml(graph_path)
        df = graph_utils.graph_global_measures(G, str(i))
        total = pd.concat([total, df], axis=1)
    
    SAVE_PATH = os.path.join(SCRIPT_OUTPUT, f"{group_name}.csv")
    total = total.T
    total.to_csv(SAVE_PATH)

#%%


total = pd.DataFrame()
for pop_name, path in treatment.items():
    df = pd.DataFrame()
    g = nx.read_gml(path)

    df = graph_utils.graph_global_measures(g, pop_name)
    total = pd.concat([total, df], axis=1)

total.to_csv(os.path.join(SCRIPT_OUTPUT, "global_measures.csv"))
total.to_latex(os.path.join(SCRIPT_OUTPUT, "global_measures.tex"))
