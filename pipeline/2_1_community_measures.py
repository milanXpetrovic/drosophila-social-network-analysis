# %%
import os
import sys

import networkx as nx
import pandas as pd

from src import settings
from src.utils import fileio, graph_utils

TREATMENT = os.environ["TREATMENT"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_0_undirected_singleedge_graph", TREATMENT)
treatment = fileio.load_files_from_folder(INPUT_DIR, file_format=".gml")

SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "community_measures", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

total = pd.DataFrame()
weights = ["None", "duration", "count"]
for group_name, group_path in treatment.items():
    G = nx.read_gml(group_path)
    for weight in weights:
        df = graph_utils.group_comm_stats(G, group_name, weight=weight)
        total = pd.concat([total, df], axis=1)

total.to_csv(os.path.join(SCRIPT_OUTPUT, "louvian_community_measures.csv"))
total.to_latex(os.path.join(SCRIPT_OUTPUT, "louvian_community_measures.tex"))
