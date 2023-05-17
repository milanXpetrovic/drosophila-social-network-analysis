import os
import sys

import pandas as pd
import networkx as nx

from src import settings
from src.utils import fileio, graph_utils

TREATMENT = sys.argv[1]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_0_undirected_singleedge_graph")
treatment = fileio.load_files_from_folder(INPUT_DIR, file_format=".gml")

SCRIPT_OUTPUT = os.path.join(
    settings.RESULTS_DIR, TREATMENT, "global_measures"
)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

total = pd.DataFrame()
for pop_name, path in treatment.items():
    df = pd.DataFrame()
    g = nx.read_gml(path)

    df = graph_utils.graph_global_measures(g, pop_name)
    total = pd.concat([total, df], axis=1)

total.to_csv(os.path.join(SCRIPT_OUTPUT, "global_measures.csv"))
total.to_latex(os.path.join(SCRIPT_OUTPUT, "global_measures.tex"))
