# %%
import os

import networkx as nx
import pandas as pd

from src import settings
from src.utils import fileio, graph_utils

TREATMENT = os.environ["TREATMENT"]
INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_2_create_total_graph", TREATMENT)
treatment = fileio.load_files_from_folder(INPUT_DIR, file_format=".gml")

SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "local_measures", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

graph_functions = graph_utils.local_measures_functions()

for group_name, group_path in treatment.items():
    G = nx.read_gml(group_path)
    data = {}
    for function_name, function_defintion in graph_functions:
        data[function_name] = function_defintion(G)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(SCRIPT_OUTPUT, f"{group_name.replace('.gml','')}"))
