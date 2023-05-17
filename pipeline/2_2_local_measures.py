# %%
import os
import sys
import json
import pandas as pd
import networkx as nx

from src import settings
from src.utils import fileio, graph_utils

TREATMENT = sys.argv[1]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, TREATMENT, "1_0_undirected_singleedge_graph")
treatment = fileio.load_files_from_folder(INPUT_DIR, file_format=".gml")

SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, TREATMENT, "local_measures")
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

graphs_d = {
    exp_name.replace(".gml", ""): nx.read_gml(path)
    for exp_name, path in treatment.items()
}

graph_functions = graph_utils.local_measures_functions()

for function_name, function_defintion in graph_functions:
    values = {}
    for group_name, group_graph in graphs_d.items():
        values.update({group_name: function_defintion(group_graph)})
    with open(f"{SCRIPT_OUTPUT}/{function_name}.json", "w") as file:
        file.write(json.dumps(values, indent=4))

    try:
        df = pd.DataFrame()
        for group_name, group_graph in graphs_d.items():
            measures = function_defintion(group_graph)
            res = pd.DataFrame.from_dict(measures, orient="index", columns=[group_name])
            df = pd.concat([df, res], axis=1)

        df.to_csv(f"{SCRIPT_OUTPUT}/{function_name}.csv")
    except:
        print(function_name)
