import os
import toml

import networkx as nx
import pandas as pd

from src import settings
from src.utils import fileio


TREATMENT = os.environ["TREATMENT"]

CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    config = toml.load(file)

TIME_WINDOW = config["TIME_WINDOW"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "1_1_create_snapshots", f"{TIME_WINDOW}_sec_window", TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "count_triads_snapshots", f"{TIME_WINDOW}_sec_window", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

for group_name, group_path in treatment.items():

    all_graphs = fileio.load_files_from_folder(group_path, '.gml')

    res = {}
    for graph_name, graph_path in all_graphs.items():
        g = nx.read_gml(graph_path)
        time_stamp = int(graph_name.replace(".gml", ""))
        res[time_stamp] = nx.triadic_census(g)

    df = pd.DataFrame.from_dict(res)
    df = df.T
    df.to_csv(os.path.join(SCRIPT_OUTPUT, f"{group_name}.csv"))
