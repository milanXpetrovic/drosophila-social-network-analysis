# %%
import os
import pandas as pd
import networkx as nx

from src import settings
from src.utils import fileio, data_utils

SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "1_0_undirected_singleedge_graph")
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

ANGLES_DIR = os.path.join(settings.OUTPUT_DIR, "0_2_angles_between_flies_matrix")
DISTANCES_DIR = os.path.join(settings.OUTPUT_DIR, "0_1_distances_between_flies_matrix")

angles = fileio.load_files_from_folder(ANGLES_DIR)
distances = fileio.load_files_from_folder(DISTANCES_DIR)

for angles_tuple, distances_tuple in zip(angles.items(), distances.items()):
    angles_name, angles_path = angles_tuple
    distances_name, distances_path = distances_tuple

    df_angles = pd.read_csv(angles_path, index_col=0)
    df_distances = pd.read_csv(distances_path, index_col=0)

    G = data_utils.add_edges_to_undirected_g(df_angles, df_distances)
    nx.write_gml(G, os.path.join(SCRIPT_OUTPUT, angles_name.replace(".csv", ".gml")))
