# %%
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import fileio, graph_utils
import pandas as pd
import networkx as nx

path = "/home/milky/drosophila-SNA/data/processed/1_2_create_graph"
all_treatments = fileio.load_multiple_folders(path)

for treatment_name, treatment_path in all_treatments.items():
    treatment = fileio.load_files_from_folder(treatment_path)
    num_groups = len(treatment)

    total = pd.DataFrame()
    for group_name, group_path in treatment.items():
        G = nx.read_gml(group_path)
        df = graph_utils.graph_global_measures(G, group_name)
        total = pd.concat([total, df], axis=1)

    sys.exit()
