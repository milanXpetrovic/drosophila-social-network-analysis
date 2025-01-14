# %%
import os
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd

from src import settings
from src.utils import fileio

TREATMENT = os.environ["TREATMENT"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "2_5_get_shuffled_networks", TREATMENT)
SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "shuffled_nets_global_measures")
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)


def graph_global_measures(g, pop_name):
    try:
        weighted_degree_count = nx.degree(g, weight='count')
        weighted_degree_count_values = [k for k in dict(weighted_degree_count).values()]
        mean_weighted_degree_count = np.mean(weighted_degree_count_values)
        standard_deviation_degree_count = np.std(weighted_degree_count_values)
        degree_heterogeneity_count = standard_deviation_degree_count / mean_weighted_degree_count

        weighted_degree_time = nx.degree(g, weight='total_interaction_times')
        weighted_degree_time_values = [k for k in dict(weighted_degree_time).values()]
        mean_weighted_degree_time = np.mean(weighted_degree_time_values)
        standard_deviation_degree_time = np.std(weighted_degree_time_values)
        degree_heterogeneity_total_interaction_time = standard_deviation_degree_time / mean_weighted_degree_time

    except BaseException: 
        degree_heterogeneity_count = 0
        degree_heterogeneity_total_interaction_time = 0

    try:
        degree_assortativity_count = nx.degree_assortativity_coefficient(g, weight='count')
        degree_assortativity_total_interaction_time = nx.degree_assortativity_coefficient(g, weight='total_interaction_times')

    except BaseException:
        degree_assortativity_count = 0
        degree_assortativity_total_interaction_time = 0

    clustering_coeff_w_count = nx.clustering(g, weight='count')
    average_cl_coeff_w_count = np.mean([k for k in clustering_coeff_w_count.values()])
    clustering_coeff_w_duration = nx.clustering(g, weight='total_interaction_times')
    average_cl_coeff_w_duration = np.mean([k for k in clustering_coeff_w_duration.values()])

    betweenness_c_w_count = nx.betweenness_centrality(g, weight='count')
    average_betw_c_w_count = np.mean([k for k in betweenness_c_w_count.values()])
    betweenness_c_w_duration = nx.betweenness_centrality(g, weight='total_interaction_times')
    average_betw_c_w_duration = np.mean([k for k in betweenness_c_w_duration.values()])

    closeness_c_w_count = nx.closeness_centrality(g, distance='count')
    ave_closeness_c_w_count = np.mean([k for k in closeness_c_w_count.values()])
    closeness_c_w_duration = nx.closeness_centrality(g, distance='total_interaction_times')
    ave_closeness_c_w_duration = np.mean([k for k in closeness_c_w_duration.values()])

    d = {
        'Total nodes': g.number_of_nodes(),
        'Total edges': g.number_of_edges(),
        'Mean degree weight=count': mean_weighted_degree_count,
        'Mean degree weight=duration(seconds)': mean_weighted_degree_time,
        'Degree heterogeneity (count)': degree_heterogeneity_count,
        'Degree heterogeneity (total duration (seconds))': degree_heterogeneity_total_interaction_time,
        'Degree aassortativity (count)': degree_assortativity_count,
        'Degree aassortativity (total duration (seconds))': degree_assortativity_total_interaction_time,
        'Average clustering coefficient weight=count': average_cl_coeff_w_count,
        'Average clustering coefficient weight=duration(seconds)': average_cl_coeff_w_duration,
        'Average betweenness centrality weight=count': average_betw_c_w_count,
        'Average betweenness centrality weight=duration(seconds)': average_betw_c_w_duration,
        'Average closseness centrality weight=count': ave_closeness_c_w_count,
        'Average closseness centrality weight=duration(seconds)': ave_closeness_c_w_duration,
    }

    df = pd.DataFrame(d, index=[pop_name.replace('.gml', '')])
    df = df.T

    return df


def process_graph(graph_info):
    graph_name, graph_path = graph_info
    G = nx.read_gml(graph_path)
    df = graph_global_measures(G, group_name)
    return df


groups = fileio.load_multiple_folders(INPUT_DIR)
for group_name, group_path in groups.items():
    total = pd.DataFrame()
    graphs = fileio.load_files_from_folder(group_path, file_format=".gml")
    GROUP_OUTPUT = os.path.join(SCRIPT_OUTPUT, TREATMENT, group_name)
    os.makedirs(os.path.join(SCRIPT_OUTPUT, TREATMENT), exist_ok=True)
    os.makedirs(GROUP_OUTPUT, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_graph, graphs.items()))

    total = pd.concat(results, axis=1)

    SAVE_PATH = os.path.join(GROUP_OUTPUT, f"{group_name}.csv")
    total = total.T
    total.to_csv(SAVE_PATH)
