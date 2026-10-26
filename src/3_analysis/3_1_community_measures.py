# %%
import os
from pathlib import Path

import networkx as nx
import pandas as pd

from settings import main_settings
from src.utils import fileio, graph_utils

import networkx as nx
import pandas as pd
import numpy as np
import community
from collections import defaultdict


def analyze_community_structure(g, group_name, weight=None):
    """
    Analyzes the community structure of a graph found by the Louvain algorithm.

    This function identifies communities, calculates statistics about their size and
    distribution, and measures their internal cohesiveness.

    Args:
        g (nx.Graph): A NetworkX graph.
        group_name (str): The name of the graph/population for labeling.
        weight (str, optional): The edge attribute to use as a weight. Defaults to None.

    Returns:
        pd.DataFrame: A transposed DataFrame with statistics about the communities.
    """
    # 1. Find the best partition (community structure)
    # Using an undirected view of the graph is standard for Louvain
    g_undirected = g.to_undirected()
    partition = community.best_partition(g_undirected, weight=weight)

    if not partition:
        # Handle empty graphs or cases where partitioning fails
        return pd.DataFrame(index=[f"{group_name.replace('.gml', '')} weight={weight}"])

    # 2. Group nodes into communities efficiently
    communities_map = defaultdict(list)
    for node, comm_id in partition.items():
        communities_map[comm_id].append(node)
    
    communities = list(communities_map.values())
    community_sizes = [len(c) for c in communities]
    community_sizes.sort(reverse=True)

    # 3. Calculate community statistics, handling edge cases
    num_communities = len(community_sizes)
    single_node_communities = sum(1 for size in community_sizes if size == 1)
    
    # Analyze communities with more than one member
    multi_node_sizes = [size for size in community_sizes if size > 1]
    if multi_node_sizes:
        avg_comm_size_no_singles = np.mean(multi_node_sizes)
        std_comm_size_no_singles = np.std(multi_node_sizes)
    else:
        avg_comm_size_no_singles = 0
        std_comm_size_no_singles = 0

    # Calculate average internal density for non-single communities
    internal_densities = [
        nx.density(g.subgraph(comm)) for comm in communities if len(comm) > 1
    ]
    avg_internal_density = np.mean(internal_densities) if internal_densities else 0

    # 4. Compile results into a dictionary
    d = {
        'total_nodes': g.number_of_nodes(),
        'number_of_communities': num_communities,
        'communities_gt_1_node': len(multi_node_sizes),
        'single_node_communities': single_node_communities,
        'percent_single_nodes': single_node_communities / g.number_of_nodes() if g.number_of_nodes() > 0 else 0,
        'largest_community_size': community_sizes[0] if community_sizes else 0,
        'second_largest_community_size': community_sizes[1] if num_communities > 1 else 0,
        'avg_community_size': avg_comm_size_no_singles,
        'std_dev_community_size': std_comm_size_no_singles,
        'avg_internal_density': avg_internal_density,
    }

    # 5. Create and return DataFrame
    col_name = group_name.replace('.gml', '')
    df = pd.DataFrame(d, index=[f'{col_name} weight={weight}'])
    
    return df.T


TREATMENT = main_settings.TREATMENT
INPUT_DIR = Path(main_settings.SAVE_DATA_LOCATION) / "5_social_network_analysis" / "data" / "processed" / "2_2_create_total_graph" /TREATMENT
treatment = fileio.load_files_from_folder(INPUT_DIR, file_format=".gml")

SCRIPT_OUTPUT = Path(main_settings.RESULTS_DIR) / "community_measures" / TREATMENT
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

total = pd.DataFrame()
weights = ["None", "total_interaction_times", "count"]
for weight in weights:
    for group_name, group_path in treatment.items():
        G = nx.read_gml(group_path)
        df = analyze_community_structure(G, group_name, weight=weight)
        total = pd.concat([total, df], axis=1)

    total.to_csv(os.path.join(SCRIPT_OUTPUT, f"community_measures_{weight}.csv"))
