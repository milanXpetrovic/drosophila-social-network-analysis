# %%
import os
from pathlib import Path

import networkx as nx
import pandas as pd

from settings import main_settings
from src.utils import fileio, graph_utils


import networkx as nx
import numpy as np
import pandas as pd
import community  # Assumes python-louvain is installed

def _calculate_metric(metric_func, g, is_centrality=True, **kwargs):
    """
    Helper function to safely calculate a graph metric and its average if applicable.
    Returns 0 if the calculation fails for any reason.
    """
    try:
        # Handle centrality measures that return dictionaries of node values
        if is_centrality:
            result = metric_func(g, **kwargs)
            return np.mean(list(result.values()))
        # Handle global measures that return a single value
        else:
            return metric_func(g, **kwargs)
    except (nx.NetworkXError, ZeroDivisionError, ValueError, BaseException):
        # Catch a broad range of potential errors from networkx or other libraries
        return 0

def _calculate_degree_heterogeneity(g, weight=None):
    """Helper function to calculate degree heterogeneity."""
    try:
        degrees = [d for n, d in g.degree(weight=weight)]
        if not degrees:
            return 0
        mean_degree = np.mean(degrees)
        if mean_degree == 0:
            return 0
        return np.std(degrees) / mean_degree
    except (KeyError, ZeroDivisionError, BaseException):
        return 0


def global_graph_analysis(g, pop_name):
    """
    Calculates and returns a DataFrame of comprehensive global-level graph analysis measures.

    This function computes basic properties, centrality averages, community structure,
    resilience, path-based metrics, and small-world characteristics for both
    unweighted and weighted graphs.
    """
    # Prepare graphs: undirected version and its largest connected component
    g_undirected = g.to_undirected()
    largest_cc = max(nx.connected_components(g_undirected), key=len, default=None)
    g_largest_cc = g.subgraph(largest_cc).copy() if largest_cc else nx.Graph()

    # Define weights and corresponding labels for iterative calculations
    weights = [None, 'count', 'total_interaction_times']
    weight_labels = ['unweighted', 'count', 'duration(seconds)']

    # --- 1. Community and Modularity ---
    # Partitioning is done once on the undirected graph
    partition = community.best_partition(g_undirected)
    modularities = {
        f'Newman modularity {label}': _calculate_metric(
            community.modularity, g_undirected, is_centrality=False, partition=partition, weight=w
        ) for w, label in zip(weights, weight_labels)
    }
    community_metrics = {
        'Number of communities': len(set(partition.values())) if partition else 0,
        'Community coverage': _calculate_metric(community.coverage, g_undirected, is_centrality=False, partition=partition),
        'Community performance': _calculate_metric(community.performance, g_undirected, is_centrality=False, partition=partition),
    }

    # --- 2. Centrality and Clustering Averages ---
    metrics_to_calculate = {
        'Average clustering coefficient': nx.clustering,
        'Average betweenness centrality': nx.betweenness_centrality,
        'Average closeness centrality': nx.closeness_centrality,
    }
    analysis_results = {}
    for name, func in metrics_to_calculate.items():
        for weight, label in zip(weights, weight_labels):
            param = {'distance': weight} if func == nx.closeness_centrality else {'weight': weight}
            # Remove None from param value if weight is None for compatibility
            if param.get('weight') is None: param.pop('weight', None)
            if param.get('distance') is None: param.pop('distance', None)
            analysis_results[f'{name} {label}'] = _calculate_metric(func, g, **param)

    # --- 3. Degree-based Metrics ---
    degree_heterogeneity = {
        f'Degree heterogeneity {label}': _calculate_degree_heterogeneity(g, weight=w)
        for w, label in zip(weights, weight_labels)
    }
    degree_assortativity = {
        f'Degree assortativity {label}': _calculate_metric(
            nx.degree_assortativity_coefficient, g, is_centrality=False, weight=w
        ) for w, label in zip(weights, weight_labels)
    }
    mean_degrees = {
        f'Mean degree weight={label}': np.mean([d for n, d in g.degree(weight=w)])
        for w, label in zip(weights[1:], weight_labels[1:])
    }

    # --- 4. Assemble Final Dictionary ---
    d = {
        # Basic Properties
        'Total nodes': g.number_of_nodes(),
        'Total edges': g.number_of_edges(),
        'Network density': nx.density(g),
        'Reciprocity': nx.reciprocity(g),
        'Transitivity': nx.transitivity(g),

        # Connectivity and Resilience
        'Number of connected components': nx.number_connected_components(g_undirected),
        'Node connectivity': _calculate_metric(nx.node_connectivity, g_undirected, is_centrality=False),
        'Edge connectivity': _calculate_metric(nx.edge_connectivity, g_undirected, is_centrality=False),

        # Path-based Metrics (on largest connected component)
        'Global efficiency': _calculate_metric(nx.global_efficiency, g_largest_cc, is_centrality=False),
        'Average shortest path length': _calculate_metric(nx.average_shortest_path_length, g_largest_cc, is_centrality=False),
        'Diameter': _calculate_metric(nx.diameter, g_largest_cc, is_centrality=False),

        # Small-World Metrics (can be computationally expensive)
        'Small-world sigma': _calculate_metric(nx.sigma, g_undirected, niter=20, nrand=10, is_centrality=False),
        'Small-world omega': _calculate_metric(nx.omega, g_undirected, niter=20, nrand=10, is_centrality=False),

        # Add calculated dictionaries
        **mean_degrees,
        **degree_heterogeneity,
        **degree_assortativity,
        **analysis_results,
        **modularities,
        **community_metrics,
    }

    df = pd.DataFrame(d, index=[pop_name.replace('.gml', '')])
    return df.T

TREATMENT = main_settings.TREATMENT
INPUT_DIR = Path(main_settings.SAVE_DATA_LOCATION) / "5_social_network_analysis" / "data" / "processed" / "2_2_create_total_graph" /TREATMENT
SCRIPT_OUTPUT = Path(main_settings.RESULTS_DIR) / "global_measures"
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

graphs = fileio.load_files_from_folder(INPUT_DIR, file_format=".gml")
total = pd.DataFrame()
for group_name, group_path in graphs.items():
    G = nx.read_gml(group_path)
    df = global_graph_analysis(G, group_name)
    total = pd.concat([total, df], axis=1)

SAVE_PATH = os.path.join(SCRIPT_OUTPUT, f"{TREATMENT}.csv")
total = total.T
total.to_csv(SAVE_PATH)
