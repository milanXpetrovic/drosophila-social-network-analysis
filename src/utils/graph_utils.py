import community
import networkx as nx
import numpy as np
import pandas as pd


def calculate_strength(g, weight_value):
    graph_freq = {}
    nodes = list(g.nodes)
    for node in nodes:
        edges = list(g.edges(node, data=True))

        freq = 0
        for edge in edges:
            (edge_a, edge_b, data) = edge
            freq += data[weight_value]

        graph_freq.update({node: freq})

    ave_strength_value = np.mean([k for k in graph_freq.values()])

    return ave_strength_value


def calculate_Qmax(G, mod_nodes):
    r"""returns maximum modularity possible given the network partition"""
    Lt = sum([G.degree(node) for node in list(G.nodes)])
    total = 0

    for mod in mod_nodes.keys():
        Lk = sum([G.degree(node) for node in mod_nodes[mod]])
        total += (1.0 * Lk / Lt) - (1.0 * Lk / Lt) ** 2

    return total


def graph_global_measures(g, pop_name):
    """ """
    # osnovne mjere
    total_nodes = len(g.nodes())
    total_edges = len(list(g.edges()))
    deg_list = [g.degree(node) for node in list(g.nodes)]
    average_degree = np.mean(deg_list)

    ave_strength_count = calculate_strength(g, "count")
    # ave_strength_duration = calculate_strength(g, "duration")
    # ave_strength_duration= calculate_strength(g, 'duration')
    try:
        edges_ave = total_edges / total_nodes

    except ZeroDivisionError:
        edges_ave = 0

    network_density = nx.density(g)

    # Measures of range
    gcc = sorted(nx.connected_components(g.to_undirected()), key=len, reverse=True)
    try:
        gc = g.to_undirected().subgraph(gcc[0])

        gc = list(max(nx.connected_components(g.to_undirected()), key=len))
        gc = g.to_undirected().subgraph(gc)

        spl = nx.average_shortest_path_length(gc)
        diameter = nx.diameter(gc, e=None)
        reach = nx.global_reaching_centrality(g, weight=None, normalized=True)

    except:
        gc = 0
        spl = 0
        diameter = 0
        reach = 0

    global_efficiency = nx.global_efficiency(g.to_undirected())

    ##########################################################################

    clustering_coeff = nx.clustering(g)
    average_cl_coeff_unweighted = np.mean([k for k in clustering_coeff.values()])
    clustering_coeff_w_count = nx.clustering(g, weight="count")
    average_cl_coeff_w_count = np.mean([k for k in clustering_coeff_w_count.values()])
    # clustering_coeff_w_duration = nx.clustering(g, weight="duration")
    # average_cl_coeff_w_duration = np.mean(
    #     [k for k in clustering_coeff_w_duration.values()]
    # )

    transitivity = nx.transitivity(g)

    degree_centrality = nx.degree_centrality(g)
    ave_deg_cent = np.mean([k for k in degree_centrality.values()])

    betweenness_centrality = nx.betweenness_centrality(g)
    average_betw_cent_unweighted = np.mean([k for k in betweenness_centrality.values()])
    betweenness_c_w_count = nx.betweenness_centrality(g, weight="count")
    average_betw_c_w_count = np.mean([k for k in betweenness_c_w_count.values()])
    # betweenness_c_w_duration = nx.betweenness_centrality(g, weight="duration")
    # average_betw_c_w_duration = np.mean([k for k in betweenness_c_w_duration.values()])

    closeness_centrality_unweighted = nx.closeness_centrality(g)
    ave_closeness_cent_unw = np.mean([k for k in closeness_centrality_unweighted.values()])
    closeness_c_w_count = nx.closeness_centrality(g, distance="count")
    ave_closeness_c_w_count = np.mean([k for k in closeness_c_w_count.values()])
    # closeness_c_w_duration = nx.closeness_centrality(g, distance="duration")
    # ave_closeness_c_w_duration = np.mean([k for k in closeness_c_w_duration.values()])

    try:
        standard_deviation_degree = round(np.std(deg_list))
        degree_heterogeneity = standard_deviation_degree / average_degree
        degree_assortativity = nx.degree_assortativity_coefficient(g)
    except:
        degree_heterogeneity, degree_assortativity = 0, 0
    # mjere za komponente
    ncc = nx.number_connected_components(g.to_undirected())
    # bggest component size
    gcc = sorted(nx.connected_components(g.to_undirected()), key=len, reverse=True)

    partition = community.best_partition(g.to_undirected())

    try:
        newman_modularity = community.modularity(partition, g.to_undirected(), weight="count")  #
    except:
        newman_modularity = 0
    modules = list(set(partition.values()))
    mod_nodes = {}
    for mod in modules:
        mod_nodes[mod] = [node for node in list(g.nodes) if partition[node] == mod]
    maximum_modularity = round(calculate_Qmax(g, mod_nodes), 4)

    try:
        relative_modularity = round(float(newman_modularity) / maximum_modularity, 3)

    except:
        relative_modularity = 0

    try:
        newman_modularity_count = community.modularity(partition, g, weight="count")  #
    except:
        newman_modularity_count = 0

    modules = list(set(partition.values()))
    mod_nodes = {}
    for mod in modules:
        mod_nodes[mod] = [node for node in list(g.nodes) if partition[node] == mod]
    maximum_modularity_count = round(calculate_Qmax(g, mod_nodes), 4)

    try:
        relative_modularity_count = round(float(newman_modularity_count) / maximum_modularity_count, 3)
    except:
        relative_modularity_count = 0

    d = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "ave_degree": average_degree,
        "ave_strength_count": ave_strength_count,
        # "ave_strength_duration": ave_strength_duration,
        "edges_ave": edges_ave,
        "network_density": network_density,
        "shortest_path_len": spl,
        "diameter": diameter,
        "reach": reach,
        "global_efficiency": global_efficiency,
        "ave_cl_coeff_unweighted": average_cl_coeff_unweighted,
        "ave_cl_coeff_w_count": average_cl_coeff_w_count,
        # "ave_cl_coeff_w_duration": average_cl_coeff_w_duration,
        "transitivity": transitivity,
        "ave_deg_centrality": ave_deg_cent,
        "ave_betw_cent_unweighted": average_betw_cent_unweighted,
        "ave_betw_c_w_count": average_betw_c_w_count,
        # "ave_betw_c_w_duration": average_betw_c_w_duration,
        "ave_closseness unweighted": ave_closeness_cent_unw,
        "ave_closseness_w_count": ave_closeness_c_w_count,
        # "ave_closseness_w_duration": ave_closeness_c_w_duration,
        "degree_heterogeneity": degree_heterogeneity,
        "degree_assortativity": degree_assortativity,
        "number_of_components": ncc,
        # "biggest_component size": bcs,
        "Newman_modularity_U": newman_modularity,
        "Newman_modularity": newman_modularity,
        "maximum_modularity": maximum_modularity,
        "relative_modularity": relative_modularity,
        "Newman_modularity count": newman_modularity_count,
        "maximum_modularity count": maximum_modularity_count,
        "relative_modularity count": relative_modularity_count,
    }

    df = pd.DataFrame(d, index=[pop_name.replace(".gml", "")])
    df = df.T

    return df


def group_comm_stats(G, group_name, weight):
    """Graph partitions found using Louvian algorithm."""
    partition = community.best_partition(G, weight=weight)

    communities, count = [], 0.0
    for c in set(partition.values()):
        count += 1.0
        list_nodes = [n for n in partition.keys() if partition[n] == c]
        communities.append(list_nodes)

    communities.sort(key=len, reverse=True)

    single_element_comm = len([c for c in communities if len(c) == 1])
    all_comm_len_no_sing = [len(com) for com in communities if len(com) > 1]
    ave_comm_size_no_sing = sum(all_comm_len_no_sing) / len(all_comm_len_no_sing)

    if len(communities) > 1:
        second_biggest_community = len(communities[1])
    else:
        second_biggest_community = 0

    d = {
        "number of nodes: ": len(G.nodes()),
        "comm_size=1 (single nodes):": single_element_comm,
        "percentage of single nodes: ": float(single_element_comm / len(G.nodes())),
        "number of communities: ": len(communities),
        "comm_size>1": len(communities) - single_element_comm,
        "biggest_community_size:": len(communities[0]),
        "second_biggest_community:": second_biggest_community,
        "ave_comm_size_no_sing:": ave_comm_size_no_sing,
    }

    col_name = group_name.replace(".gml", "")
    df = pd.DataFrame(d, index=[f"{col_name} weight={weight}"])

    return df.T


def local_measures_functions():
    """Return list of tuples. Each tuple consists of two values.
    First one is string name of the funciton and second is function.

    Returns:
        list: list of tuples
    """

    graph_functions = [
        ("Degree centrality", lambda g: nx.degree_centrality(g)),
        ("In-degree centrality", lambda g: nx.in_degree_centrality(g)),
        ("Out-degree centrality", lambda g: nx.out_degree_centrality(g)),
        ("Eigenvector centrality", lambda g: nx.eigenvector_centrality(g)),
        ("Closeness centrality", lambda g: nx.closeness_centrality(g)),
        ("Strength distribution, weight=count", lambda g: calculate_strength(g, "count")),
        ("Betweenness centrality weight=None", lambda g: nx.betweenness_centrality(g, weight=None)),
        ("Betweenness centrality weight=count", lambda g: nx.betweenness_centrality(g, weight="count")),
        ("Clustering coefficient weight=None", lambda g: nx.clustering(g, weight=None)),
        ("Clustering coefficient weight=count", lambda g: nx.clustering(g, weight="count")),
        ("PageRank centrality", lambda g: nx.pagerank(g)),
        ("Degree", lambda g: dict(nx.degree(g))),
        ("In-degree", lambda g: dict(g.in_degree())),
        ("Out-degree", lambda g: dict(g.out_degree()))
    ]

    return graph_functions
