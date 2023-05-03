import community

import numpy as np
import pandas as pd
import networkx as nx


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
    ave_strength_duration = calculate_strength(g, "duration")
    ##ave_strength_duration= calculate_strength(g, 'duration')

    edges_ave = total_edges / total_nodes
    network_density = nx.density(g)

    # Measures of range
    # gcc = sorted(nx.connected_component(g), key=len, reverse=True)
    # gc = g.subgraph(gcc[0])
    gc = list(max(nx.connected_components(g), key=len))
    gc = g.subgraph(gc)

    spl = nx.average_shortest_path_length(gc)
    diameter = nx.diameter(gc, e=None)

    # Number of edges separating the focal node from other nodes of interest
    reach = nx.global_reaching_centrality(g, weight=None, normalized=True)
    global_efficiency = nx.global_efficiency(g)

    ##########################################################################
    # mjere grupiranja
    clustering_coeff = nx.clustering(g)
    average_cl_coeff_unweighted = np.mean([k for k in clustering_coeff.values()])
    clustering_coeff_w_count = nx.clustering(g, weight="count")
    average_cl_coeff_w_count = np.mean([k for k in clustering_coeff_w_count.values()])
    clustering_coeff_w_duration = nx.clustering(g, weight="duration")
    average_cl_coeff_w_duration = np.mean(
        [k for k in clustering_coeff_w_duration.values()]
    )

    transitivity = nx.transitivity(g)

    degree_centrality = nx.degree_centrality(g)
    ave_deg_cent = np.mean([k for k in degree_centrality.values()])

    betweenness_centrality = nx.betweenness_centrality(g)
    average_betw_cent_unweighted = np.mean([k for k in betweenness_centrality.values()])
    betweenness_c_w_count = nx.betweenness_centrality(g, weight="count")
    average_betw_c_w_count = np.mean([k for k in betweenness_c_w_count.values()])
    betweenness_c_w_duration = nx.betweenness_centrality(g, weight="duration")
    average_betw_c_w_duration = np.mean([k for k in betweenness_c_w_duration.values()])

    closeness_centrality_unweighted = nx.closeness_centrality(g)
    ave_closeness_cent_unw = np.mean(
        [k for k in closeness_centrality_unweighted.values()]
    )
    closeness_c_w_count = nx.closeness_centrality(g, distance="count")
    ave_closeness_c_w_count = np.mean([k for k in closeness_c_w_count.values()])
    closeness_c_w_duration = nx.closeness_centrality(g, distance="duration")
    ave_closeness_c_w_duration = np.mean([k for k in closeness_c_w_duration.values()])

    standard_deviation_degree = round(np.std(deg_list))
    degree_heterogeneity = standard_deviation_degree / average_degree
    degree_assortativity = nx.degree_assortativity_coefficient(g)

    # mjere za komponente
    ncc = nx.number_connected_components(g)
    # bggest component size
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    bcs = len(g.subgraph(gcc[0]))

    partition = community.best_partition(g)

    newman_modularity = community.modularity(partition, g, weight="count")  #
    modules = list(set(partition.values()))
    mod_nodes = {}
    for mod in modules:
        mod_nodes[mod] = [node for node in list(g.nodes) if partition[node] == mod]
    maximum_modularity = round(calculate_Qmax(g, mod_nodes), 4)
    relative_modularity = round(float(newman_modularity) / maximum_modularity, 3)

    newman_modularity_D = community.modularity(partition, g, weight="duration")  #
    modules = list(set(partition.values()))
    mod_nodes = {}
    for mod in modules:
        mod_nodes[mod] = [node for node in list(g.nodes) if partition[node] == mod]
    maximum_modularity_D = round(calculate_Qmax(g, mod_nodes), 4)
    relative_modularity_D = round(float(newman_modularity_D) / maximum_modularity_D, 3)

    d = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "ave_degree": average_degree,
        "ave_strength_count": ave_strength_count,
        "ave_strength_duration": ave_strength_duration,
        "edges_ave": edges_ave,
        "network_density": network_density,
        "shortest_path_len": spl,
        "diameter": diameter,
        "reach": reach,
        "global_efficiency": global_efficiency,
        "ave_cl_coeff_unweighted": average_cl_coeff_unweighted,
        "ave_cl_coeff_w_count": average_cl_coeff_w_count,
        "ave_cl_coeff_w_duration": average_cl_coeff_w_duration,
        "transitivity": transitivity,
        "ave_deg_centrality": ave_deg_cent,
        "ave_betw_cent_unweighted": average_betw_cent_unweighted,
        "ave_betw_c_w_count": average_betw_c_w_count,
        "ave_betw_c_w_duration": average_betw_c_w_duration,
        "ave_closseness unweighted": ave_closeness_cent_unw,
        "ave_closseness_w_count": ave_closeness_c_w_count,
        "ave_closseness_w_duration": ave_closeness_c_w_duration,
        "degree_heterogeneity": degree_heterogeneity,
        "degree_assortativity": degree_assortativity,
        "number_of_components": ncc,
        "biggest_component size": bcs,
        "Newman_modularity_U": newman_modularity,
        "Newman_modularity": newman_modularity,
        "maximum_modularity": maximum_modularity,
        "relative_modularity": relative_modularity,
        "Newman_modularity D": newman_modularity_D,
        "maximum_modularity D": maximum_modularity_D,
        "relative_modularity D": relative_modularity_D,
    }

    df = pd.DataFrame(d, index=[pop_name.replace(".gml", "")])
    df = df.T

    return df
