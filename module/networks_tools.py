import os
import networkx as nx
import networkx.algorithms.community as nxcom

import matplotlib.pyplot as plt
import community
import numpy as np
import pandas as pd

import random
from numpy import random as nprand
random.seed(123)
nprand.seed(123)

from statistics import mean, stdev
import scipy.stats

import logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

def add_edges_to_undirected_g(G, df,
                              DISTANCE_BETWEEN_FLIES,
                              TOUCH_DURATION_FRAMES,
                              FPS):
    """[summary]

    Args:
        G ([type]): [description]
        df ([type]): [description]
        DISTANCE_BETWEEN_FLIES ([type]): [description]
        TOUCH_DURATION_FRAMES ([type]): [description]
        FPS ([type]): [description]

    Returns:
        [type]: [description]
    """

    for column in df.columns:
        df_ = df[column]

        df_ = df_[df_ <= DISTANCE_BETWEEN_FLIES+1]

        clear_list_of_df = [d for _, d in df_.groupby(df_.index - np.arange(len(df_))) if len(d) >= TOUCH_DURATION_FRAMES]

        node_1, node_2 = column.split(' ')

        if node_1 not in G:
            G.add_node(node_1)

        if node_2 not in G:
            G.add_node(node_2)

        count_all_interactions = len(clear_list_of_df)
        duration_all_interactions = sum([len(series) for series in clear_list_of_df])


        if count_all_interactions >= 1:
            duration=float(duration_all_interactions/FPS)
            count=int(count_all_interactions)

            frequency=float((count/duration)/0.5)

            averaging=float(1/599.5)*(float(duration/count)-0.5)

            G.add_edge(node_1, node_2,
                       duration=duration,
                       count=count,
                       frequency=frequency,
                       averaging=averaging)

    return G


def add_multiedges_to_undirected_g(G, df, DISTANCE_BETWEEN_FLIES,
                                   TOUCH_DURATION_FRAMES):
    """[summary]

    Args:
        G ([type]): [description]
        df ([type]): [description]
        DISTANCE_BETWEEN_FLIES ([type]): [description]
        TOUCH_DURATION_FRAMES ([type]): [description]

    Returns:
        [type]: [description]
    """


    for column in df.columns:
        df_ = df[column]
        df_ = df_[df_ <= DISTANCE_BETWEEN_FLIES+1]
        clear_list_of_df = [d for _, d in
                            df_.groupby(df_.index - np.arange(len(df_))) if
                            len(d) >= TOUCH_DURATION_FRAMES]

        if len(clear_list_of_df) >= 1:
            node_1, node_2 = column.split(' ')

            if node_1 not in G:
                G.add_node(node_1)

            if node_2 not in G:
                G.add_node(node_2)

            for interaction in clear_list_of_df:
                G.add_edge(node_1, node_2,
                            duration=len(interaction))

    return G


def dynamics_of_network_graph(G, df, DISTANCE_BETWEEN_FLIES,
                              TOUCH_DURATION_FRAMES):
    """[summary]

    Args:
        G ([type]): [description]
        df ([type]): [description]
        DISTANCE_BETWEEN_FLIES ([type]): [description]
        TOUCH_DURATION_FRAMES ([type]): [description]

    Returns:
        [type]: [description]
    """

    for column in df.columns:
        df_ = df[column]
        df_ = df_[df_ <= DISTANCE_BETWEEN_FLIES+1]
        clear_list_of_df = [d for _, d in
                            df_.groupby(df_.index - np.arange(len(df_))) if
                            len(d) >= TOUCH_DURATION_FRAMES]

        if len(clear_list_of_df) >= 1:
            node_1, node_2 = column.split(' ')
            for interaction in clear_list_of_df:
                G.add_edge(node_1, node_2,
                            start=interaction.index[0],
                            end=interaction.index[-1])

    return G


def calculate_Qmax(G, mod_nodes):
    """Returns maximum modularity possible given the network partition

    Args:
        G ([type]): [description]
        mod_nodes ([type]): [description]

    Returns:
        [type]: [description]
    """

    Lt= sum([G.degree(node) for node in list(G.nodes)])
    total  =0

    for mod in mod_nodes.keys():
        Lk = sum([G.degree(node) for node in mod_nodes[mod]])
        total+= (1.0*Lk/Lt) - (1.0*Lk/Lt)**2

    return total


def calculate_avg_wd(G, partition, n_nodes):
    """ Returns average within-module degree

    Args:
        G ([type]): [description]
        partition ([type]): [description]
        n_nodes ([type]): [description]

    Returns:
        [type]: [description]
    """

    wdlist = []

    for node1 in list(G.nodes):
        nbrs = G.neighbors(node1)
        mod1 = partition[node1]
        mod_nbrs = [node2 for node2 in nbrs if partition[node2]==mod1]
        wd = len(mod_nbrs)
        wdlist.append(wd)

    return sum(wdlist)/(1.*n_nodes)


def calculate_strength(g, weight_value):
    """[summary]

    Args:
        g ([type]): [description]
        weight_value ([type]): [description]

    Returns:
        [type]: [description]
    """

    graph_freq = {}
    nodes = list(g.nodes)
    for node in nodes:
        edges = list(g.edges(node, data=True))

        freq = 0
        for edge in edges:
            (_, _, data) = edge
            freq += data[weight_value]

        graph_freq.update({int(node):freq})

    ave_strength_value = mean(graph_freq[k] for k in graph_freq)

    return ave_strength_value


def order_columns(df):
    """[summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """

    coc_columns = [col for col in df if col.startswith('COC')]
    ctrl_columns = [col for col in df if col.startswith('CTRL')]

    df_coc = df[coc_columns]
    df_ctrl = df[ctrl_columns]

    df = pd.DataFrame()
    df['median_COC'] = df_coc.loc[:, :].median(axis=1)
    df['mean_COC'] = df_coc.loc[:, :].mean(axis=1)
    df['std_COC'] = df_coc.loc[:, :].sem(axis=1)

    df['median_CTRL'] = df_ctrl.loc[:, :].median(axis=1)
    df['mean-CTRL'] = df_ctrl.loc[:, :].mean(axis=1)
    df['std_CTRL'] = df_ctrl.loc[:, :].sem(axis=1)

    df = pd.concat([df, df_ctrl, df_coc], axis=1)

    return df


def convert_multigraph_to_weighted(multiGraph, FPS):
    """
    Args:
        multiGraph ([type]): [description]
        FPS ([type]): [description]

    Returns:
        [type]: [description]
    """

    G = nx.Graph()
    G.add_nodes_from(multiGraph.nodes())

    for u,v,data in multiGraph.edges(data=True):
        if G.has_edge(u,v):
            G[u][v]['duration'] += data['duration']
            G[u][v]['count'] +=1

            duration = G[u][v]['duration']
            count = G[u][v]['count']
            frequency=float((count/duration)/0.5)
            averaging=float(1/599.5)*(float(duration/count)-0.5)

        else:
            if data['duration'] > 0:
                duration=float(data['duration']/FPS)
                count=1
                frequency=float((count/duration)/0.5)
                averaging=float(1/599.5)*(float(duration/count)-0.5)

                G.add_edge(u, v,
                           duration=duration,
                           count=count,
                           frequency=frequency,
                           averaging=averaging)
            else:
                continue

    return G


def remove_nodes_with_degree_less_than(G, degree):
    """[summary]

    Args:
        G ([type]): [description]
        degree ([type]): [description]

    Returns:
        [type]: [description]
    """

    remove = [node for node, node_degree in dict(G.degree()).items() if node_degree < degree]
    G.remove_nodes_from(remove)

    return G


def get_x_labels(my_dict_keys):
    """[summary]

    Args:
        my_dict_keys ([type]): [description]

    Returns:
        [type]: [description]
    """

    my_dict_keys = list(my_dict_keys)
    coc_count = 1
    ctrl_count = 1
    new_values = []

    for item in my_dict_keys:
        if item.startswith('CTRL'):
            new_values.append('CTRL_'+str(ctrl_count))
            ctrl_count += 1

        else:
            new_values.append('COC_'+str(coc_count))
            coc_count += 1

    return new_values


def get_strengtgs_dict(g, weight_value):
    """[summary]

    Args:
        g ([type]): [description]
        weight_value ([type]): [description]

    Returns:
        [type]: [description]
    """

    nodes = list(g.nodes)
    strength_d = {}

    for node in nodes:
        edges = list(g.edges(node, data=True))
        freq = float(0)

        for edge in edges:
            (_, _, data) = edge
            freq += data[weight_value]

        strength_d.update({int(node):freq})

    #ave_strength_value = mean(graph_freq[k] for k in graph_freq)

    return strength_d


def draw_box_plot(d, graph_title):
    """[summary]

    Args:
        dictionary ([type]): [description]
        graph_title ([type]): [description]

    Returns:
        [type]: [description]
    """

    labels, data = [*zip(*d.items())]
    labels = [label for label in labels]
    data = [list(d.values()) for d in data]

    fig = plt.figure(figsize=(9,6))
    _ = plt.boxplot(data)

    return fig


def group_values(multiple_dicts):
    """[summary]

    Args:
        multiple_dicts ([type]): [description]

    Returns:
        [type]: [description]
    """

    d = {}
    coc_values = []
    ctrl_values = []
    for pop_name, values in multiple_dicts.items():

        for _, value in values.items():
            if pop_name.startswith('COC'):
                coc_values.append(value)

            else:
                ctrl_values.append(value)

    d.update({'COC': coc_values})
    d.update({'CTRL': ctrl_values})

    return d


def graph_global_measures(g, pop_name):
    """
    Args:
        g ([type]): [description]
        pop_name ([type]): [description]

    Returns:
        [type]: [description]
    """

    # osnovne mjere
    total_nodes = len(g.nodes())
    total_edges = len(list(g.edges()))
    deg_list = [g.degree(node) for node in list(g.nodes)]
    average_degree = np.mean(deg_list)

    ave_strength_count = calculate_strength(g, 'count')
    ave_strength_duration= calculate_strength(g, 'duration')

    edges_ave = total_edges / total_nodes
    network_density = nx.density(g)

    gc = list(max(nx.connected_components(g), key=len))
    gc = g.subgraph(gc)

    spl = nx.average_shortest_path_length(gc)
    diameter = nx.diameter(gc, e=None)

    reach = nx.global_reaching_centrality(g, weight=None, normalized=True)
    global_efficiency = nx.global_efficiency(g)

    clustering_coeff = nx.clustering(g)
    average_cl_coeff_unweighted = mean(clustering_coeff[k] for k in clustering_coeff)
    clustering_coeff_w_count = nx.clustering(g, weight='count')
    average_cl_coeff_w_count = mean(clustering_coeff_w_count[k] for k in clustering_coeff_w_count)
    clustering_coeff_w_duration = nx.clustering(g, weight='duration')
    average_cl_coeff_w_duration = mean(clustering_coeff_w_duration[k] for k in clustering_coeff_w_duration)

    transitivity = nx.transitivity(g)

    degree_centrality = nx.degree_centrality(g)
    ave_deg_cent = mean(degree_centrality[k] for k in degree_centrality)

    betweenness_centrality = nx.betweenness_centrality(g)
    average_betw_cent_unweighted = mean(betweenness_centrality[k] for k in betweenness_centrality)
    betweenness_c_w_count = nx.betweenness_centrality(g, weight='count')
    average_betw_c_w_count = mean(betweenness_c_w_count[k] for k in betweenness_c_w_count)
    betweenness_c_w_duration = nx.betweenness_centrality(g, weight='duration')
    average_betw_c_w_duration = mean(betweenness_c_w_duration[k] for k in betweenness_c_w_duration)

    closeness_centrality_unweighted = nx.closeness_centrality(g)
    ave_closeness_cent_unw = mean(closeness_centrality_unweighted[k] for k in closeness_centrality_unweighted)
    closeness_c_w_count = nx.closeness_centrality(g,distance='count')
    ave_closeness_c_w_count = mean(closeness_c_w_count[k] for k in closeness_c_w_count)
    closeness_c_w_duration = nx.closeness_centrality(g,distance='duration')
    ave_closeness_c_w_duration = mean(closeness_c_w_duration[k] for k in closeness_c_w_duration)

    standard_deviation_degree =round(np.std(deg_list))
    degree_heterogeneity = standard_deviation_degree/average_degree
    degree_assortativity = nx.degree_assortativity_coefficient(g)

    ncc = nx.number_connected_components(g)
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    bcs = len(g.subgraph(gcc[0]))

    partition = community.best_partition(g)

    newman_modularity = community.modularity(partition, g, weight='count') #
    modules = list(set(partition.values()))
    mod_nodes= {}
    for mod in modules:
        mod_nodes[mod] = [node for node in list(g.nodes) if partition[node]==mod]
    maximum_modularity = round(calculate_Qmax(g, mod_nodes),4)
    relative_modularity = round(float(newman_modularity)/maximum_modularity,3)

    newman_modularity_D = community.modularity(partition, g, weight='duration') #
    modules = list(set(partition.values()))
    mod_nodes= {}
    for mod in modules:
        mod_nodes[mod] = [node for node in list(g.nodes) if partition[node]==mod]
    maximum_modularity_D = round(calculate_Qmax(g, mod_nodes),4)
    relative_modularity_D = round(float(newman_modularity_D)/maximum_modularity_D,3)

    d = {'total_nodes': total_nodes,
         'total_edges': total_edges,
         'ave_degree': average_degree,
         'ave_strength_count': ave_strength_count,
         'ave_strength_duration': ave_strength_duration,
         'edges_ave':edges_ave,
         'network_density':network_density,
         'shortest_path_len': spl,
         'diameter': diameter,
         'reach':reach,
         'global_efficiency':global_efficiency,
         'ave_cl_coeff_unweighted': average_cl_coeff_unweighted,
         'ave_cl_coeff_w_count': average_cl_coeff_w_count,
         'ave_cl_coeff_w_duration': average_cl_coeff_w_duration,
         'transitivity': transitivity,
         'ave_deg_centrality': ave_deg_cent,
         'ave_betw_cent_unweighted': average_betw_cent_unweighted,
         'ave_betw_c_w_count': average_betw_c_w_count,
         'ave_betw_c_w_duration': average_betw_c_w_duration,
         'ave_closseness unweighted': ave_closeness_cent_unw,
         'ave_closseness_w_count': ave_closeness_c_w_count,
         'ave_closseness_w_duration': ave_closeness_c_w_duration,
         'degree_heterogeneity': degree_heterogeneity,
         'degree_assortativity': degree_assortativity,
         'number_of_components': ncc,
         'biggest_component size': bcs,
         'Newman_modularity_U': newman_modularity,
         'Newman_modularity': newman_modularity,
         'maximum_modularity': maximum_modularity,
         'relative_modularity': relative_modularity,
         'Newman_modularity D': newman_modularity_D,
         'maximum_modularity D': maximum_modularity_D,
         'relative_modularity D': relative_modularity_D}

    df = pd.DataFrame(d, index=[pop_name.replace('.gml','')])
    df = df.T

    return df


def single_pop_comm_stats(G, pop_name, weight=None):
    #first compute the best partition
    partition = community.best_partition(G, weight=weight)
    #size = float(len(set(partition.values())))
    count = 0.
    communities = []
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        communities.append(list_nodes)

    communities.sort(key=len, reverse=True)
    single_element_comm = 0
    for com in communities:
        if(len(com) == 1):
            single_element_comm +=1

    #list of comm, with size of each comm.
    #all_comm_len = [len(com) for com in communities]
    
    all_comm_len_no_sing = [len(com) for com in communities if len(com) > 1]
    ave_comm_size_no_sing = sum(all_comm_len_no_sing)/len(all_comm_len_no_sing)

    d = {'number of nodes: ': len(G.nodes()),
         'no_comm_size=1 (single nodes):': single_element_comm,
         'percentage of single nodes: ': float(single_element_comm/len(G.nodes())),
         'number of communities: ': len(communities),
         'no_comm_size>1': len(communities)-single_element_comm,
         'biggest_community_size:': len(communities[0]),
         'second_biggest_community:': len(communities[1]),
         'ave_comm_size_no_sing:':ave_comm_size_no_sing}

    df = pd.DataFrame(d, index=[pop_name])
    df = df.T

    return df


def comm_stats(experiments, weight='count'):
    """
    """

    total = pd.DataFrame()

    for pop_name, path in experiments.items():
        G = nx.read_gml(path)
        df = single_pop_comm_stats(G, pop_name, weight=weight)
        total = pd.concat([total,df], axis=1)

    return total


def network_measures_distribution():
    """[summary]

    Returns:
        [type]: [description]
    """

    graph_functions = [
        ('Degree centrality', lambda g: nx.degree_centrality(g)),
        ('Eigenvector centrality', lambda g: nx.eigenvector_centrality(g)),
        ('Closeness centrality', lambda g: nx.closeness_centrality(g)),
        ('Information centrality', lambda g: nx.information_centrality(max(nx.connected_component_subgraphs(g), key=len))),
        ('Page rank', lambda g: nx.pagerank(g, alpha=0.9)),
        ('Strength distribution, weight=duration', lambda g: get_strengtgs_dict(g, 'duration')),
        ('Strength distribution, weight=count', lambda g: get_strengtgs_dict(g, 'count')),
        ('Strength distribution, weighy=frequency', lambda g: get_strengtgs_dict(g, 'frequency')),
        ('Strength distribution, weight=average', lambda g: get_strengtgs_dict(g, 'averaging')),
        ('Betweenness centrality weight=None', lambda g: nx.betweenness_centrality(g, weight=None)),
        ('Betweenness centrality weight=duration', lambda g: nx.betweenness_centrality(g, weight='duration')),
        ('Betweenness centrality weight=count', lambda g: nx.betweenness_centrality(g, weight='count')),
        ('Betweenness centrality weight=frequency', lambda g: nx.betweenness_centrality(g, weight='frequency')),
        ('Betweenness centrality weight=averaging', lambda g: nx.betweenness_centrality(g, weight='averaging')),
        ('Clustering coefficient weight=duration', lambda g: nx.clustering(g, weight='duration')),
        ('Clustering coefficient weight=count', lambda g: nx.clustering(g, weight='count')),
        ('Clustering coefficient weight=frequency', lambda g: nx.clustering(g, weight='frequency')),
        ('Clustering coefficient weight=averaging', lambda g: nx.clustering(g, weight='averaging'))
        ]

    return graph_functions
