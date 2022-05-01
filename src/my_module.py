import os
import re
import sys
import random

import community
import numpy as np
import pandas as pd
import networkx as nx

import scipy.stats
from statistics import mean
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def natural_sort(l):
    """
    Naturaly sort list of strings and return sorted.

    Input: ['file_1', 'file_10', 'file_11', ..., 'file_2', 'file_20', 'file_21']
    Output: ['file_1', file_2, ..., 'file_10', 'file_11', ..., 'file_20', 'file_21']
    """

    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def load_multiple_folders(path):
    """
    Import foldera sa vise foldera unutar kojih su csv podaci 
    Returns files with given type from folder. If no files are found SystemExit 
    is raised and the script stops running.

    Parameters
    ----------
    path : str
        variable description

    file_format : str

    Returns
    -------
    variable : type
        variable description

    """

    if not os.listdir(path):
        sys.exit('Directory is empty')

    experiments = {}
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for folder in d:
            experiments.update({folder: os.path.join(r, folder)})

    return experiments


def load_files_from_folder(path, file_format):
    """
    Returns files with given type from folder. If no files are found SystemExit 
    is raised and the script stops running.

    Parameters
    ----------
    path : str
        variable description

    file_format : str

    Returns
    -------
    found_files : dict
        variable description
    """

    if not os.listdir(path):
        sys.exit('Directory is empty')

    found_files = {}

    for r, d, f in os.walk(path):
        sorted_files = natural_sort(f)
        for file in f:
            if file_format in sorted_files:
                found_files.update({file: os.path.join(r, file)})
            else:
                sys.exit('File format: {} not found in {}'.format(
                    file_format, path))

    return found_files


def load_dfs_to_list(path, min_x, min_y, file_format='.csv'):
    """
    Takes folder with individuals and returns list of dataframes for each
    individual.

    Parameters
    ----------
    variable : type
        variable description

    Returns
    -------
    variable : type
        variable description

    """

    if not os.listdir(path):
        sys.exit('Directory is empty')

    files_dict = {}

    for r, d, f in os.walk(path):
        f = natural_sort(f)
        for file in f:
            if file_format in file:
                files_dict.update({file: os.path.join(r, file)})

    df_list = []
    for fly_name, fly_path in files_dict.items():
        df = pd.read_csv(fly_path, index_col=0)
        df = prepproc(df, min_x, min_y)
        df = round_coordinates(df, decimal_places=0)
        df = df[['pos_x', 'pos_y']]
        df_list.append(df)

    return df_list


def check_data(path):
    """AI is creating summary for check_data

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """

    columns = ['pos x', 'pos y', 'ori', 'major axis len', 'minor axis len',
               'body area', 'fg area', 'img contrast', 'min fg dist', 'wing l x',
               'wing l y', 'wing r x', 'wing r y', 'wing l ang', 'wing l len',
               'wing r ang', 'wing r len', 'leg 1 x', 'leg 1 y', 'leg 2 x', 'leg 2 y',
               'leg 3 x', 'leg 3 y', 'leg 4 x', 'leg 4 y', 'leg 5 x', 'leg 5 y',
               'leg 6 x', 'leg 6 y', 'leg 1 ang', 'leg 2 ang', 'leg 3 ang',
               'leg 4 ang', 'leg 5 ang', 'leg 6 ang', 'vel', 'ang_vel', 'min_wing_ang',
               'max_wing_ang', 'mean_wing_length', 'axis_ratio', 'fg_body_ratio',
               'contrast', 'dist_to_wall']

    valid_data = False
    fly_dict = load_files_from_folder(path)
    valid_files_count = 0

    for fly_name, path in fly_dict.items():
        df = pd.read_csv(path, index_col=0)
        df_columns = list(df.columns)

        if df_columns == columns:
            valid_files_count += 1

    if len(fly_dict) == valid_files_count:
        valid_data = True

    return valid_data


def round_coordinates(df, decimal_places=0):
    # zaokruzivanje vrijednosti koordinata x i y na 0 decimala
    df = df.round({'pos_x': decimal_places, 'pos_y': decimal_places})

    return df


def prepproc(df, min_x, min_y):
    """_summary_

    Args:
        df (_type_): _description_
        min_x (_type_): _description_
        min_y (_type_): _description_

    Returns:
        _type_: _description_
    """

    # fill nan values
    #df = df.where(df.notnull(), other=(df.fillna(method='ffill')+df.fillna(method='bfill'))/2)
    df = df.fillna(method='ffill')
    df.columns = df.columns.str.replace(' ', '_')

    df['pos_x'] = df.pos_x.subtract(min_x)
    df['pos_y'] = df.pos_y.subtract(min_y)

    # provjera podataka ako su nan, ciscenje i popunjavanje praznih
    # if df['pos_x'].isnull().values.any() or df['pos_y'].isnull().values.any():
    #     raise TypeError("Nan value found!")
    # else:
    #     ## oduzimanje najmanje vrijednosti od svih vrijednosti u stupcu
    #     df['pos_x'] = df.pos_x.subtract(min(df['pos_x']))
    #     df['pos_y'] = df.pos_y.subtract(min(df['pos_y']))
    #df['ori'] = np.rad2deg(df['ori'])

    return df


def find_pop_mins(path):
    """

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    fly_dict = load_files_from_folder(path)

    pop_min_x = []
    pop_min_y = []
    for fly_name, path in fly_dict.items():

        df = pd.read_csv(path, index_col=0)
        pop_min_x.append(min(df['pos x']))
        pop_min_y.append(min(df['pos y']))

    return min(pop_min_x), min(pop_min_y)


def inspect_population_coordinates(path, pop_name):
    """ Draws scatter plot of x and y coordinates in population.
    This function is used to inspect validity of trackings if all coordinates 
    are inside of the arena.
    """
    x_pop_all = pd.Series()
    y_pop_all = pd.Series()
    fly_dict = load_files_from_folder(path)

    for fly_name, path in fly_dict.items():

        df = pd.read_csv(path, usecols=['pos_x', 'pos_y'])
        df = df.dropna()

        x_pop_all = pd.concat([x_pop_all, df.pos_x], axis=0)
        y_pop_all = pd.concat([y_pop_all, df.pos_y], axis=0)

    x = x_pop_all
    y = y_pop_all

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.plot(x, y, 'k.', markersize=0.05)

    plt.title('movements of ' + pop_name)
    plt.show()


# =============================================================================
""" TRAJECTORY DESCRIPTIVE AND FEATTURE CREATING FUNCTIONS """
# =============================================================================


def get_acc_in_path(vel_list):
    list_of_acc = []

    while len(vel_list) > 1:
        list_of_acc.append(vel_list[1]-vel_list[0])
        vel_list.pop(0)

    return list_of_acc


def descriptor(list_of_values, value_name):
    # returns dictionary of statistical values descriptors
    #list_of_values = np.array(list_of_values)

    min_ = min(list_of_values)
    max_ = max(list_of_values)
    mean = np.mean(list_of_values)
    median = np.median(list_of_values)
    std = np.std(list_of_values)
    p10 = np.percentile(list_of_values, 10)
    p25 = np.percentile(list_of_values, 25)
    p75 = np.percentile(list_of_values, 75)
    p90 = np.percentile(list_of_values, 90)

    stat_dict = {
        'min_' + value_name + '_value': min_,
        'max_' + value_name + '_value': max_,
        'mean_' + value_name + '_value': mean,
        'median_' + value_name + '_value': median,
        'std_' + value_name + '_value': std,
        'p10_' + value_name + '_value': p10,
        'p25_' + value_name + '_value': p25,
        'p75_' + value_name + '_value': p75,
        'p90_' + value_name + '_value': p90}

    return stat_dict


def get_path_values(df, window_size):
    """ PATH VALUES BASED ON STEP LENGTH"""
    df['step'] = np.sqrt(np.square(df['pos_x'].diff()) +
                         np.square(df['pos_y'].diff()))

    df['spl'] = np.sqrt(np.square(df['pos_x'].diff(periods=window_size)) +
                        np.square(df['pos_y'].diff(periods=window_size)))

    df['rpl'] = df['step'].rolling(window_size).sum()
    df['str_index'] = df['spl']/df['rpl']
    df['abs_change_x'] = df['pos_x'].diff(periods=window_size).abs()
    df['abs_change_y'] = df['pos_y'].diff(periods=window_size).abs()

    df = df.fillna(0)

    return df


def df_descriptor(df, value, window_size):

    df[value + '_min'] = df[value].rolling(window_size).min()
    df[value + '_max'] = df[value].rolling(window_size).max()
    df[value + '_mean'] = df[value].rolling(window_size).mean()
    df[value + '_median'] = df[value].rolling(window_size).median()
    df[value + '_std'] = df[value].rolling(window_size).std()
    df[value + '_p10'] = df[value].rolling(window_size).quantile(0.1)
    df[value + '_p25'] = df[value].rolling(window_size).quantile(0.25)
    df[value + '_p75'] = df[value].rolling(window_size).quantile(0.75)
    df[value + '_p90'] = df[value].rolling(window_size).quantile(0.90)

    return df


def min_max_normalization_df(df):

    return (df-df.min())/(df.max()-df.min())


# =============================================================================
""" POPULATION ANALYSIS, BETWEEN DISTANCES, NETWORK CONSTRUCTION"""
# =============================================================================


def distances_between_all_flies(files):
    """
    Input
    -----
    List of dataframes
    Returns
    -------
    Dataframe of all distances between flies.
    """

    final_df = pd.DataFrame()
    for i in range(len(files)):

        df1 = files[i]

        next_flie = i + 1

        if next_flie <= len(files):
            for j in range(next_flie, len(files)):
                df2 = files[j]

                df = pd.concat([df1['pos_x'], df1['pos_y'],
                                df2['pos_x'], df2['pos_y']], axis=1)

                df.columns = ['pos_x1', 'pos_y1', 'pos_x2', 'pos_y2']

                df['x_axis_dif'] = (df['pos_x1'] - df['pos_x2']).abs()
                df['y_axis_dif'] = (df['pos_y1'] - df['pos_y2']).abs()

                name = str(i+1) + ' ' + str(j+1)
                final_df[name] = np.sqrt(np.square(df['x_axis_dif']) +
                                         np.square(df['y_axis_dif']))

    return final_df


def dist_flie_to_others(fly_name):
    """
    Returns fly distances compared to other flies
    Parameters
    ----------
    fly_name : TYPE
        DESCRIPTION.
    Returns
    -------
    df_fly_dist : TYPE
        DESCRIPTION.
    """
    df_fly_dist = 0

    return df_fly_dist


def add_edges_to_undirected_g(G, df,
                              DISTANCE_BETWEEN_FLIES,
                              TOUCH_DURATION_FRAMES,
                              FPS):

    for column in df.columns:
        df_ = df[column]

        df_ = df_[df_ <= DISTANCE_BETWEEN_FLIES+1]

        clear_list_of_df = [d for _, d in df_.groupby(
            df_.index - np.arange(len(df_))) if len(d) >= TOUCH_DURATION_FRAMES]

        node_1, node_2 = column.split(' ')

        if node_1 not in G:
            G.add_node(node_1)

        if node_2 not in G:
            G.add_node(node_2)

        count_all_interactions = len(clear_list_of_df)
        duration_all_interactions = sum(
            [len(series) for series in clear_list_of_df])

        if count_all_interactions >= 1:
            duration = float(duration_all_interactions/FPS)
            count = int(count_all_interactions)

            frequency = float((count/duration)/0.5)

            averaging = float(1/599.5)*(float(duration/count)-0.5)

            G.add_edge(node_1, node_2,
                       duration=duration,
                       count=count,
                       frequency=frequency,
                       averaging=averaging)

    return G


def add_multiedges_to_undirected_g(G, df, DISTANCE_BETWEEN_FLIES,
                                   TOUCH_DURATION_FRAMES):

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
    r"""returns maximum modularity possible given the network partition"""
    Lt = sum([G.degree(node) for node in list(G.nodes)])
    total = 0

    for mod in mod_nodes.keys():
        Lk = sum([G.degree(node) for node in mod_nodes[mod]])
        total += (1.0*Lk/Lt) - (1.0*Lk/Lt)**2

    return total


def calculate_avg_wd(G, partition, n_nodes):
    r"""returns average within-module degree"""
    wdlist = []
    for node1 in list(G.nodes):
        nbrs = G.neighbors(node1)
        mod1 = partition[node1]
        mod_nbrs = [node2 for node2 in nbrs if partition[node2] == mod1]
        wd = len(mod_nbrs)
        wdlist.append(wd)

    return sum(wdlist)/(1.*n_nodes)


def calculate_strength(g, weight_value):
    graph_freq = {}
    nodes = list(g.nodes)
    for node in nodes:
        edges = list(g.edges(node, data=True))

        freq = 0
        for edge in edges:
            (edge_a, edge_b, data) = edge
            freq += data[weight_value]

        graph_freq.update({int(node): freq})

    ave_strength_value = mean(graph_freq[k] for k in graph_freq)

    return ave_strength_value


def graph_global_measures(g, pop_name):
    """ """
    # osnovne mjere
    total_nodes = len(g.nodes())
    total_edges = len(list(g.edges()))
    deg_list = [g.degree(node) for node in list(g.nodes)]
    average_degree = np.mean(deg_list)

    ave_strength_count = calculate_strength(g, 'count')
    ave_strength_duration = calculate_strength(g, 'duration')
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
    average_cl_coeff_unweighted = mean(
        clustering_coeff[k] for k in clustering_coeff)
    clustering_coeff_w_count = nx.clustering(g, weight='count')
    average_cl_coeff_w_count = mean(
        clustering_coeff_w_count[k] for k in clustering_coeff_w_count)
    clustering_coeff_w_duration = nx.clustering(g, weight='duration')
    average_cl_coeff_w_duration = mean(
        clustering_coeff_w_duration[k] for k in clustering_coeff_w_duration)

    transitivity = nx.transitivity(g)

    degree_centrality = nx.degree_centrality(g)
    ave_deg_cent = mean(degree_centrality[k] for k in degree_centrality)

    betweenness_centrality = nx.betweenness_centrality(g)
    average_betw_cent_unweighted = mean(
        betweenness_centrality[k] for k in betweenness_centrality)
    betweenness_c_w_count = nx.betweenness_centrality(g, weight='count')
    average_betw_c_w_count = mean(
        betweenness_c_w_count[k] for k in betweenness_c_w_count)
    betweenness_c_w_duration = nx.betweenness_centrality(g, weight='duration')
    average_betw_c_w_duration = mean(
        betweenness_c_w_duration[k] for k in betweenness_c_w_duration)

    closeness_centrality_unweighted = nx.closeness_centrality(g)
    ave_closeness_cent_unw = mean(
        closeness_centrality_unweighted[k] for k in closeness_centrality_unweighted)
    closeness_c_w_count = nx.closeness_centrality(g, distance='count')
    ave_closeness_c_w_count = mean(
        closeness_c_w_count[k] for k in closeness_c_w_count)
    closeness_c_w_duration = nx.closeness_centrality(g, distance='duration')
    ave_closeness_c_w_duration = mean(
        closeness_c_w_duration[k] for k in closeness_c_w_duration)

    standard_deviation_degree = round(np.std(deg_list))
    degree_heterogeneity = standard_deviation_degree/average_degree
    degree_assortativity = nx.degree_assortativity_coefficient(g)

    # mjere za komponente
    ncc = nx.number_connected_components(g)
    # bggest component size
    gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    bcs = len(g.subgraph(gcc[0]))

    partition = community.best_partition(g)

    newman_modularity = community.modularity(partition, g, weight='count')
    modules = list(set(partition.values()))
    mod_nodes = {}
    for mod in modules:
        mod_nodes[mod] = [node for node in list(
            g.nodes) if partition[node] == mod]
    maximum_modularity = round(calculate_Qmax(g, mod_nodes), 4)
    relative_modularity = round(float(newman_modularity)/maximum_modularity, 3)

    newman_modularity_D = community.modularity(partition, g, weight='duration')
    modules = list(set(partition.values()))
    mod_nodes = {}
    for mod in modules:
        mod_nodes[mod] = [node for node in list(
            g.nodes) if partition[node] == mod]
    maximum_modularity_D = round(calculate_Qmax(g, mod_nodes), 4)
    relative_modularity_D = round(
        float(newman_modularity_D)/maximum_modularity_D, 3)

    d = {'total_nodes': total_nodes,
         'total_edges': total_edges,
         'ave_degree': average_degree,
         'ave_strength_count': ave_strength_count,
         'ave_strength_duration': ave_strength_duration,
         'edges_ave': edges_ave,
         'network_density': network_density,
         'shortest_path_len': spl,
         'diameter': diameter,
         'reach': reach,
         'global_efficiency': global_efficiency,
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

    df = pd.DataFrame(d, index=[pop_name.replace('.gml', '')])
    df = df.T

    return df


def order_columns(df):
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


def make_random_graph(number_of_nodes, number_of_edges):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(1, number_of_nodes+1)])

    while G.number_of_edges() < number_of_edges:
        node_1, node_2 = random.sample(range(1, number_of_nodes+1), 2)

        if G.has_edge(node_1, node_2):
            count = G[node_1][node_2]['count']
            old_duration = G[node_1][node_2]['count']
            new_duration = random.uniform(0.5, 600)

            if old_duration + new_duration > 600:
                continue

            else:
                G[node_1][node_2]['count'] += 1
                G[node_1][node_2]['duration'] = old_duration + new_duration

                count = G[node_1][node_2]['count']
                duration = G[node_1][node_2]['duration']

                G[node_1][node_2]['frequency'] = float((count/duration)/0.5)
                G[node_1][node_2]['averaging'] = float(
                    1/599.5)*(float(duration/count)-0.5)

                # created_edges +=1

        else:
            duration = random.uniform(0.5, 600)
            count = 1

            frequency = float((count/duration)/0.5)
            averaging = float(1/599.5)*(float(duration/count)-0.5)

            G.add_edge(node_1, node_2,
                       duration=duration,
                       count=count,
                       frequency=frequency,
                       averaging=averaging)

            # created_edges +=1

    return G


def generate_random_multigraph(g):
    edges = list(g.edges(data=True))
    weights = [weight.get('duration') for u, v, weight in edges]
    random_mg = nx.MultiGraph()

    random_mg.add_nodes_from(g.nodes())

    while random_mg.number_of_edges() < len(g.edges()):
        node_1, node_2 = random.sample(range(1, len(g.nodes())+1), 2)
        weight = weights.pop(0)
        random_mg.add_edge(node_1, node_2, duration=weight)

    return random_mg


def convert_multigraph_to_weighted(multiGraph, FPS):
    G = nx.Graph()
    G.add_nodes_from(multiGraph.nodes())

    for u, v, data in multiGraph.edges(data=True):
        if G.has_edge(u, v):
            G[u][v]['duration'] += data['duration']
            G[u][v]['count'] += 1

            duration = G[u][v]['duration']
            count = G[u][v]['count']
            frequency = float((count/duration)/0.5)
            averaging = float(1/599.5)*(float(duration/count)-0.5)

        else:
            if data['duration'] > 0:
                duration = float(data['duration']/FPS)
                count = 1
                frequency = float((count/duration)/0.5)
                averaging = float(1/599.5)*(float(duration/count)-0.5)

                G.add_edge(u, v,
                           duration=duration,
                           count=count,
                           frequency=frequency,
                           averaging=averaging)
            else:
                continue

    return G


def create_n_samples_of_random_graph(g, pop_name, n_samples, FPS):
    total = pd.DataFrame()
    for i in range(n_samples):
        df = pd.DataFrame()

        random_multi_g = generate_random_multigraph(g)
        random_weighted_g = convert_multigraph_to_weighted(random_multi_g, FPS)

        df = graph_global_measures(random_weighted_g, 'graph_' + str(i))

        total = pd.concat([total, df], axis=1)

    graph_cols = [col for col in df if col.startswith('graph')]

    df = pd.DataFrame()
    #df['median'] = total.loc[:, graph_cols].median(axis=1)
    df[pop_name.replace('.gml', '') +
       '_mean_random'] = total.loc[:, graph_cols].mean(axis=1)
    # df['std'] = total.loc[:, graph_cols].std(axis=1)

    return df


def remove_nodes_with_degree_less_than(G, degree):

    remove = [node for node, node_degree in dict(
        G.degree()).items() if node_degree < degree]
    G.remove_nodes_from(remove)

    return G


def get_x_labels(my_dict_keys):

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

    nodes = list(g.nodes)
    strength_d = {}

    for node in nodes:
        edges = list(g.edges(node, data=True))
        freq = float(0)

        for edge in edges:
            (edge_a, edge_b, data) = edge
            freq += data[weight_value]

        strength_d.update({int(node): freq})

    #ave_strength_value = mean(graph_freq[k] for k in graph_freq)

    return strength_d


def network_measures_distribution():

    graph_functions = [
        ('Degree centrality', lambda g: nx.degree_centrality(g)),
        ('Eigenvector centrality', lambda g: nx.eigenvector_centrality(g)),
        ('Closeness centrality', lambda g: nx.closeness_centrality(g)),
        ('Information centrality', lambda g: nx.information_centrality(
            max(nx.connected_component_subgraphs(g), key=len))),
        ('Page rank', lambda g: nx.pagerank(g, alpha=0.9)),
        ('Strength distribution, weight=duration',
         lambda g: get_strengtgs_dict(g, 'duration')),
        ('Strength distribution, weight=count',
         lambda g: get_strengtgs_dict(g, 'count')),
        ('Strength distribution, weighy=frequency',
         lambda g: get_strengtgs_dict(g, 'frequency')),
        ('Strength distribution, weight=average',
         lambda g: get_strengtgs_dict(g, 'averaging')),
        ('Betweenness centrality weight=None',
         lambda g: nx.betweenness_centrality(g, weight=None)),
        ('Betweenness centrality weight=duration',
         lambda g: nx.betweenness_centrality(g, weight='duration')),
        ('Betweenness centrality weight=count',
         lambda g: nx.betweenness_centrality(g, weight='count')),
        ('Betweenness centrality weight=frequency',
         lambda g: nx.betweenness_centrality(g, weight='frequency')),
        ('Betweenness centrality weight=averaging',
         lambda g: nx.betweenness_centrality(g, weight='averaging')),
        ('Clustering coefficient weight=duration',
         lambda g: nx.clustering(g, weight='duration')),
        ('Clustering coefficient weight=count',
         lambda g: nx.clustering(g, weight='count')),
        ('Clustering coefficient weight=frequency',
         lambda g: nx.clustering(g, weight='frequency')),
        ('Clustering coefficient weight=averaging',
         lambda g: nx.clustering(g, weight='averaging'))
    ]

    return graph_functions


def stat_test(d):
    """
    :param g:
    :param exp_name:
    :return:
    """

    stat_test_results = {}

    for measure_name, dict_of_values in d.items():
        ctrl = dict_of_values['CTRL']
        coc = dict_of_values['COC']

        t_statistic, p_value = scipy.stats.ttest_ind(ctrl, coc)

        stat_test_results.update({measure_name: [t_statistic, p_value]})

    df_res = pd.DataFrame.from_dict(stat_test_results, orient='index',
                                    columns=['t_statistic', 'p_value'])

    return df_res


# def plot_heatmap():
#     for pop_name, path in populations.items():
#         files = hf.load_to_df_list(path)

#         x = []
#         y = []

#         for fly_number in range(0, len(files)):
#             df1 = files[fly_number] #fly 1
#             #print ('working with ----> ', str(fly_number))

#             for i in range(0, len(files)):
#                 if i != fly_number:
#                     df2 = files[i]
#                     df = pd.concat([df1['pos_x'], df1['pos_y'] ,
#                                     df2['pos_x']-df1['pos_x'],df2['pos_y']-df1['pos_y'],
#                                     df2['ori'].round(1)], axis=1)

#                     df.columns = ['pos_x1', 'pos_y1', 'pos_x2', 'pos_y2', 'ori']
#                     df['pos_x1'] = 0
#                     df['pos_y1'] = 0

#                     df['x_axis_dif'] = (df['pos_x1'] - df['pos_x2']).abs().astype('int')
#                     df['y_axis_dif'] = (df['pos_y1'] - df['pos_y2']).abs().astype('int')

#                     df['dist'] = np.sqrt(np.square(df['x_axis_dif'])
#                                         + np.square(df['y_axis_dif'])).round(1)

#                     df = df[['pos_x2', 'pos_y2', 'ori', 'dist']]
#                     df.columns = ['pos_x', 'pos_y', 'ori', 'dist']

#                     df = df[df.dist<500]

#                     x.extend(list(df.pos_x))
#                     y.extend(list(df.pos_y))

#                 else:
#                     pass


#     # import the required packages
#     fig=plt.figure()
#     ax=fig.add_subplot(1,1,1)

#     centreCircle0 = plt.Circle((0,0),500,color="black",fill=False)
#     centreCircle11 = plt.Circle((0,0),320,color="black",fill=False)
#     centreCircle1 = plt.Circle((0,0),160,color="black",fill=False)
#     centreCircle2 = plt.Circle((0,0),80,color="black",fill=False)
#     centreCircle3 = plt.Circle((0,0),40,color="black",fill=False)
#     centreCircle4 = plt.Circle((0,0),20,color="black",fill=False)
#     centreSpot = plt.Circle((0,0),5,color="black")

#     ax.add_patch(centreCircle0)
#     ax.add_patch(centreCircle11)
#     ax.add_patch(centreCircle1)
#     ax.add_patch(centreCircle2)
#     ax.add_patch(centreCircle3)
#     ax.add_patch(centreCircle4)

#     ax.add_patch(centreSpot)

#     sns.kdeplot(x, y, shade="True", n_levels=15)
#     plt.ylim(-500, 500)
#     plt.xlim(-500, 500)

#     plt.title(pop_name)
#     plt.savefig('/home/milky/dm/results/' + str(pop_name)  + '.png')


def draw_box_plot(d, graph_title, counter):
    """[summary]

    Args:
        dictionary ([type]): [description]
        graph_title ([type]): [description]

    Returns:
        [type]: [description]
    """

    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "sans-serif"
    plt.rc('font', size=8)
    plt.rc('axes', titlesize=10)

    width_height_in = (5, 3.5)
    fig = plt.figure(figsize=width_height_in, dpi=300)

    labels, data = [*zip(*d.items())]
    labels = [label for label in labels]

    data = [list(d.values()) for d in data]
    boxplot = plt.boxplot(data, patch_artist=True, zorder=1)

    coc_count = 1
    ctrl_count = 1

    xticks_labels = []
    for label in labels:
        if label.startswith('COC'):
            xticks_labels.append('COC ' + str(coc_count))
            coc_count += 1

        else:
            xticks_labels.append('CTRL ' + str(ctrl_count))
            ctrl_count += 1

    num = list(range(1, len(xticks_labels)+1))
    plt.xticks(num, xticks_labels, rotation=90)

    ctrl_vals = []
    coc_vals = []

    for key, values in d.items():
        if key.startswith('COC'):

            coc_vals = coc_vals + list(values.values())

        else:
            ctrl_vals = ctrl_vals + list(values.values())

    average_bsl = mean(ctrl_vals)
    average_coc = mean(coc_vals)

    colors = ['pink']*11 + ['lightgreen']*9

    for item, color in zip(boxplot['boxes'], colors):
        item.set_facecolor(color)

    plt.axhline(y=average_bsl, linewidth=1, color='green',
                label='Mean CTRL', zorder=2)
    plt.axhline(y=average_coc, linewidth=1,
                color='red', label='Mean COC', zorder=2)

    plt.legend()
    plt.tight_layout()

    path = r'C:\Users\icecream\Desktop\plosclanak\pkg\pipeline\3_output\local_measures_distribution/'

    name = path + "Fig" + str(counter) + '.png'
    plt.savefig(name, dpi=400, format='png')

    name = path + "Fig" + str(counter) + '.eps'
    plt.savefig(name, dpi=400, format='eps')

    # plt.rcParams['font.sans-serif'] = "Arial"
    # plt.rcParams['font.family'] = "sans-serif"
    # plt.rc('font', size=8)
    # plt.rc('axes', titlesize=10)

    # width_height_in = (5, 3)
    # fig = plt.figure(figsize=width_height_in, dpi=450)
    # # plt.title(graph_title)
    # labels, data = [*zip(*d.items())]
    # plt.boxplot(data)

    # plt.xticks(num, labels)
    # ctrl1, 1,2, 3 itd dodati na xplot
    # mean linije pustiti
    # na y os dodati measure value
    # dodati legendu

    # range_num = list(range(1, len(labels)))
    # plt.xticks(range_num, xticks_labels, rotation=90) # rotation=90

    # path = r'C:\Users\icecream\Desktop\plosclanak\pkg\pipeline\3_output\local_measures_distribution/'
    # name = path + graph_title +'.png' #+'.eps'
    # plt.savefig(name, dpi=400, format='png')
    # return fig

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
