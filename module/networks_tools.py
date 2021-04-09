import os
import re
import sys
import random

import community
import numpy as np
import pandas as pd 
import networkx as nx

from matplotlib import cm
import matplotlib.pyplot as plt

from statistics import mean, stdev
import scipy.stats
# try:
#     import modin.pandas as pd

# except ImportError:
#     import pandas as pd  

import logging 
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

def add_edges_to_undirected_g(G, df,
                              DISTANCE_BETWEEN_FLIES,
                              TOUCH_DURATION_FRAMES,
                              FPS):
    
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
	Lt= sum([G.degree(node) for node in list(G.nodes)])
	total  =0
	
	for mod in mod_nodes.keys():
		Lk = sum([G.degree(node) for node in mod_nodes[mod]])
		total+= (1.0*Lk/Lt) - (1.0*Lk/Lt)**2 

	return total


def calculate_avg_wd(G, partition, n_nodes):
	r"""returns average within-module degree"""
	wdlist = []
	for node1 in list(G.nodes):
		nbrs = G.neighbors(node1)
		mod1 = partition[node1]
		mod_nbrs = [node2 for node2 in nbrs if partition[node2]==mod1]
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
        
        graph_freq.update({int(node):freq})
    
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
    ave_strength_duration= calculate_strength(g, 'duration')
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
    
    #mjere za komponente
    ncc = nx.number_connected_components(g)
    # bggest component size
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
    
    while G.number_of_edges()<number_of_edges:
        node_1, node_2 = random.sample(range(1, number_of_nodes+1), 2)
        
        if G.has_edge(node_1, node_2):
            count = G[node_1][node_2]['count']
            old_duration = G[node_1][node_2]['count']
            new_duration = random.uniform(0.5, 600)
            
            if old_duration + new_duration > 600:
                continue

            
            else:
                G[node_1][node_2]['count'] +=1
                G[node_1][node_2]['duration'] = old_duration + new_duration
                
                count = G[node_1][node_2]['count']
                duration = G[node_1][node_2]['duration']
                
                G[node_1][node_2]['frequency'] = float((count/duration)/0.5)
                G[node_1][node_2]['averaging'] = float(1/599.5)*(float(duration/count)-0.5)
                
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
    
    while random_mg.number_of_edges()< len(g.edges()):  
        node_1, node_2 = random.sample(range(1, len(g.nodes())+1), 2)
        weight = weights.pop(0)
        random_mg.add_edge(node_1, node_2, duration=weight)
        
    return random_mg

    
def convert_multigraph_to_weighted(multiGraph, FPS):
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
    df[pop_name.replace('.gml', '') + '_mean_random'] = total.loc[:, graph_cols].mean(axis=1)
    # df['std'] = total.loc[:, graph_cols].std(axis=1)
    
    return df


def remove_nodes_with_degree_less_than(G, degree):
   
    remove = [node for node, node_degree in dict(G.degree()).items() if node_degree < degree]
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

        strength_d.update({int(node):freq})
    
    #ave_strength_value = mean(graph_freq[k] for k in graph_freq)

    return strength_d


def draw_box_plot(dictionary, graph_title):

    labels, data = [*zip(*d.items())]
    labels = [label for label in labels]
    data = [list(d.values()) for d in data]
         
    fig = plt.figure(figsize=(9,6))
    _ = plt.boxplot(data)
    
    return fig
    

def group_values(multiple_dicts):
    
    d = {}
    
    coc_values = []
    ctrl_values = []
    
    for pop_name, values in multiple_dicts.items():     
        
        for fly_label, value in values.items():
            
            if pop_name.startswith('COC'):
                
                coc_values.append(value)
        
            else:
                ctrl_values.append(value)
            
    d.update({'COC': coc_values})
    d.update({'CTRL': ctrl_values})
    
    return d


def network_measures_distribution():

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