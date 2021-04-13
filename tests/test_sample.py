from .context import sample



#! PIPELINE
#*preproc data
## load and clear functions

#*ML
## transform data, create dataset etc

#*networks
##? create grapsh
### from data, random from sample, random
##? graph analysis
### global
### mid 
### local

#*reports
## tables, plots,...

#TODO:
##! report
#!fix mid comunity stats script and add functions to module

##! random graphs
#! check and fix all functions 

#! reogranizacija pipeline i slaganje main skripte


#
# exec(open("2_2_community_stats.py").read())
# print('done')

"""
0_1_get_path_features
0_2_distances_between_flies_matrix

1_0_undirected_singleedge_graph
1_1_undirected_multiedge_graph
1_2_dynamics_of_network

1_3_1_create_random_networks
1_3_2_create_random_networks 

2_0_get_graph_measures_table
2_1_get_fly_distance_traveled
2_2_community_stats

3_1_distance_traveled_plots
3_2_population_retention_heatmap
3_3_network_measure_distribution_plot
3_4_information_centrality

4_0_statistic_tests
"""


# import sys
# sys.path.append('F:\1_coding_projects\my_module\module')
# import my_module as mmm
# from my_module import get_strengtgs_dict


# import numpy as np
# import itertools
# import networkx as nx 

# import matplotlib.pyplot as plt

# DATA_PATH = r'C:\Users\icecream\Desktop\plosclanak\pkg\pipeline\2_pipeline\1_0_undirected_singleedge_graph\out'

# experiments_dict = mmm.load_files_from_folder(DATA_PATH, file_format='.gml')

# ## REad all graphs with nx.read_gml(path)
# graphs_d = {exp_name[0:-4]: nx.read_gml(path) for exp_name, path in experiments_dict.items()} 

# #get list of all graphs
# graph_functions = mmm.network_measures_distribution()
    

# total = {}
# for foo_name, foo in graph_functions:
#     # graphs_d = {exp_name: mmm.remove_nodes_with_degree_less_than(g, 4) for exp_name, g in graphs_d.items()}
    
#     values = {exp_name: foo(g) for exp_name, g in graphs_d.items()}
    
#     values = mmm.group_values(values)
    
#     total.update({foo_name: values})
    
    
# res = mmm.stat_test(total)
      
    
# # plt.title(title)
# # print(labels)
# # range_num = [num for num in range(0, len(labels)+1)]
# # plt.xticks(range_num, labels, rotation=60)
# # plt.axvspan(5.5, 11.5, alpha=0.05, color='red', label='COC pop')
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
# # plt.cla()
# # plt.clf()
# # plt.close() 
# # name = '../3_output/local_measures_distribution_total/'+graph_title+'.eps'
