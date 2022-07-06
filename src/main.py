
import my_module as mm


import os
  
os.system('python .pipeline/0_check_data.py')


### START(0)
    load_config():

### PREPROC(1)
preproc_data(raw_data):
inspec_raw_data():


### PROC(2)
get_social_interactions():
get_adjacency_matrix():
get_edge_list_with_timestamps(raw_data): 
get_path_features(raw_data): 


### TABLE_RESULTS(3)
static_graph_measures():
    global_graph_measures(G): 
    community_graph_measures(G):
    individual_graph_measures(G):


### PLOT_RESULTS_REPORT(4)
retention_heatmap(raw_data): 
distance_traveled(raw_data/results):
measures_correlation(result_table):
path_features_ethogram():
count_of_interactions_in_time():


raw_data_path
datum
vrijeme
pop
tretman
broj_jedniki
izolirani/grupirani
djevci/pareni
svjetlo
trajanje_videa
napomena 

adjacency_matrix:

dynamic_graph:

graph_measures:
    global:

    middle:

    local:

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
