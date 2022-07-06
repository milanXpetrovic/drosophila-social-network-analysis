# My module: Toolkit for trajectory data processing,network construction and analysis

[![License](https://img.shields.io/badge/license-BSD--3%20Clause-green)](https://github.com/milanXpetrovic/my_module/blob/main/LICENSE.md)


## TO DO:

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


"""
Python module
Purpose of this Python module is to create toolkit for analysing data from
complex systems. Analysis is from the data collected by monitoring the participants
in that system, i.e. individuals. 

The project is open-source and aims to create research tools.
If you are interested in development, feel free to contact us.

Currently the module is being tested on data created by monitoring the biological
system (drosophila melanogaster populations) and social interaction networks are being created.

LIST OF SUB-MODULES:

- Toolkit: Functions for data manipulation, organizing and reading the
contents of a large number of folders and files within them or the files themselves.
Another option is to check the validity of the data, or the missing records, 
then the possibility of cleaning or fixing them.

- Ml: functionalities for data transformation (eg extracting statistical values,
smoothing data, etc.) and creating forms that for process and transform data 
for implementation of machine learning.

- Networks: Analysis of individuals within a biologically complex systems.
Interpretation of data in the form of complex networks and the presentation
of data through graphs will be enabled. And the second part of the sub-module
contains functions for analysis and calculations over the created graphs.
In the last part of this module, there would be machine learning functions on graphs.

"""

*Components of this module are:*
- Data loading, validity checking and preprocessing functions
- Path descriptors and ML features creating functions
- Population analysis, distances between elements from trajectory
- Complex network construction
- Complex network analysis
