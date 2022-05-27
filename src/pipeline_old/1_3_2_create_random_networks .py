import pandas as pd
import networkx as nx 

import package_functions as hf

DATA_PATH = '../2_pipeline/1_1_undirected_multiedge_graph/out/'
SAVE_PATH = '../3_output/'

FPS = 24

experiments = hf.load_files_from_folder(DATA_PATH, file_format='.gml')

total = pd.DataFrame()
for pop_name, path in experiments.items():
    g = nx.read_gml(path)
    
    random_multi_g = hf.generate_random_multigraph(g)

    random_weighted_g = hf.convert_multigraph_to_weighted(random_multi_g, FPS)

    res_rnd = hf.create_n_samples_of_random_graph(random_weighted_g, pop_name, 100, FPS)
    
    total = pd.concat([total, res_rnd], axis=1)
    
    print(pop_name)
    
total = hf.order_columns(total)

total.to_excel(SAVE_PATH+'graph_shuffle_measures.xlsx')


