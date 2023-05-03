import pandas as pd 
import networkx as nx

import package_functions as hf

DISTANCE_BETWEEN_FLIES = 17 #px distance, arena is 120mm wide, 1000x1000 on x,y axis
#17px is equal to 2 body distances between flies
TOUCH_DURATION_SEC = 0.5

FPS = 24 #video fps
TOUCH_DURATION_FRAMES = int(TOUCH_DURATION_SEC*FPS)

DATA_PATH = '../2_pipeline/0_2_distances_between_flies_matrix/out/'
SAVE_PATH = '../2_pipeline/1_2_dynamics_of_network/out/'

experiments = hf.load_files_from_folder(DATA_PATH)
    
for pop_name, path in experiments.items():  
    df = pd.read_csv(path, index_col=0)
    G = nx.MultiGraph()  
    
    node_list = list(set((" ".join(["".join(pair) for pair in
                                    list(df.columns)])).split(' ')))
    G.add_nodes_from(node_list)
    
    G = hf.add_multiedges_to_undirected_g(G, df,
                                          DISTANCE_BETWEEN_FLIES,
                                          TOUCH_DURATION_FRAMES)
    
    name = SAVE_PATH + pop_name.replace('.csv','.gml')
    nx.write_gml(G, name, data=True)