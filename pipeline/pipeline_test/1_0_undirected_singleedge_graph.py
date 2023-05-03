# df = pd.read_csv(path, index_col=0)
# G=nx.Graph()
# node_list = list(set((" ".join(["".join(pair) for pair in list(df.columns)])).split(' ')))
# G.add_nodes_from(node_list)
# G = hf.add_edges_to_undirected_g(G, df,
#                                   DISTANCE_BETWEEN_FLIES,
#                                   TOUCH_DURATION_FRAMES,
#                                   FPS)
# name = SAVE_PATH + pop_name.replace('.csv','.gml')
# nx.write_gml(G, name)

# %%
import pandas as pd
import networkx as nx
import numpy as np
import package_functions as pf

DISTANCE_BETWEEN_FLIES = 18  # px distance, arena is 120mm wide, 1000x1000 on x,y axis
# 17px is equal to 2 body distances between flies
TOUCH_DURATION_SEC = 0.5

FPS = 24  # video fps
TOUCH_DURATION_FRAMES = int(TOUCH_DURATION_SEC*FPS)

DATA_PATH = '../data/preproc/0_2_distances_between_flies_matrix/'
SAVE_PATH = '../data/preproc/1_0_find_edges/'

INDIVIDUALS = '../data/preproc/0_0_preproc_data/'
individuals = pf.load_files_from_folder(INDIVIDUALS)

experiments = pf.load_files_from_folder(DATA_PATH)

for pop_name, path in experiments.items():
    name = SAVE_PATH + pop_name #.replace('.csv','.txt')

    with open(name, 'w') as f:  
        df = pd.read_csv(path, index_col=0)    

        for column in df.columns:
            df_ = df[column]
            df_ = df_[df_ <= DISTANCE_BETWEEN_FLIES+1]
            clear_list_of_df = [d for _, d in df_.groupby(
                df_.index - np.arange(len(df_))) if len(d) >= TOUCH_DURATION_FRAMES]

            if len(clear_list_of_df) >= 1:
                
                for row in clear_list_of_df:
                    start = row.index[0]
                    duration = len(row)
                    node_1, node_2 = column.split(' ')
                    value_to_write = node_1 + ' ' + node_2 + ' ' + str(start) + ' ' + str(duration) + '\n'
                    f.write(value_to_write)    
                    

#%%
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = '../data/preproc/1_0_find_edges/'

experiments = pf.load_files_from_folder(DATA_PATH)

for pop_name, path in experiments.items():

    label, date, time = pop_name.split('_')

    df = pd.read_csv(path, sep=' ')
    df.columns = ['node1', 'node2', 'start', 'duration']

    if label == 'CTRLall' or label == 'COCall':
        plt.title(pop_name)
        fig, ax = plt.subplots(figsize=(15, 2))

        ax.set_xlim(0, 14400)
        ax.set_ylim(bottom=0, top=1)

    for row in df.iterrows():
        _, row = row
        start = row['start']
        duration = row['duration']
        
        if label == 'CTRLall':
            ax.axvline(x=start, ymin=0, ymax=1, color="green", linewidth=duration, alpha=0.01)

        elif label == 'COCall':
            ax.axvline(x=start, ymin=0, ymax=1, color="red", linewidth=duration, alpha=0.01)

        else:
            pass

    if label == 'CTRLall' or label == 'COCall':
        plt.title(pop_name)
        plt.show()
    else:
        pass

## fly34 fly35 10757 21