import random
import pandas as pd
import networkx as nx 

import package_functions as hf

SAVE_PATH = '../3_output/'

number_of_nodes = 30
number_of_edges = random.randint(23, 232) #102

total = pd.DataFrame()
for i in range(1000):
    number_of_nodes = 30
    number_of_edges = random.randint(71, 215)
    df = pd.DataFrame()
    pop_name = 'graph_' + str(i)
    
    g = hf.make_random_graph(number_of_nodes, number_of_edges)
    
    df = hf.graph_global_measures(g, pop_name)
    
    total = pd.concat([total, df], axis=1)
    
#total = hf.order_columns(total)

graph_cols = [col for col in df if col.startswith('graph')]

df = pd.DataFrame()
df['median'] = total.loc[:, graph_cols].median(axis=1)
df['mean'] = total.loc[:, graph_cols].mean(axis=1)
df['std'] = total.loc[:, graph_cols].std(axis=1)

df.to_excel(SAVE_PATH+'graph_random_measures.xlsx')


