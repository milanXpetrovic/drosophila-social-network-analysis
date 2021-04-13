
import sys
sys.path.append("/home/milky/my_module/module")

import random
random.seed(10)

import networks.randomize as nrz

import networkx as nx 

G = nx.Graph()
node_list = list(range(0, 50))

G.add_nodes_from(node_list)

for u in node_list:
    for v in node_list:

        if u < v and random.choice([True, False]):
            G.add_edge(u, v, weight=random.uniform(0, 1))
        
        else:
            continue


print(G.number_of_nodes())
print(G.number_of_edges())