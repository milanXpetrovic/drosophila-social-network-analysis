# import os
# import networkx as nx
# import networkx.algorithms.community as nxcom
# from matplotlib import pyplot as plt
# import community
# import pandas as pd

# # get reproducible results
# import random
# from numpy import random as nprand
# random.seed(123)
# nprand.seed(123)

# import package_functions as hf

import sys
sys.path.append(r"F:\1_coding_projects\my_module")

import module.import_and_preproc as mt
import module.my_module as hf
import module.networks_tools as nt

DATA_PATH = '../2_pipeline/1_0_undirected_singleedge_graph/out/'
SAVE_PATH = '../3_output/'


def graphlets_detect(G):
    """
    Detect graphlets in given graph.
    """
    
    d = None    
    return d


def motifs_detect(G):
    """
    Detecti motifs in given graph.

    """
    d = None
    return d


def do_stuff(total):
    res = {}
    for index, row in total.iterrows():
        d = {}
        row = dict(row)
        
        d = {}
        coc_values = []
        ctrl_values = []
        
        for pop_name, values in row.items():
        
            if pop_name.startswith('COC'):
                coc_values.append(values)
    
            else:
                ctrl_values.append(values)
        
        d.update({'COC': coc_values})
        d.update({'CTRL': ctrl_values})
        
        res.update({index: d})
    
    return res


experiments = mt.load_files_from_folder(DATA_PATH, file_format='.gml')
    

weights = ['duration', 'count', 'frequency', 'averaging']


import pandas as pd 

total = pd.DataFrame()

for weight in weights: 
    
    total = total.append(pd.Series(name=weight))
 
    res = nt.comm_stats(experiments, weight=weight)
    
    # res = hf.order_columns(res)
    
    # total = pd.concat([total, res], axis=0)


    res = do_stuff(res)
    res = hf.stat_test(res)
    
    total = pd.concat([total, res], axis=0)
    
    
# total = hf.stat_test(res)
total = total.round(decimals=3)

# total = total.round(decimals=3)
# total = total.loc[:, 'median_COC':'std_CTRL']

total.to_csv(SAVE_PATH+'comm_louvian_ttest.csv') 
total.to_latex(SAVE_PATH+'comm_louvian_ttest.tex')
   




