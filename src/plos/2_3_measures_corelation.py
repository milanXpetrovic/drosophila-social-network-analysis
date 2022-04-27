# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:18:17 2021

@author: icecream
"""
import sys
sys.path.append(r"F:\1_coding_projects\my_module")

import src.import_and_preproc as toolkit
import src.my_module as hf
import src.networks_tools as netkit

import itertools
import numpy as np
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt

DATA_PATH = r'C:\Users\icecream\Desktop\plosclanak\pkg\pipeline\2_pipeline\1_0_undirected_singleedge_graph\out'
SAVE_PATH = '../3_output/'

experiments_dict = toolkit.load_files_from_folder(DATA_PATH, file_format='.gml')

## REad all graphs with nx.read_gml(path)
graphs_d = {exp_name[0:-4]: nx.read_gml(path) for exp_name, path in experiments_dict.items()} 
#get list of all graphs
graph_functions = netkit.network_measures_distribution()
    
total = {}
for foo_name, foo in graph_functions:
    # graphs_d = {exp_name: mmm.remove_nodes_with_degree_less_than(g, 4) for exp_name, g in graphs_d.items()}
    
    values = {exp_name: foo(g) for exp_name, g in graphs_d.items()}
    values = hf.group_values(values)
    total.update({foo_name: values})
    

df = pd.DataFrame()    

for measure, d in total.items():
    for key, value in d.items():
        if key.startswith('COC'):
            try:
                df[measure]=value
            
            except ValueError:
                print(measure)

plt.tight_layout()
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
f = plt.figure(figsize=(7, 5))

plt.matshow(df.corr(method='pearson'), fignum=f.number)

plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)

# plt.title('COC Correlation Matrix', fontsize=16);

  
counter = 1
  
path = r'C:\Users\icecream\Desktop\plosclanak\pkg\pipeline\3_output/'

name = path + "coc_corr_pearson" + '.png' 
plt.savefig(name, dpi=400, format='png', bbox_inches = "tight")

name = path + "coc_corr_pearson" +'.eps' 
plt.savefig(name, dpi=400, format='eps', bbox_inches = "tight")



# x = np.arange(10, 20)
# y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
# scipy.stats.pearsonr(x, y)    # Pearson's r

# correlation coefficient, p-value

# scipy.stats.spearmanr(x, y)   # Spearman's rho
# scipy.stats.kendalltau(x, y)  

# total = hf.stat_test(total)
# total = total.round(decimals=3)
# total = total.loc[:, 'median_COC':'std_CTRL']

# total.to_csv(SAVE_PATH+'local_measures_ttest.csv')
# total.to_latex(SAVE_PATH+'local_measures_ttest.tex')      



















# total = {}
# for foo_name, foo in graph_functions:
#     # graphs_d = {exp_name: mmm.remove_nodes_with_degree_less_than(g, 4) for exp_name, g in graphs_d.items()}
    
#     values = {exp_name: foo(g) for exp_name, g in graphs_d.items()}
    
#     values = hf.group_values(values)
    
#     total.update({foo_name: values})

# total = hf.stat_test(total)

# total = hf.order_columns(total)
# total = total.round(decimals=3)
# total = total.loc[:, 'median_COC':'std_CTRL']

# total.to_latex(SAVE_PATH+'global_graph_measures_no_single.tex')

# total.to_excel(SAVE_PATH+'global_graph_measures_no_single.xlsx')

#ADD
#'reciprocity', 'fragmentation'


