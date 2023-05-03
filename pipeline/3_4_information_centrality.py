# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 23:35:30 2021

@author: icecream
"""

import package_functions as hf
import networkx as nx
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = '../2_pipeline/1_0_undirected_singleedge_graph/out/'
SAVE_PATH = '../3_output/'


experiments_dict = hf.load_files_from_folder(DATA_PATH, file_format='.gml')


# Information centrality
d = {}  
for exp_name, path in experiments_dict.items():         
    g = nx.read_gml(path)
    gc = max(nx.connected_component_subgraphs(g), key=len)
    information_centrality = list(nx.information_centrality(gc).values())
    d.update({exp_name[0:-4] : information_centrality})

df = hf.together_values(d)
df['data'] = df.data.round(5)

names = ['ctrl', 'coc']
colors = ['#56B4E9', '#D55E00']

x1 = list(df[df['group'] == 'CTRL']['data'])


import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Density Plot and Histogram of all arrival delays
sns.distplot(x1, hist=True, kde=True, 
             bins=int(180/12), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1})


plt.xlim([0, 0.14])
plt.ylim([0, 25])
plt.savefig('ctrl_hist.eps', dpi=350)
plt.show()

plt.cla()   # Clear axis
plt.clf()   # Clear figure

x2 = list(df[df['group'] == 'COC']['data'])

sns.distplot(x2, hist=True, kde=True, 
             bins=int(180/12), color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1})

plt.xlim([0, 0.14])
plt.ylim([0, 25])
plt.savefig('coc_hist.eps', dpi=350)
plt.hist([x1, x2], bins = 15, color = colors, label=names)

# # Plot formatting
# plt.legend()
# # plt.xlabel('Delay (min)')
# # plt.ylabel('Normalized Flights')
# # plt.title('Side-by-Side Histogram with Multiple Airlines')

# plt.savefig('ic_hist.png')
# plt.hist([x1, x2, x3, x4, x5], bins = int(180/15), normed=True,
#          color = colors, label=names)


 
# group = 'group'
# column = 'data'

# import pandas as pd
# df['data'].hist(by=df['group'], bins=20)



# names, vals, xs = [], [] ,[]

# for i, (name, subdf) in enumerate(grouped):
#     names.append(name)
#     vals.append(subdf[column].tolist())
#     xs.append(np.random.normal(i+1, 0.04, subdf.shape[0]))


# plt.boxplot(vals, labels=names)
# ngroup = len(vals)
# clevels = np.linspace(0., 1., ngroup)

# for x, val, clevel in zip(xs, vals, clevels):
#     plt.scatter(x, val, c=cm.prism(clevel), alpha=0.2)

# plt.xlabel("Populations")
# plt.ylabel("Value")

# #name = '../3_output/local_measures_distribution_total/'+graph_title+'.eps'

# plt.title(graph_title)
# plt.legend()
# plt.tight_layout()
# #plt.savefig(name, dpi=350)
# plt.show()
# plt.cla()
# plt.clf()
# plt.close()