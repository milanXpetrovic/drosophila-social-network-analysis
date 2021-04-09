import os
import re
import sys
import random

import community
import numpy as np
import pandas as pd 
import networkx as nx

from matplotlib import cm
import matplotlib.pyplot as plt

from statistics import mean, stdev
import scipy.stats
# try:
#     import modin.pandas as pd

# except ImportError:
#     import pandas as pd  

import logging 
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)



def stat_test(d):
    """
    :param g:
    :param exp_name:
    :return:
    """    
    
    stat_test_results = {}

    for measure_name, dict_of_values in d.items():
        ctrl = dict_of_values['CTRL']
        coc = dict_of_values['COC']
        
        t_statistic, p_value = scipy.stats.ttest_ind(ctrl, coc)
    
        stat_test_results.update( {measure_name: [t_statistic, p_value]})
    

    df_res = pd.DataFrame.from_dict(stat_test_results, orient='index',
                                    columns=['t_statistic', 'p_value'])
        
    return df_res

    




# def plot_heatmap():
#     for pop_name, path in populations.items():
#         files = hf.load_to_df_list(path)

#         x = []
#         y = []
            
#         for fly_number in range(0, len(files)):
#             df1 = files[fly_number] #fly 1
#             #print ('working with ----> ', str(fly_number))
        
#             for i in range(0, len(files)):
#                 if i != fly_number:
#                     df2 = files[i]
#                     df = pd.concat([df1['pos_x'], df1['pos_y'] ,
#                                     df2['pos_x']-df1['pos_x'],df2['pos_y']-df1['pos_y'],
#                                     df2['ori'].round(1)], axis=1)
                    
#                     df.columns = ['pos_x1', 'pos_y1', 'pos_x2', 'pos_y2', 'ori']
#                     df['pos_x1'] = 0
#                     df['pos_y1'] = 0
                    
#                     df['x_axis_dif'] = (df['pos_x1'] - df['pos_x2']).abs().astype('int')
#                     df['y_axis_dif'] = (df['pos_y1'] - df['pos_y2']).abs().astype('int')
                    
#                     df['dist'] = np.sqrt(np.square(df['x_axis_dif']) 
#                                         + np.square(df['y_axis_dif'])).round(1)
                    
#                     df = df[['pos_x2', 'pos_y2', 'ori', 'dist']]
#                     df.columns = ['pos_x', 'pos_y', 'ori', 'dist']
                    
#                     df = df[df.dist<500]
                    
#                     x.extend(list(df.pos_x))
#                     y.extend(list(df.pos_y))
            
#                 else:
#                     pass
                    
                
#     # import the required packages
#     fig=plt.figure()
#     ax=fig.add_subplot(1,1,1)
    
#     centreCircle0 = plt.Circle((0,0),500,color="black",fill=False)
#     centreCircle11 = plt.Circle((0,0),320,color="black",fill=False)
#     centreCircle1 = plt.Circle((0,0),160,color="black",fill=False)
#     centreCircle2 = plt.Circle((0,0),80,color="black",fill=False)
#     centreCircle3 = plt.Circle((0,0),40,color="black",fill=False)
#     centreCircle4 = plt.Circle((0,0),20,color="black",fill=False)
#     centreSpot = plt.Circle((0,0),5,color="black")
    
#     ax.add_patch(centreCircle0)
#     ax.add_patch(centreCircle11)
#     ax.add_patch(centreCircle1)
#     ax.add_patch(centreCircle2)
#     ax.add_patch(centreCircle3)
#     ax.add_patch(centreCircle4)
    
#     ax.add_patch(centreSpot)
    
#     sns.kdeplot(x, y, shade="True", n_levels=15)
#     plt.ylim(-500, 500)
#     plt.xlim(-500, 500)
    
#     plt.title(pop_name)
#     plt.savefig('/home/milky/dm/results/' + str(pop_name)  + '.png')

