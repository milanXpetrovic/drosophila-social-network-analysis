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


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)] 
    
    return sorted(l, key = alphanum_key)


def load_multiple_folders(path):
    ## import foldera sa vise foldera unutar kojih su csv podaci 
    if not os.listdir(path):
        sys.exit('Directory is empty')
    
    experiments = {}
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for folder in d:
            experiments.update({folder : os.path.join(r, folder)})
         
    return experiments       
 

def load_files_from_folder(path, file_format='.csv'):
    ## import folder sa csvomima
    if not os.listdir(path):
        sys.exit('Directory is empty')
    
    files_dict= {}
        
    for r, d, f in os.walk(path):
        f = natural_sort(f)
        for file in f:
            if file_format in file:
                files_dict.update({file : os.path.join(r, file)})
                
    return files_dict


def load_dfs_to_list(path, min_x, min_y, file_format='.csv'):
    """Takes folder with individuals and returns list of dataframes for each
    individual.
    """
    if not os.listdir(path):
        sys.exit('Directory is empty')
        
    files_dict= {}
        
    for r, d, f in os.walk(path):
        f = natural_sort(f)
        for file in f:
            if file_format in file:
                files_dict.update({file : os.path.join(r, file)})
    
    df_list = []
    for fly_name, fly_path in files_dict.items():  
        df = pd.read_csv(fly_path, index_col=0)
        df = prepproc(df, min_x, min_y)
        df = round_coordinates(df, decimal_places=0)
        df = df[['pos_x', 'pos_y']]
        df_list.append(df)
    
    return df_list


def check_data(path):
    
    columns = ['pos x', 'pos y', 'ori', 'major axis len', 'minor axis len',
        'body area', 'fg area', 'img contrast', 'min fg dist', 'wing l x',
        'wing l y', 'wing r x', 'wing r y', 'wing l ang', 'wing l len',
        'wing r ang', 'wing r len', 'leg 1 x', 'leg 1 y', 'leg 2 x', 'leg 2 y',
        'leg 3 x', 'leg 3 y', 'leg 4 x', 'leg 4 y', 'leg 5 x', 'leg 5 y',
        'leg 6 x', 'leg 6 y', 'leg 1 ang', 'leg 2 ang', 'leg 3 ang',
        'leg 4 ang', 'leg 5 ang', 'leg 6 ang', 'vel', 'ang_vel', 'min_wing_ang',
        'max_wing_ang', 'mean_wing_length', 'axis_ratio', 'fg_body_ratio',
        'contrast', 'dist_to_wall']
    
    valid_data = False
    fly_dict = load_files_from_folder(path)
    valid_files_count = 0 
    
    for fly_name, path in fly_dict.items(): 
        df = pd.read_csv(path, index_col=0)
        df_columns = list(df.columns)
    
        if df_columns == columns:
            valid_files_count += 1
    
    if len(fly_dict) == valid_files_count:
        valid_data = True
    
    return valid_data
        

def round_coordinates(df, decimal_places=0):
    ## zaokruzivanje vrijednosti koordinata x i y na 0 decimala
    df = df.round({'pos_x': decimal_places, 'pos_y': decimal_places})
    
    return df


def prepproc(df, min_x, min_y):
    ## fill nan values 
    #df = df.where(df.notnull(), other=(df.fillna(method='ffill')+df.fillna(method='bfill'))/2)
    df = df.fillna(method='ffill')
    df.columns = df.columns.str.replace(' ', '_')
    
    df['pos_x'] = df.pos_x.subtract(min_x)
    df['pos_y'] = df.pos_y.subtract(min_y)

    ## provjera podataka ako su nan, ciscenje i popunjavanje praznih
    # if df['pos_x'].isnull().values.any() or df['pos_y'].isnull().values.any():
    #     raise TypeError("Nan value found!") 
    # else:        
    #     ## oduzimanje najmanje vrijednosti od svih vrijednosti u stupcu
    #     df['pos_x'] = df.pos_x.subtract(min(df['pos_x']))
    #     df['pos_y'] = df.pos_y.subtract(min(df['pos_y']))
    #df['ori'] = np.rad2deg(df['ori'])
    
    return df


def find_pop_mins(path):
    fly_dict = load_files_from_folder(path)
    
    pop_min_x = []
    pop_min_y = []
    for fly_name, path in fly_dict.items(): 
    
        df = pd.read_csv(path, index_col=0)
        pop_min_x.append(min(df['pos x']))
        pop_min_y.append(min(df['pos y']))
        
    return min(pop_min_x), min(pop_min_y)
    
         
def inspect_population_coordinates(path, pop_name):
    """ Draws scatter plot of x and y coordinates in population.
    This function is used to inspect validity of trackings if all coordinates 
    are inside of the arena.
    """
    x_pop_all = pd.Series()
    y_pop_all = pd.Series()
    fly_dict= load_files_from_folder(path)
    
    for fly_name, path in fly_dict.items():  
        
        df = pd.read_csv(path, usecols=['pos_x', 'pos_y'])
        df = df.dropna()
        
        x_pop_all = pd.concat([x_pop_all, df.pos_x], axis=0)
        y_pop_all = pd.concat([y_pop_all, df.pos_y], axis=0)

    x = x_pop_all
    y = y_pop_all
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.plot(x, y, 'k.', markersize=0.05)
    
    plt.title('movements of ' + pop_name)
    plt.show()


def distances_between_all_flies(files):
    """
    Input
    -----
    List of dataframes

    Returns
    -------
    Dataframe of all distances between flies.
    """

    final_df = pd.DataFrame()
    for i in range(len(files)):
    
        df1 = files[i]
    
        next_flie = i + 1
    
        if next_flie <= len(files):     
            for j in range(next_flie, len(files)):   
                df2 = files[j]
    
                df = pd.concat([df1['pos_x'], df1['pos_y'] ,
                                df2['pos_x'],df2['pos_y']], axis=1)
                
                df.columns = ['pos_x1', 'pos_y1', 'pos_x2', 'pos_y2']
                
                df['x_axis_dif'] = (df['pos_x1'] - df['pos_x2']).abs()
                df['y_axis_dif'] = (df['pos_y1'] - df['pos_y2']).abs()
                
                name = str(i+1) + ' ' + str(j+1)
                final_df[name] = np.sqrt(np.square(df['x_axis_dif']) + 
                                           np.square(df['y_axis_dif']))
            
    return final_df


def dist_flie_to_others(fly_name):
    """
    Returns fly distances compared to other flies

    Parameters
    ----------
    fly_name : TYPE
        DESCRIPTION.

    Returns
    -------
    df_fly_dist : TYPE
        DESCRIPTION.

    """
    df_fly_dist = 0

    return df_fly_dist
