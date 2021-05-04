import os
import re
import sys
import random

import community
import numpy as np
import pandas as pd 
import networkx as nx

import scipy.stats
from statistics import mean
import matplotlib.pyplot as plt


import logging 
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


# =============================================================================
""" TRAJECTORY DESCRIPTIVE AND FEATTURE CREATING FUNCTIONS """
# =============================================================================
def get_acc_in_path(vel_list):
    list_of_acc = []

    while len(vel_list) > 1:        
        list_of_acc.append(vel_list[1]-vel_list[0])
        vel_list.pop(0)
     
    return list_of_acc


def descriptor(list_of_values, value_name):
    ## returns dictionary of statistical values descriptors
    #list_of_values = np.array(list_of_values)
    
    min_ = min(list_of_values)
    max_ = max(list_of_values)
    mean = np.mean(list_of_values)
    median = np.median(list_of_values)
    std = np.std(list_of_values)
    p10 = np.percentile(list_of_values, 10)
    p25 = np.percentile(list_of_values, 25)
    p75 = np.percentile(list_of_values, 75)
    p90 = np.percentile(list_of_values, 90)
    
    stat_dict = {
        'min_' + value_name + '_value': min_, 
        'max_' + value_name + '_value': max_, 
        'mean_' + value_name + '_value': mean, 
        'median_' + value_name + '_value': median, 
        'std_' + value_name + '_value': std,
        'p10_' + value_name + '_value': p10,
        'p25_' + value_name + '_value': p25,
        'p75_' + value_name + '_value': p75,
        'p90_' + value_name + '_value': p90}
    
    return stat_dict    


def get_path_values(df, window_size):
    """ PATH VALUES BASED ON STEP LENGTH"""
    df['step'] = np.sqrt(np.square(df['pos_x'].diff()) +
                         np.square(df['pos_y'].diff()))
    
    df['spl'] = np.sqrt(np.square(df['pos_x'].diff(periods=window_size)) +
                        np.square(df['pos_y'].diff(periods=window_size)))
    
    df['rpl'] = df['step'].rolling(window_size).sum()
    df['str_index'] = df['spl']/df['rpl']
    df['abs_change_x'] = df['pos_x'].diff(periods=window_size).abs()
    df['abs_change_y'] = df['pos_y'].diff(periods=window_size).abs()
    
    df = df.fillna(0) 
    
    return df

    
def df_descriptor(df, value, window_size):
    
    df[value + '_min'] = df[value].rolling(window_size).min()
    df[value + '_max'] = df[value].rolling(window_size).max()
    df[value + '_mean'] = df[value].rolling(window_size).mean()
    df[value + '_median'] = df[value].rolling(window_size).median()
    df[value + '_std'] = df[value].rolling(window_size).std()
    df[value + '_p10'] = df[value].rolling(window_size).quantile(0.1)
    df[value + '_p25'] = df[value].rolling(window_size).quantile(0.25)
    df[value + '_p75'] = df[value].rolling(window_size).quantile(0.75)
    df[value + '_p90'] = df[value].rolling(window_size).quantile(0.90)
    
    return df 
    

def min_max_normalization_df(df):
    
    return (df-df.min())/(df.max()-df.min())





