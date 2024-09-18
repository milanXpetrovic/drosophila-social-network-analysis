#%%
import sys
from itertools import product

import pandas as pd

from src.utils import fileio

path = "./data/CS_10D_csv"

edgelists = fileio.load_files_from_folder(path)


for n, p in edgelists.items():
    df = pd.read_csv(p, usecols=["sender", "receiver", "time"])
    df.sort_values(by="time")
    # df.columns = ["sender", "receiver", "time"]

    senders = df['sender'].unique()
    receivers = df['receiver'].unique()
    all_pairs = list(product(senders, receivers))

    df["increment"] = 1

    # for sender, receiver in all_pairs:
    #     if sender == receiver:
    #         continue
        
    #     selected_rows = df[(df['sender'] == sender) & (df['receiver'] == receiver)]

    #     for value, index_value in enumerate(selected_rows.index):
    #         df.at[index_value, 'increment'] = value+1


    df.to_csv(path+"/"+n)
    

# %%
