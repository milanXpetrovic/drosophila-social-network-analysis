# %%

import os
import sys
from itertools import product

import pandas as pd

from src.utils import fileio

input_data = '/srv/milky/drosophila-datasets/drosophila-isolation/data/processed/1_0_find_interactions'
edgelists = fileio.load_files_from_folder(os.path.join(input_data, "Cs_5DIZ"))

save_path = "./data/CsCh/"
for n, p in edgelists.items():
    df = pd.read_csv(p, index_col=0)

    df.sort_values(by="start_of_interaction")

    df = df[['node_1', 'node_2', 'start_of_interaction']]
    df.columns = ["sender", "receiver", "time"]
    df["increment"] = 1

    # senders = df['sender'].unique()
    # receivers = df['receiver'].unique()
    # all_pairs = list(product(senders, receivers))
    # for sender, receiver in all_pairs:
    #     if sender == receiver:
    #         continue
    #     selected_rows = df[(df['sender'] == sender) & (df['receiver'] == receiver)]
    #     for value, index_value in enumerate(selected_rows.index):
    #         df.at[index_value, 'increment'] = value+1

    df = df[["time", "sender", "receiver", "increment"]]
    df = df.reset_index(drop=True)
    df.to_csv(save_path + n)
    # sys.exit()
