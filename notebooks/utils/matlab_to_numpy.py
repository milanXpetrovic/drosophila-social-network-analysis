# %%

## Util function to convert Schneider and Levine data to numpy soo I can test heatmap ploting function

import os

import pandas as pd
import scipy.io

from src.utils import fileio

PATH = ""
SAVE_PATH = ""
groups = fileio.load_files_from_folder(PATH, '.mat')

for group_name, group_path in groups.items():
    group_name = group_name.replace(".mat", "")

    if not os.path.exists(SAVE_PATH + group_name):
        os.makedirs(SAVE_PATH + group_name)

    mat = scipy.io.loadmat(group_path)
    trx = mat['trx'][0]
    for i in range(len(trx)):
        f = trx[i]

        x = f[0].tolist()[0]
        y = f[1].tolist()[0]
        theta = f[2].tolist()[0]
        a = f[3].tolist()[0]
        b = f[4].tolist()[0]

        df = pd.DataFrame({
            'pos x': x,
            'pos y': y,
            'ori': theta,
            'a':a,
            'b':b
        },
            columns=['pos x', 'pos y', 'ori', 'a', 'b'])
        df.to_csv(SAVE_PATH + group_name + "/fly" + str(i+1) + ".csv")