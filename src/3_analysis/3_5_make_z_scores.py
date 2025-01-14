# %%
import os

import numpy as np
import pandas as pd

import settings
from src.utils import fileio

INPUT_PATH = "/srv/milky/drosophila-datasets/drosophila-isolation/data/results_static/global_measures_static"
SCRIPT_OUTPUT = f"{INPUT_PATH}/excel_static"
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

config = fileio.get_config(settings.CONFIG_NAME)

for TREATMENT_NAME in config["TREATMENTS"]:
    treatment = TREATMENT_NAME
    pseudo_treatment = f"pseudo_{treatment}"

    df_real = pd.read_csv(f"{INPUT_PATH}/{treatment}.csv", index_col=0)
    df_pseudo = pd.read_csv(f"{INPUT_PATH}/{pseudo_treatment}.csv", index_col=0, on_bad_lines='skip')

    os.makedirs(f"{SCRIPT_OUTPUT}/{treatment}", exist_ok=True)

    df_real.to_excel(f"{SCRIPT_OUTPUT}/{treatment}/{treatment}.xlsx")
    df_pseudo.to_excel(f"{SCRIPT_OUTPUT}/{treatment}/{pseudo_treatment}.xlsx")

    pseudo_mean = df_pseudo.mean()
    pseudo_std = df_pseudo.std()

    df_z_scores = (df_real - pseudo_mean) / pseudo_std
    df_z_scores.to_excel(f"{SCRIPT_OUTPUT}/{treatment}/z_scores_{treatment}.xlsx")
