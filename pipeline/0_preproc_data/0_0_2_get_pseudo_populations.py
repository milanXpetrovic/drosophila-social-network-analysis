#%%

# Generates pseudo populations by random sampling N flies from given treatment, each fly from different group
import os
import toml
import pandas as pd

from src import settings
from src.utils import data_utils, fileio

CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    config = toml.load(file)

INPUT_DIR = os.path.join(settings.INPUT_DIR, TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

for group_name, group_path in treatment.items():

