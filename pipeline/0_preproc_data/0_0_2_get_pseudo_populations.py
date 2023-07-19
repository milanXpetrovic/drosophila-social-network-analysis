# %%

# Generates pseudo populations by random sampling N flies from given treatment,
# each fly from different group

import os
import toml
import pandas as pd

from src import settings
from src.utils import data_utils, fileio

TREATMENT = os.environ["TREATMENT"]

CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    config = toml.load(file)

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "0_0_preproc_data", TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

TREATMENT_CONFIG = os.path.join(settings.CONFIG_DIR, "trackings", f"{TREATMENT}.toml")
with open(TREATMENT_CONFIG) as f:
    treatment_config = toml.load(f)

    ANGLE = treatment_config["ANGLE"]
    DISTANCE = treatment_config["DISTANCE"]
    TIME = treatment_config["TIME"]
