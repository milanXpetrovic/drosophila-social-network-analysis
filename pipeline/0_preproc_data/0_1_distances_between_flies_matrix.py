# %%
import os
import sys

from src import settings
from src.utils import data_utils, fileio

TREATMENT = os.environ["TREATMENT"]

SCRIPT_OUTPUT = os.path.join(settings.OUTPUT_DIR, "0_1_distances_between_flies_matrix", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "0_0_preproc_data", TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

for group_name, group_path in treatment.items():
    fly_dict = fileio.load_files_from_folder(group_path)
    distances = data_utils.distances_between_all_flies(fly_dict)
    distances.to_csv(os.path.join(SCRIPT_OUTPUT, f"{group_name}.csv"))
