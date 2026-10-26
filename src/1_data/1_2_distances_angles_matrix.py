# %%
import os
from pathlib import Path

from settings import main_settings
from settings import social_network_analysis as settings
from utils import data_utils, fileio

TREATMENT = main_settings.TREATMENT

INPUT_DIR = Path(settings.INPUT_DIR) / TREATMENT
treatment = fileio.load_multiple_folders(INPUT_DIR)

DISTANCES_OUTPUT = Path(settings.OUTPUT_DIR) / "1_1_1_distances_matrix" / TREATMENT
DISTANCES_OUTPUT.mkdir(parents=True, exist_ok=True)

ANGLES_OUTPUT = Path(settings.OUTPUT_DIR) / "1_1_2_angles_matrix" / TREATMENT
ANGLES_OUTPUT.mkdir(parents=True, exist_ok=True)

for group_name, group_path in treatment.items():
    fly_dict = fileio.load_files_from_folder(group_path)

    distances = data_utils.distances_between_all_flies(fly_dict)
    distances.to_csv(os.path.join(DISTANCES_OUTPUT, f"{group_name}.csv"))

    angles = data_utils.angles_between_all_flies(fly_dict)
    angles.to_csv(os.path.join(ANGLES_OUTPUT, f"{group_name}.csv"))
    print(ANGLES_OUTPUT)
