#%%
import os

from src import settings
from src.utils import data_utils, fileio

TREATMENT = os.environ["TREATMENT"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "0_0_preproc_data", TREATMENT)
treatment = fileio.load_multiple_folders(INPUT_DIR)

DISTANCES_OUTPUT = os.path.join(settings.OUTPUT_DIR, "0_1_1_distances_matrix", TREATMENT)
os.makedirs(DISTANCES_OUTPUT, exist_ok=True)
ANGLES_OUTPUT = os.path.join(settings.OUTPUT_DIR, "0_1_2_angles_matrix", TREATMENT)
os.makedirs(ANGLES_OUTPUT, exist_ok=True)

for group_name, group_path in treatment.items():
    fly_dict = fileio.load_files_from_folder(group_path)
    
    distances = data_utils.distances_between_all_flies(fly_dict)
    distances.to_csv(os.path.join(DISTANCES_OUTPUT, f"{group_name}.csv"))

    angles = data_utils.angles_between_all_flies(fly_dict)
    angles.to_csv(os.path.join(ANGLES_OUTPUT, f"{group_name}.csv"))
