# %%
import os
import pandas as pd

from src import settings
from src.utils import fileio, data_utils


SCRIPT_OUTPUT = os.path.join(
    settings.OUTPUT_DIR, os.path.basename(__file__).replace(".py", "")
)

treatment = fileio.load_multiple_folders(settings.INPUT_DIR)

os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

for group_name, group_path in treatment.items():
    os.makedirs(os.path.join(SCRIPT_OUTPUT, group_name), exist_ok=True)

    fly_dict = fileio.load_files_from_folder(group_path)
    min_x, min_y = data_utils.find_group_mins(group_path)

    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path)
        df = df.head(settings.DATAFRAME_LEN)
        df = data_utils.prepproc(df, min_x, min_y)
        df = data_utils.round_coordinates(df, decimal_places=0)

        df.to_csv(os.path.join(SCRIPT_OUTPUT, group_name, fly_name))