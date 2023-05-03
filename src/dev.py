# %%
import os
import my_module as mm
import pandas as pd

import yaml
from yaml.loader import SafeLoader


CONFIG = "../configs/main.yaml"


# def check_data(path):
#     """Check if there are files in the given folder and that each of them contains the default column names that are checked via the validation_columns variable."""

with open(CONFIG) as f:
    config = yaml.load(f, Loader=SafeLoader)


files_to_check = mm.load_files_from_folder(
    config["raw_data_path"], config["file_extension"]
)

needed_rows_len = config["video_fps"] * config["video_length_sec"] + 1


for file_name, file_path in files_to_check.items():
    file_to_check = pd.read_csv(file_path)

    columns_in_file = files_to_check.columns.tolist()

    if columns_in_file == config["validation_columns"]:
        valid_columns = True

    if len(file_to_check) >= config["video_fps"] * config["video_length_sec"] + 1:
        valid_rows_len = True

## KADA NEMA VRIJEDNOSTI PISE 'inf' U .CSV PODACIMA

# %%
df = df.loc[df["column_name"] == some_value]

# %%

df.replace("-", np.nan)

for column in df.columns.tolist():
    print(column)
    print(df[column].isnull().values.any())
