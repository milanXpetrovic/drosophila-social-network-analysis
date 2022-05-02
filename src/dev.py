# %%
import os
import my_module as mm
import pandas as pd

import yaml
from yaml.loader import SafeLoader


CONFIG = '../configs/main.yaml'


# def check_data(path):
#     """Check if there are files in the given folder and that each of them contains the default column names that are checked via the validation_columns variable."""

with open(CONFIG) as f:
    config = yaml.load(f, Loader=SafeLoader)
    raw_data_path = config['raw_data_path']
    file_extension = config['file_extension']
    validation_columns = config['validation_columns']


def check_if_valid_columns(raw_data_path, file_extension, validation_columns):
    """Returns True if all files from folder contain given validation columns."""

    valid_data = False
    valid_files_count = 0
    files_to_check = mm.load_files_from_folder(raw_data_path, file_extension)

    for file_name, file_path in files_to_check.items():
        columns_in_file = pd.read_csv(file_path, nrows=1).columns.tolist()

        if columns_in_file == validation_columns:
            valid_files_count += 1

    if len(files_to_check) == valid_files_count:
        valid_data = True

    return valid_data
