import os
import re
import sys


def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def load_multiple_folders(path):
    # import foldera sa vise foldera unutar kojih su csv podaci
    if not os.path.exists(path) or not os.path.isdir(path):
        sys.exit(f"{path} is invalid path!")

    # Check if the directory is empty
    subfolders = [f.name for f in os.scandir(path) if f.is_dir()]

    if not subfolders:
        sys.exit("Directory is empty!")

    files = {}

    for folder in subfolders:
        folder_path = os.path.join(path, folder)
        files.update({folder: folder_path})

    return files


def load_files_from_folder(path, file_format=".csv", n_sort=False):
    # import folder sa csvomima
    if not os.listdir(path):
        sys.exit("Directory is empty")

    files_dict = {}

    for r, d, f in os.walk(path):
        if n_sort:
            f = natural_sort(f)
        for file in f:
            if file_format in file:
                files_dict.update({file: os.path.join(r, file)})

    return files_dict


def load_dfs_to_list(path, min_x, min_y, file_format=".csv"):
    """Takes folder with individuals and returns list of dataframes for each
    individual.
    """
    if not os.listdir(path):
        sys.exit("Directory is empty")

    files_dict = {}

    for r, d, f in os.walk(path):
        f = natural_sort(f)
        for file in f:
            if file_format in file:
                files_dict.update({file: os.path.join(r, file)})

    df_list = []
    for fly_name, fly_path in files_dict.items():
        df = pd.read_csv(fly_path, index_col=0)
        df = prepproc(df, min_x, min_y)
        df = round_coordinates(df, decimal_places=0)
        df = df[["pos_x", "pos_y"]]
        df_list.append(df)

    return df_list
