# %%
import scipy.io
import pandas as pd
import numpy as np
import re
import json
import sys, os

TREATMENT = "CTRL_5DIZ"
# import fileio

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
        for file in f:
            if file_format in file:
                files_dict.update({file: os.path.join(r, file)})

    return files_dict


path = f"./data/trackings/{TREATMENT}"
all_groups = load_multiple_folders(path)

# normalization_path = "/home/milky/soc/data/input/pxpermm/CsCh.json"
# with open(normalization_path, "r") as json_file:
#     pxpermm_group = json.load(json_file)

ix = 1 
for group_name, group_path in all_groups.items():
    csv_dict = load_files_from_folder(group_path)
    structured_arrays = []
    for fly_name, fly_path in csv_dict.items():
        df = pd.read_csv(fly_path)
        df = df[["pos x", "pos y", "ori", "major axis len", "minor axis len"]]
        df = df.interpolate()

        FPS = 24
        FLY_ID = re.findall(r'\d+', fly_name)[0]
        x = np.array([df["pos x"].to_numpy()])
        y = np.array([df["pos y"].to_numpy()])
        theta = np.array([df["ori"].to_numpy()])
        a = np.array([(df["major axis len"]/4).to_numpy()])
        b = np.array([df["minor axis len"].to_numpy()])
        id = np.array([[np.uint8(FLY_ID)]])
        pxpermm = np.array([[np.float64(30.0)]])
        fps = np.array([[np.float64(FPS)]])
        structured_array = np.array([(x, y, theta, a, b, id, pxpermm, fps)],
                                    dtype=[('x', object), ('y', object), ('theta', object), ('a', object), ('b', object), ('id', object), ('pxpermm', object), ('fps', object)])

        structured_arrays.append(structured_array)

    trx_csv = np.array(structured_arrays)
    trx_csv = trx_csv.T

    mat_csv = {
        '__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Thu Jul 06 10:49:32 2023',
        '__version__': '1.0',
        '__globals__': [],
        'trx': trx_csv
    }

    scipy.io.savemat(f"./test/{TREATMENT}_movie_{ix}.mat", mat_csv)
    
    ix+=1
