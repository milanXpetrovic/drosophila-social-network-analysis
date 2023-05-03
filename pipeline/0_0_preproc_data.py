import os
import pandas as pd
import package_functions as hf

EXPERIMENT_DURATION = 600  # experiment duration time must be in seconds
FPS = 24
DATAFRAME_LEN = EXPERIMENT_DURATION * FPS

DATA_PATH = r"F:/0_fax/DM_dataset/raw_trackings_pop"
SAVE_PATH = "../2_pipeline/0_0_preproc_data/out"

experiments = hf.load_multiple_folders(DATA_PATH)

for pop_name, path in experiments.items():
    if not hf.check_data(path):
        continue

    if not os.path.exists(SAVE_PATH + "/" + pop_name):
        os.mkdir(SAVE_PATH + "/" + pop_name)

    fly_dict = hf.load_files_from_folder(path)

    # hf.inspect_population_coordinates(path, pop_name)

    min_x, min_y = hf.find_pop_mins(path)

    for fly_name, path in fly_dict.items():
        df = pd.read_csv(path, index_col=0)
        df = df.head(DATAFRAME_LEN)
        df = hf.prepproc(df, min_x, min_y)
        df = hf.round_coordinates(df, decimal_places=0)

        df.to_csv(SAVE_PATH + "/" + pop_name + "/" + fly_name)
