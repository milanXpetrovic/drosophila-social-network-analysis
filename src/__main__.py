# %%
import os
import subprocess
import sys
import toml
import time

from src import settings

scripts = [
    # "0_0_0_get_normalization.py",
    # "0_0_1_preproc_data.py",
    # "0_1_distances_between_flies_matrix.py",
    # "0_2_angles_between_flies_matrix.py",
    # "1_0_find_interactions.py",
    # "1_1_create_snapshots.py",
    # "1_2_create_total_graph.py",
    # "2_0_1_distance_traveled.py",
    # "2_0_global_measures.py",
    # "2_1_community_measures.py",
    # "2_2_local_measures.py",
    # "2_3_local_measures_snapshots.py",
    # "3_0_population_retention_heatmap.py",
]

config_file_path = "/home/milky/drosophila-SNA/configs/main.toml"
with open(config_file_path, "r") as file:
    config = toml.load(file)

for TREATMENT_NAME in config["TREATMENTS"]:
    os.environ["TREATMENT"] = TREATMENT_NAME
    TREATMENT = os.environ["TREATMENT"]
    TREATMENT_CONFIG = os.path.join(settings.CONFIG_DIR, "trackings", f"{TREATMENT}.toml")

    print("-" * 15, f" START: {TREATMENT} ", "-" * 25)
    for script in scripts:
        try:
            print(f"STARTING with: {script}")
            script_path = os.path.join(settings.PIPELINE_DIR, script)
            subprocess.run(["python", script_path])
            print(f"DONE with: {script}")
            time.sleep(1)
        except BaseException:
            sys.exit()

    print("-" * 15, f" DONE: {TREATMENT} ", "-" * 25)
