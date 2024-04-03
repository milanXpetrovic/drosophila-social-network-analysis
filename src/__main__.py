# %%
import os
import subprocess
import sys
import time

import toml

from src import settings

scripts = [
    "1_data/0_0_0_get_csv_from_xlsx.py",
    "1_data/0_0_get_normalization.py",
    "1_data/0_1_preproc_data.py",
    "1_data/0_2_distances_angles_matrix.py",

    "2_networks/1_0_find_interactions.py",
    "2_networks/1_2_create_total_graph.py",
    "2_networks/1_4_get_pseudo_populations.py",
    "3_analysis/2_0_1_distance_traveled.py",
    "3_analysis/2_0_global_measures.py",
    "3_analysis/2_2_local_measures.py",
]

temporal_networks_scripts = [
    # "2_networks/1_1_create_snapshots.py",
    # # "2_networks/1_3_create_adj_matrix.py",
    # # "3_analysis/2_0_global_measures_snapshots.py",
    # # "3_analysis/2_3_local_measures_snapshots.py", s
    # # "2_analysis/2_4_count_triads_snapshots.py",
]


CONFIG_PATH = os.path.join(settings.CONFIG_DIR, "main.toml")
with open(CONFIG_PATH, "r") as file:
    config = toml.load(file)

for TREATMENT_NAME in config["TREATMENTS"]:
    os.environ["TREATMENT"] = TREATMENT_NAME
    TREATMENT = os.environ["TREATMENT"]
    TREATMENT_CONFIG = os.path.join(
        settings.CONFIG_DIR, "interaction_criteria", f"{TREATMENT}.toml"
    )

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
