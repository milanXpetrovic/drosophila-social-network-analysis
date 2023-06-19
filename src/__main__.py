# %%
import os
import time
import sys
import subprocess

from src import settings
from src.utils import fileio

scripts = [
    # "0_0_preproc_data.py",
    # "0_1_distances_between_flies_matrix.py",
    # "0_2_angles_between_flies_matrix.py",
    # "1_0_find_interactions.py",
    "1_1_create_snapshots.py",
    "2_0_global_measures.py",
    "2_1_community_measures.py",
    "2_2_local_measures.py",
    "3_0_population_retention_heatmap.py",
    "3_1_distance_traveled.py",
]

START = 0
END = 30

os.environ["START_TIME"] = str(START)
os.environ["END_TIME"] = str(END)

TIME_WINDOW = 120  # seconds
os.environ["TIME_WINDOW"] = str(TIME_WINDOW)

treatments = ["CsCh"]  # , "ELAV"

for treatment_value in treatments:
    os.environ["TREATMENT"] = treatment_value

    TREATMENT = os.environ["TREATMENT"]
    TREATMENT_CONFIG = os.path.join(
        settings.CONFIG_DIR, "trackings", f"{TREATMENT}.toml"
    )

    print("-" * 15, f" START: {TREATMENT} ", "-" * 15)
    for script in scripts:
        try:
            print(f"STARTING with: {script}")
            script_path = os.path.join(settings.PIPELINE_DIR, script)
            subprocess.run(["python", script_path])

            print(f"DONE with: {script}")
            time.sleep(1)
        except:
            sys.exit()

    print("-" * 15, f" DONE: {TREATMENT} ", "-" * 15)
