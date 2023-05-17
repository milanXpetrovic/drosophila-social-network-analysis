# %%
import os
import time
import toml
import subprocess
from src import settings

from src.utils import fileio


os.environ["TREATMENT"] = "wt"

TREATMENT = os.environ["TREATMENT"]
TREATMENT_CONFIG = os.path.join(settings.CONFIG_DIR, "trackings", f"{TREATMENT}.toml")

with open(TREATMENT_CONFIG) as f:
    treatment_config = toml.load(f)
    ANGLE = treatment_config["ANGLE"]
    DISTANCE = treatment_config["DISTANCE"]
    TIME = treatment_config["TIME"]

print(ANGLE)

#%%
# %%
print("-" * 20)
print(f"working with: {settings.TREATMENT}")
print("-" * 20)

scripts = [
    "0_0_preproc_data.py",
    "0_1_distances_between_flies_matrix.py",
    "0_2_angles_between_flies_matrix.py",
    "1_0_undirected_singleedge_graph.py",
    "2_0_global_measures.py",
    "2_1_community_measures.py",
    "2_2_local_measures.py",
    "3_0_population_retention_heatmap.py",
    "3_1_distance_traveled.py",
]

for script in scripts:
    script_path = os.path.join(settings.PIPELINE_DIR, script)
    subprocess.run(["python", script_path])

    print(f"done with: {script}")
    time.sleep(2)

print(f"done: {settings.TREATMENT}")
print("-" * 20)
