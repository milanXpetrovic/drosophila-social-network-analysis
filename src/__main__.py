# %%
import os
import subprocess
import sys
import time

import toml

# # flake8: noqa: E402
# current_path = os.path.abspath(os.path.dirname(__file__))
# # flake8: noqa: E402
# sys.path.append(os.path.join(os.path.abspath(os.path.join(current_path, '..', '..')), 'src'))
# # flake8: noqa: E402
from settings import main_settings
from settings import social_network_analysis as settings

scripts = [
    # "1_data/1_2_distances_angles_matrix.py",
    # "1_data/1_3_find_closest_neighbour.py",

    # "2_networks/2_0_find_interactions.py",
    # "2_networks/2_3_create_adj_matrix.py",
    # "2_networks/2_4_get_pseudo_populations.py",
    # "2_networks/2_2_create_total_graph.py",
    # "2_networks/2_5_get_shuffled_networks.py"

    # "3_analysis/3_0_1_distance_traveled.py",
    # "3_analysis/3_0_global_measures.py",
    # "3_analysis/3_2_local_measures.py",
    # "3_analysis/3_6_global_measures_shuffled.py",
    # # # # ##  "2_analysis/2_1_community_measures.py",
    # "2_analysis/2_0_population_retention_heatmap.py",
]

temporal_networks_scripts = [
    # "2_networks/2_1_create_snapshots.py",
    # "3_analysis/3_0_global_measures_snapshots.py",
    # "3_analysis/3_3_local_measures_snapshots.py",
    # # "2_analysis/2_4_count_triads_snapshots.py",
]


def main():
    TREATMENT = main_settings.TREATMENT

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


if __name__ == '__main__':
    main()
