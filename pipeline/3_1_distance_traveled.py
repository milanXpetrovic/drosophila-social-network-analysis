import os
import sys

import numpy as np
import pandas as pd

from src import settings
from src.utils import fileio, plotting

TREATMENT = os.environ["TREATMENT"]

INPUT_DIR = os.path.join(settings.OUTPUT_DIR, "0_0_preproc_data", TREATMENT)
trials = fileio.load_multiple_folders(INPUT_DIR)

SCRIPT_OUTPUT = os.path.join(settings.RESULTS_DIR, "distances_traveled", TREATMENT)
os.makedirs(SCRIPT_OUTPUT, exist_ok=True)

for group_name, group_path in trials.items():
    group_distances = {}
    fly_dict = fileio.load_files_from_folder(group_path)

    GROUP_OUTPUT = os.path.join(settings.RESULTS_DIR, TREATMENT, "distances_traveled", group_name)
    os.makedirs(GROUP_OUTPUT, exist_ok=True)

    res = pd.DataFrame()
    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path, usecols=["pos x", "pos y"])
        pos_x, pos_y = df["pos x"].to_numpy(), df["pos y"].to_numpy()

        distances = np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2)
        total_distance = np.sum(distances)

        fly_name = fly_name.replace(".csv", "")
        distances = pd.Series(distances, name=fly_name)
        res = pd.concat([res, distances], axis=1)

    # df = pd.DataFrame.from_dict(
    #     group_distances, orient="index", columns=["Total Distance"]
    # )
    res.to_csv(
        os.path.join(
            SCRIPT_OUTPUT,
            f"{group_name}.csv",
        )
    )

# for pop_name, path in experiments.items():
#     df = pd.read_csv(path, index_col=0)
#     total = pd.concat([total, df], axis=1)

# total.columns = columns
# df = total

# coc_columns = [col for col in df if col.startswith("COC")]
# ctrl_columns = [col for col in df if col.startswith("CTRL")]
# df_coc = df[coc_columns].T.values.tolist()
# df_ctrl = df[ctrl_columns].T.values.tolist()
# df_coc = [[x for x in y if not np.isnan(x)] for y in df_coc]
# df_ctrl = [[x for x in y if not np.isnan(x)] for y in df_ctrl]

# average_coc = mean([mean(e) for e in df_coc])
# average_ctrl = mean([mean(e) for e in df_ctrl])

# all_pop = [[x for x in y if not np.isnan(x)] for y in df.T.values.tolist()]
# plt.boxplot(all_pop)

# plt.axhline(y=average_ctrl, color="blue", label="Mean CTRL")
# plt.axhline(y=average_coc, color="red", label="Mean COC")
# plt.axvspan(0.5, len(df_coc) + 0.5, alpha=0.05, color="red", label="COC popultaions")
# plt.legend()
# plt.title("COC vs CTRL distances walked distribution")

# plt.savefig(SAVE_PATH + "distances_walked.png", dpi=350)
# plt.show()
# plt.clf()

# coc_list = [j for i in df_coc for j in i]
# ctrl_list = [j for i in df_ctrl for j in i]

# plt.boxplot(all_pop)
# plt.axvspan(1.5, 2.5, alpha=0.05, color='red', label='COC popultaions')
# plt.legend()
# # plt.savefig('ctrl_vs_coc.png', dpi=350)
# plt.show()
# plt.clf()

# average_coc = [mean(e) for e in all_pop[0:6]]
# average_ctrl = [mean(e) for e in all_pop[6:]]
# all_pop3 = [average_ctrl, average_coc]

# plt.boxplot(all_pop3)
# plt.axvspan(1.5, 2.5, alpha=0.05, color='red', label='COC popultaions')
# plt.legend()
# plt.show()
