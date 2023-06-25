# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils import fileio
import seaborn as sns

colors = ["blue", "green", "red", "purple"]

path = "/home/milky/drosophila-SNA/data/results/global_measures/60_sec_window"
all_treatments = fileio.load_multiple_folders(path)
measure_name = "network_density"
max_time_points = 0
for treatment_name, treatment_path in all_treatments.items():
    all_groups = fileio.load_files_from_folder(treatment_path)

    for group_name, group_path in all_groups.items():
        df = pd.read_csv(group_path, index_col=0)
        max_time_points = max(max_time_points, len(df))

for measure_name in list(df.columns):
    fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    max_density = 0

    for i, (treatment_name, treatment_path) in enumerate(all_treatments.items()):
        all_groups = fileio.load_files_from_folder(treatment_path)
        treatment_measure_values = []
        num_groups = len(all_groups)

        for group_name, group_path in all_groups.items():
            df = pd.read_csv(group_path, index_col=0)
            measure_values = df[measure_name].values
            treatment_measure_values.extend(measure_values)

        sns.kdeplot(data=treatment_measure_values, ax=axs[i], color=colors[i])
        kde = sns.kdeplot(data=treatment_measure_values, ax=axs[i], color=colors[i])

        if kde.get_lines():
            x = np.linspace(np.min(treatment_measure_values), np.max(treatment_measure_values), 300)
            y = np.interp(x, kde.get_lines()[0].get_data()[0], kde.get_lines()[0].get_data()[1])
            axs[i].fill_between(x, 0, y, alpha=0.1, color=colors[i])
            max_density = max(max_density, np.max(y))

        try:
            mean_value = np.mean(treatment_measure_values)
            axs[i].axvline(x=mean_value, color="black", linestyle="--", label="Mean")
            axs[i].text(mean_value, max_density, f"Mean: {mean_value:.2f}", color="black", ha="center", va="bottom")

            axs[i].set_xlim(np.min(treatment_measure_values), np.max(treatment_measure_values))
            axs[i].set_ylim(0, max_density + max_density * 0.3)  # Set y-axis limit dynamically
            axs[i].set_ylabel(measure_name)
            axs[i].set_title("{} ({} groups)".format(treatment_name, num_groups))

        except:
            continue

    axs[-1].set_xlabel(measure_name)
    fig.suptitle("Distribution KDE Plot of {}".format(measure_name))
    plt.tight_layout()
    plt.show()
