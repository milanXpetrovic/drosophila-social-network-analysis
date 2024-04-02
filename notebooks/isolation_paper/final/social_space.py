#%%
import pandas as pd 
import numpy as np
import networkx as nx 
import sys, os
import matplotlib.pyplot as plt
from itertools import combinations
from src.utils import fileio

from src import settings
from src.utils import fileio

TREATMENT = "LDA_OCT_5DIZ"

ANGLES_DIR = os.path.join(settings.OUTPUT_DIR, "0_1_2_angles_matrix", TREATMENT)
DISTANCES_DIR = os.path.join(settings.OUTPUT_DIR, "0_1_1_distances_matrix", TREATMENT)

angles = fileio.load_files_from_folder(ANGLES_DIR)
distances = fileio.load_files_from_folder(DISTANCES_DIR)


res = pd.DataFrame(columns = ['angle', 'distance'])

for angles_tuple, distances_tuple in zip(angles.items(), distances.items()):
    angles_name, angles_path = angles_tuple
    distances_name, distances_path = distances_tuple

    if angles_name != distances_name:
        sys.exit()
    
    df_angles = pd.read_csv(angles_path, index_col=0)
    df_distances = pd.read_csv(distances_path, index_col=0)

    for col_name in df_angles.columns:
        main_fly = col_name.split(' ')[0]
        main_fly = main_fly.replace('.csv', '')

        ang = df_angles[col_name]
        dist = df_distances[col_name]
        
        df = pd.DataFrame({'angle': ang,'distance':dist})

        res = pd.concat([res, df], ignore_index=True)


def round_to_custom_precision(x, precision):
    return round(x / precision) * precision

res = res[res['distance'] < 20]

res['angle'] = res['angle'].apply(lambda x: round_to_custom_precision(x, 5))
res['distance'] = res['distance'].apply(lambda x: round_to_custom_precision(x, 0.25))

res['angle_distance_pair'] = res['angle'].astype(str) + '_' + res['distance'].astype(str)

unique_pairs_count = res['angle_distance_pair'].value_counts()


num_angles = 73  # 360 degrees / 5
num_distances = 81  # 20 / 0.25 + 1

# Create an empty matrix filled with zeros
matrix = np.zeros((num_distances, num_angles))

for pair, count in unique_pairs_count.items():
    angle, distance = map(float, pair.split('_'))
    angle_index = int((angle + 180) // 5)  # Convert angle to index
    distance_index = int(distance * 4)  # Convert distance to index
    matrix[distance_index, angle_index] = count

print(matrix)

degree_bins = np.linspace(-180, 180, matrix.shape[1] + 1)
distance_bins = np.linspace(0, 20, matrix.shape[0] + 1)

# Create polar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
img = ax.pcolormesh(np.radians(degree_bins), distance_bins, matrix, cmap="jet", shading='auto')

# Set radial gridlines
ax.set_rgrids(np.arange(0, 6.251, 1.0), angle=0)

# Display grid
ax.grid(True)

# Title and layout
plt.title("Histogram of Unique Pairs")
plt.tight_layout()
plt.show()