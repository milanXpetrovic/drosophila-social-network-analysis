import itertools
from docutils import SettingsSpec

import numpy as np
import pandas as pd
import networkx as nx

from src import settings
from src.utils import fileio

import warnings

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


def find_group_mins(path):
    """Find the minimum x and y coordinates across all files in a folder.

    Parameters:
    - path (str): The path to the folder containing the files.

    Returns:
    - min_x (float): The minimum x-coordinate value across all files.
    - min_y (float): The minimum y-coordinate value across all files.
    """

    fly_dict = fileio.load_files_from_folder(path)

    pop_min_x, pop_min_y = [], []

    for _, path in fly_dict.items():
        df = pd.read_csv(path)
        pop_min_x.append(min(df["pos x"]))
        pop_min_y.append(min(df["pos y"]))

    return min(pop_min_x), min(pop_min_y)


def prepproc(df, min_x, min_y):
    """Preprocess a DataFrame by filling NaN values, subtracting minimum x and y coordinates.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing coordinates.
    - min_x (float): The minimum x-coordinate value to subtract from the "pos x" column.
    - min_y (float): The minimum y-coordinate value to subtract from the "pos y" column.

    Returns:
    - df (pd.DataFrame): The preprocessed DataFrame with filled NaN values and subtracted minimum
      x and y coordinates.
    """

    df = df.fillna(method="ffill")

    df["pos x"] = df["pos x"].subtract(min_x)
    df["pos y"] = df["pos y"].subtract(min_y)

    return df


def round_coordinates(df, decimal_places=0):
    """Round the coordinates in a DataFrame to the specified number of decimal places.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing coordinates.
    - decimal_places (int, optional): The number of decimal places to round the coordinates to.
      Default is 0.

    Returns:
    - df (pd.DataFrame): The input DataFrame with rounded coordinates.
    """

    df = df.round({"pos x": decimal_places, "pos y": decimal_places})

    return df


def distances_between_all_flies(fly_dict):
    """Calculate distances between all pairs of flies.

    Parameters:
    - fly_dict (dict): A dictionary where keys represent fly identifiers and values represent
      DataFrames containing coordinates of the flies.

    Returns:
    - df (pd.DataFrame): A DataFrame containing distances between all pairs of flies. The column
      names are formatted as "fly1_key fly2_key", and the values represent the distances between
      the corresponding flies.
    """

    loaded_fly_dict = {}
    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path, index_col=0)
        loaded_fly_dict.update({fly_name: df})

    df = pd.DataFrame()
    for fly1_key, fly2_key in list(itertools.permutations(fly_dict.keys(), 2)):
        df1 = loaded_fly_dict[fly1_key].copy(deep=True)
        df2 = loaded_fly_dict[fly2_key].copy(deep=True)

        df1_x = np.array(df1["pos x"].values)
        df1_y = np.array(df1["pos y"].values)
        df1_major_axis_len = np.array(df1["major axis len"].values)

        df2_x = np.array(df2["pos x"].values)
        df2_y = np.array(df2["pos y"].values)

        name = f"{fly1_key} {fly2_key}"

        distance = np.sqrt((df1_x - df2_x) ** 2 + (df1_y - df2_y) ** 2)
        distance = distance / np.mean(df1_major_axis_len)
        df[name] = np.round(distance, decimals=2)
        # new_df = pd.concat([df, np.sqrt((df1_x - df2_x) ** 2 + (df1_y - df2_y) ** 2)], axis=1)
        # df = new_df

    return df


def angledifference_nd(angle1, angle2):
    """Calculates the difference between two angles in degrees."""

    difference = angle2 - angle1
    adjustlow = difference < -180
    adjusthigh = difference > 180
    while any(adjustlow) or any(adjusthigh):
        difference[adjustlow] = difference[adjustlow] + 360
        difference[adjusthigh] = difference[adjusthigh] - 360
        adjustlow = difference < -180
        adjusthigh = difference > 180

    return difference


def angles_between_all_flies(fly_dict):
    """"""

    loaded_fly_dict = {}
    for fly_name, fly_path in fly_dict.items():
        df = pd.read_csv(fly_path, index_col=0)
        loaded_fly_dict.update({fly_name: df})

    df = pd.DataFrame()
    for fly1_key, fly2_key in list(itertools.permutations(fly_dict.keys(), 2)):
        df1 = loaded_fly_dict[fly1_key].copy(deep=True)
        df2 = loaded_fly_dict[fly2_key].copy(deep=True)

        df1_x = np.array(df1["pos x"].values)
        df1_y = np.array(df1["pos y"].values)
        df2_x = np.array(df2["pos x"].values)
        df2_y = np.array(df2["pos y"].values)

        df1_ori = np.array(df1["ori"].values)

        checkang = np.arctan2(df2_y - df1_y, df2_x - df1_x)
        checkang = checkang * 180 / np.pi

        angle = angledifference_nd(checkang, df1_ori * 180 / np.pi)

        name = f"{fly1_key} {fly2_key}"
        df[name] = np.round(angle, decimals=0)

        # new_df = pd.concat([df, np.sqrt((df1_x - df2_x) ** 2 + (df1_y - df2_y) ** 2)], axis=1)
        # df = new_df

    return df


def create_undirected_singleedge_graph(df_angles, df_distances, ANGLE, DISTANCE, TIME):
    node_list = list(
        set((" ".join(["".join(pair) for pair in list(df_angles.columns)])).split(" "))
    )
    node_list = [x.replace(".csv", "") for x in node_list]

    G = nx.Graph()
    G.add_nodes_from(node_list)

    for angles_col, distances_col in zip(df_angles.columns, df_distances.columns):
        if angles_col != distances_col:
            print(angles_col, distances_col)
            import sys

            sys.exit()

        df = pd.concat([df_angles[angles_col], df_distances[distances_col]], axis=1)
        df.columns = ["angle", "distance"]

        distance_mask = df["distance"] <= DISTANCE  # settings.DISTANCE[1]

        angle_mask = (df["angle"] >= ANGLE[0]) & (
            df["angle"] <= ANGLE[1]
        )
        df = df[distance_mask & angle_mask]

        min_soc_duration = TIME[0] * settings.FPS
        max_soc_duration = TIME[1] * settings.FPS

        clear_list_of_df = [
            d
            for _, d in df.groupby(df.index - np.arange(len(df)))
            if len(d) >= min_soc_duration and len(d) <= max_soc_duration
        ]

        node_1, node_2 = angles_col.split(" ")
        node_1, node_2 = node_1.replace(".csv", ""), node_2.replace(".csv", "")

        if node_1 not in G:
            G.add_node(node_1)

        if node_2 not in G:
            G.add_node(node_2)

        duration_all_interactions = sum([len(series) for series in clear_list_of_df])
        count_all_interactions = len(clear_list_of_df)
        if count_all_interactions >= 1:
            times = [len(x) for x in clear_list_of_df]
            duration = float(duration_all_interactions / settings.FPS)
            count = int(count_all_interactions)

            G.add_edge(
                node_1,
                node_2,
                times=times,
                duration=duration,
                count=count,
            )

    return G
