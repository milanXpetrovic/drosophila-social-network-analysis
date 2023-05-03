import pandas as pd

from src.utils import fileio


def find_group_mins(path):
    """Returns min x and y value from given group."""

    fly_dict = fileio.load_files_from_folder(path)

    pop_min_x, pop_min_y = [], []

    for _, path in fly_dict.items():
        df = pd.read_csv(path)
        pop_min_x.append(min(df["pos x"]))
        pop_min_y.append(min(df["pos y"]))

    return min(pop_min_x), min(pop_min_y)


def prepproc(df, min_x, min_y):
    df = df.fillna(method="ffill")

    df["pos x"] = df["pos x"].subtract(min_x)
    df["pos y"] = df["pos y"].subtract(min_y)

    ## provjera podataka ako su nan, ciscenje i popunjavanje praznih
    # if df['pos_x'].isnull().values.any() or df['pos_y'].isnull().values.any():
    #     raise TypeError("Nan value found!")
    # else:
    #     ## oduzimanje najmanje vrijednosti od svih vrijednosti u stupcu
    #     df['pos_x'] = df.pos_x.subtract(min(df['pos_x']))
    #     df['pos_y'] = df.pos_y.subtract(min(df['pos_y']))
    # df['ori'] = np.rad2deg(df['ori'])

    return df


def round_coordinates(df, decimal_places=0):
    """Round columns pos x and pos y to decimal_places, default set to 0."""

    df = df.round({"pos x": decimal_places, "pos y": decimal_places})

    return df
