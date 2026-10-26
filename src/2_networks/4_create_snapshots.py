# %%

import os
from pathlib import Path

import networkx as nx
import pandas as pd
from src.utils import fileio

from settings import main_settings


def create_graph(df_slice, all_nodes):
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    if df_slice.empty:
        return G

    edge_attrs = df_slice.groupby(['node_1', 'node_2'])['duration'].agg(
        ['count', 'sum']).rename(columns={'sum': 'total_interaction_times'})
    for (node_1, node_2), attrs in edge_attrs.iterrows():
        G.add_edge(node_1, node_2, **attrs.to_dict())
    return G


def generate_snapshots(df, all_nodes, window_size, step_size, max_val, output_dir, by_index=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    start, snapshot_idx = 0, 1
    while start < max_val:
        end = start + window_size
        df_slice = df.iloc[int(start):int(end)] if by_index else df[(
            df['start_of_interaction'] >= start) & (df['start_of_interaction'] < end)]

        if not df_slice.empty:
            G = create_graph(df_slice, all_nodes)
            nx.write_gml(G, output_dir / f"{snapshot_idx}.gml")
            snapshot_idx += 1

        start += step_size
        if step_size == 0:
            break


FPS = main_settings.VIDEO_FPS
TREATMENT = main_settings.TREATMENT

TIME_WINDOWS_SEC = [15, 30, 60, 120]
DENSITY_WINDOWS_PERCENT = [0.10, 0.25, 0.33, 0.5]
OVERLAP_PERCENTAGE = 0

INPUT_DIR = Path(main_settings.SAVE_DATA_LOCATION) / \
    "5_social_network_analysis/data/processed" / "2_0_find_interactions" / TREATMENT

BASE_OUTPUT_DIR = Path(main_settings.SAVE_DATA_LOCATION) / "5_social_network_analysis" / "snapshots"
treatment_files = fileio.load_files_from_folder(INPUT_DIR)

for group_name, group_path in treatment_files.items():
    df = pd.read_csv(group_path, index_col=0).sort_values("start_of_interaction").reset_index(drop=True)
    all_nodes = pd.unique(df[['node_1', 'node_2']].values.ravel('K'))
    overlap_str = f"overlap_{int(OVERLAP_PERCENTAGE*100)}p"

    for window_sec in TIME_WINDOWS_SEC:
        output_dir = BASE_OUTPUT_DIR / overlap_str / \
            f"time_{window_sec}s" / TREATMENT / group_name.replace('.csv', '')
        window = window_sec * FPS
        step = max(1, window * (1 - OVERLAP_PERCENTAGE))
        generate_snapshots(df, all_nodes, window, step, df['start_of_interaction'].max(), output_dir)

    for density_percent in DENSITY_WINDOWS_PERCENT:
        output_dir = BASE_OUTPUT_DIR / overlap_str /\
            f"density_{int(density_percent*100)}p" / TREATMENT / group_name.replace('.csv', '')
        window = int(len(df) * density_percent)
        step = max(1, int(window * (1 - OVERLAP_PERCENTAGE)))
        generate_snapshots(df, all_nodes, window, step, len(df), output_dir, by_index=True)
