# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import \
    fileio  # Assuming this module has load_multiple_folders and load_files_from_folder

# --- Configuration ---

# Paths
PATH_TRACKINGS_ROOT = '/srv/milky/drosophila-datasets/drosophila-isolation/data/processed/0_0_preproc_data'
PATH_DISTANCES_ROOT = '/srv/milky/drosophila-datasets/drosophila-isolation/data/processed/0_1_1_distances_matrix'
PATH_GLOBAL_MEASURES_ROOT = "/srv/milky/drosophila-datasets/drosophila-isolation/data/results/global_measures"
OUTPUT_DIR = './' # Or specify a directory like './results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Treatments and Colors
TREATMENTS = ['CS_10D', 'CsCh', 'Cs_5DIZ', 'LDA_5DIZ', 'OCT_5DIZ', 'LDA_OCT_5DIZ']
COLORBLIND_PALETTE = sns.color_palette("colorblind", n_colors=len(TREATMENTS)) # Adjust n_colors if needed
# Ensure enough colors if more treatments than 5 are used, or map explicitly if needed
COLORS = {
    'CS_10D': COLORBLIND_PALETTE[0],
    'Cs_5DIZ': COLORBLIND_PALETTE[1],
    'LDA_OCT_5DIZ': COLORBLIND_PALETTE[2],
    'LDA_5DIZ': COLORBLIND_PALETTE[3],
    'OCT_5DIZ': COLORBLIND_PALETTE[4],
    # Add CsCh if needed:
    # 'CsCh': COLORBLIND_PALETTE[5] if len(COLORBLIND_PALETTE) > 5 else 'black',
}

# Columns and Renaming
TRACKING_COLS_TO_LOAD = ['pos x', 'pos y', 'ori', 'vel', 'ang_vel', 'min_wing_ang', 'max_wing_ang', 'mean_wing_length']
TRACKING_COLS_AGGREGATED = ['vel', 'ang_vel', 'min_wing_ang', 'max_wing_ang', 'mean_wing_length', 'dist_to_wall', 'dist_from_center', 'dist_to_closest', 'mean_dist_to_all']
NETWORK_MEASURE_COLS = [
    'Mean degree weight=count', 'Mean degree weight=duration(seconds)',
    'Degree heterogeneity (count)', 'Degree heterogeneity (total duration (seconds))',
    'Degree assortativity (count)', 'Degree assortativity (total duration (seconds))', # Fixed typo: aassortativity -> assortativity
    'Average clustering coefficient weight=count', 'Average clustering coefficient weight=duration(seconds)',
    'Average betweenness centrality weight=count', 'Average betweenness centrality weight=duration(seconds)',
    'Average closseness centrality weight=count', 'Average closseness centrality weight=duration(seconds)'
]

# Combine all columns that will eventually be in the merged dataframe
ALL_FEATURE_COLS = TRACKING_COLS_AGGREGATED + NETWORK_MEASURE_COLS

# Map original column names to nicer names for plotting/output
COLUMN_NAME_MAP = {
    'vel': 'Velocity',
    'ang_vel': 'Angular velocity',
    'min_wing_ang': 'Minimum wing angle',
    'max_wing_ang': 'Maximum wing angle',
    'mean_wing_length': 'Mean Wing length',
    'dist_to_wall': 'Distance to wall',
    'dist_from_center': 'Distance from center',
    'dist_to_closest': 'Distance to closest Fly',
    'mean_dist_to_all': 'Mean distance to all flies',
    'Mean degree weight=count': 'Mean degree (Weight: Count)',
    'Mean degree weight=duration(seconds)': 'Mean degree (Weight: Duration)',
    'Degree heterogeneity (count)': 'Degree Heterogeneity (Count)',
    'Degree heterogeneity (total duration (seconds))': 'Degree Heterogeneity (Duration)',
    'Degree assortativity (count)': 'Degree Assortativity (Count)', # Fixed typo
    'Degree assortativity (total duration (seconds))': 'Degree Assortativity (Duration)', # Fixed typo
    'Average clustering coefficient weight=count': 'Clustering Coefficient (Count)',
    'Average clustering coefficient weight=duration(seconds)': 'Clustering Coefficient (Duration)',
    'Average betweenness centrality weight=count': 'Betweenness Centrality (Count)',
    'Average betweenness centrality weight=duration(seconds)': 'Betweenness Centrality (Duration)',
    'Average closseness centrality weight=count': 'Closeness Centrality (Count)',
    'Average closseness centrality weight=duration(seconds)': 'Closeness Centrality (Duration)'
}

# Define the desired order using the *mapped* names
DESIRED_PLOT_ORDER = [COLUMN_NAME_MAP[col] for col in ALL_FEATURE_COLS if col in COLUMN_NAME_MAP]
# Alternative, if you want to explicitly define the order of mapped names:
# DESIRED_PLOT_ORDER = [
#     'Velocity', 'Angular velocity', 'Minimum wing angle', 'Maximum wing angle',
#     'Mean Wing length', 'Distance to wall', 'Distance from center',
#     'Distance to closest Fly', 'Mean distance to all flies',
#     'Mean degree (Weight: Count)', 'Mean degree (Weight: Duration)',
#     'Degree Heterogeneity (Count)', 'Degree Heterogeneity (Duration)',
#     'Degree Assortativity (Count)', 'Degree Assortativity (Duration)',
#     'Clustering Coefficient (Count)', 'Clustering Coefficient (Duration)',
#     'Betweenness Centrality (Count)', 'Betweenness Centrality (Duration)',
#     'Closeness Centrality (Count)', 'Closeness Centrality (Duration)'
# ]


# Arena Parameters
ARENA_RADIUS = 30.5
ARENA_CENTER = (ARENA_RADIUS, ARENA_RADIUS) # Assuming square arena centered in coordinate system


# --- Helper Functions ---

def process_fly_data(fly_path, fly_name, group_name, treatment_name, distances_df):
    """Loads and processes data for a single fly."""
    df = pd.read_csv(fly_path, usecols=TRACKING_COLS_TO_LOAD)

    pos_x = df['pos x'].to_numpy()
    pos_y = df['pos y'].to_numpy()

    # Calculate distances
    dist_sq_from_center = (pos_x - ARENA_CENTER[0])**2 + (pos_y - ARENA_CENTER[1])**2
    df['dist_from_center'] = np.sqrt(dist_sq_from_center)
    # Approximation: dist_to_wall = radius - dist_from_center (more accurate if needed)
    df['dist_to_wall'] = ARENA_RADIUS - df['dist_from_center'] # Original calculation was slightly different but likely equivalent if center=(radius,radius)

    # Get fly-specific distances from the preloaded group distance matrix
    fly_dist_cols = distances_df.columns[distances_df.columns.str.startswith(fly_name)]
    if not fly_dist_cols.empty:
        fly_distances = distances_df[fly_dist_cols]
        df['dist_to_closest'] = fly_distances.min(axis=1)
        df['mean_dist_to_all'] = fly_distances.mean(axis=1)
    else:
        # Handle case where fly might not be in distance matrix (e.g., single fly group?)
        df['dist_to_closest'] = np.nan
        df['mean_dist_to_all'] = np.nan


    # Calculate mean values for the selected columns
    fly_summary = df[TRACKING_COLS_AGGREGATED].mean().to_dict() # Use TRACKING_COLS_AGGREGATED

    # Add metadata
    fly_summary['Fly'] = fly_name.replace('.csv', '')
    fly_summary['Group'] = group_name
    fly_summary['Treatment'] = treatment_name

    return fly_summary

def load_and_process_tracking_data(treatments, trackings_root, distances_root):
    """Loads tracking and distance data for all treatments and processes it."""
    all_fly_summaries = []
    # Assumes fileio.load_multiple_folders returns {treatment_name: path_to_treatment_folder}
    treatment_tracking_paths = fileio.load_multiple_folders(trackings_root)

    for t in treatments:
        if t not in treatment_tracking_paths:
            print(f"Warning: Tracking data not found for treatment {t}")
            continue

        # Assumes fileio.load_multiple_folders returns {group_name: path_to_group_folder}
        group_tracking_paths = fileio.load_multiple_folders(treatment_tracking_paths[t])
        # Assumes fileio.load_files_from_folder returns {filename: filepath}
        treatment_distances_files = fileio.load_files_from_folder(os.path.join(distances_root, t), '.csv')

        for group_name, group_path in group_tracking_paths.items():
            distance_file_key = f'{group_name}.csv'
            if distance_file_key not in treatment_distances_files:
                print(f"Warning: Distances file not found for group {group_name} in treatment {t}")
                continue
            distances_df = pd.read_csv(treatment_distances_files[distance_file_key], index_col=0)

            fly_files = fileio.load_files_from_folder(group_path, '.csv')
            for fly_name, fly_path in fly_files.items():
                fly_summary = process_fly_data(fly_path, fly_name, group_name, t, distances_df)
                all_fly_summaries.append(fly_summary)

    if not all_fly_summaries:
        return pd.DataFrame() # Return empty DataFrame if no data loaded

    # Create DataFrame once from the list of dictionaries
    df = pd.DataFrame(all_fly_summaries)
    df.set_index(['Treatment', 'Group', 'Fly'], inplace=True)
    return df


def load_network_measures(treatments, measures_root, columns_to_load):
    """Loads raw network measures and calculates z-scores relative to pseudo controls."""
    raw_measures_list = []
    z_scored_measures_list = []

    # Assumes fileio.load_files_from_folder returns {filename: filepath}
    measure_files = fileio.load_files_from_folder(measures_root, '.csv')

    for t in treatments:
        measure_file = f"{t}.csv"
        pseudo_file = f"pseudo_{t}.csv"

        if measure_file not in measure_files:
            print(f"Warning: Measure file not found: {measure_file}")
            continue
        if pseudo_file not in measure_files:
            print(f"Warning: Pseudo measure file not found: {pseudo_file}")
            continue

        # Load main measures
        df = pd.read_csv(measure_files[measure_file], index_col=0)
        df = df[columns_to_load] # Select only the desired network columns
        df['Treatment'] = t
        raw_measures_list.append(df)

        # Load pseudo measures for z-scoring
        df_pseudo = pd.read_csv(measure_files[pseudo_file], index_col=0)
        df_pseudo = df_pseudo[columns_to_load]

        # Calculate Z-scores relative to pseudo controls
        pseudo_mean = df_pseudo.mean()
        pseudo_std = df_pseudo.std()
        # Avoid division by zero or NaN std dev
        pseudo_std = pseudo_std.replace(0, np.nan) # Or handle differently if needed
        df_z_scores = (df[columns_to_load] - pseudo_mean) / pseudo_std
        df_z_scores['Treatment'] = t
        z_scored_measures_list.append(df_z_scores)

    # Concatenate results
    raw_measures_df = pd.concat(raw_measures_list)
    raw_measures_df.index.name = 'Group' # Assuming index is Group name
    raw_measures_df.set_index('Treatment', append=True, inplace=True)
    raw_measures_df = raw_measures_df.reorder_levels(['Treatment', 'Group'])

    z_scored_measures_df = pd.concat(z_scored_measures_list)
    z_scored_measures_df.index.name = 'Group'
    z_scored_measures_df.set_index('Treatment', append=True, inplace=True)
    z_scored_measures_df = z_scored_measures_df.reorder_levels(['Treatment', 'Group'])


    return raw_measures_df, z_scored_measures_df


def calculate_aggregated_stats(df, level='Treatment'):
    """Calculates mean and std grouped by the specified level."""
    return df.groupby(level=level).agg(['mean', 'std'])


def calculate_plot_z_scores(df, selected_treatments, columns):
    """Calculates z-scores for selected treatments relative to their combined mean/std."""
    # Filter for the treatments to be plotted
    filtered_df = df.loc[df.index.get_level_values('Treatment').isin(selected_treatments), columns]

    if filtered_df.empty:
        print("Warning: No data found for selected treatments in calculate_plot_z_scores.")
        return pd.DataFrame()

    # Calculate Z-scores based on the mean/std *within the selected subset*
    subset_mean = filtered_df.mean()
    subset_std = filtered_df.std().replace(0, np.nan) # Avoid division by zero

    z_scores = (filtered_df - subset_mean) / subset_std

    # Aggregate Z-scores by treatment
    z_scores_agg = z_scores.groupby(level='Treatment').agg(['mean', 'sem']) # Use sem for error bars
    z_scores_agg = z_scores_agg.reindex(index=selected_treatments) # Ensure order

    return z_scores_agg


def plot_means_with_error_bars(z_scores_agg, plot_order, colors, error_type='sem', xlim=(-2, 2), title="Feature Z-Scores Comparison"):
    """Plots means with error bars for selected treatments."""
    if z_scores_agg.empty:
        print("Skipping plot: No data to plot.")
        return

    # Ensure z_scores_agg columns (measures) are ordered correctly
    # The aggregation gives multiindex columns ('Measure', 'mean'/'sem')
    # We need to extract data per treatment
    treatments = z_scores_agg.index.unique().tolist()
    
    # Select only measures present in both plot_order and the dataframe columns
    measures_in_data = z_scores_agg.columns.get_level_values(0).unique()
    ordered_measures_to_plot = [m for m in plot_order if m in measures_in_data]
    
    if not ordered_measures_to_plot:
        print("Warning: None of the desired plot order measures found in the data.")
        return

    fig, ax = plt.subplots(figsize=(6, max(5, len(ordered_measures_to_plot) * 0.4))) # Adjust height based on number of measures

    y_positions = np.arange(len(ordered_measures_to_plot))
    
    # Add small vertical offset for each treatment to avoid overlap
    num_treatments = len(treatments)
    offset_scale = 0.15 if num_treatments > 1 else 0 # Adjust as needed
    offsets = np.linspace(-offset_scale * (num_treatments - 1) / 2, offset_scale * (num_treatments - 1) / 2, num_treatments)

    for i, treatment in enumerate(treatments):
        means = []
        errors = []
        valid_measures_labels = []
        for measure_label in ordered_measures_to_plot:
             # Check if the measure exists for this treatment before accessing
            if measure_label in z_scores_agg.columns.get_level_values(0):
                means.append(z_scores_agg.loc[treatment, (measure_label, 'mean')])
                errors.append(z_scores_agg.loc[treatment, (measure_label, error_type)])
                valid_measures_labels.append(measure_label)
            else:
                 print(f"Warning: Measure '{measure_label}' not found for treatment '{treatment}'.")

        # Get y-positions corresponding to the valid measures found
        valid_y_positions = [y_positions[ordered_measures_to_plot.index(m)] for m in valid_measures_labels]

        ax.errorbar(means, np.array(valid_y_positions) + offsets[i], xerr=errors, fmt='o', color=colors.get(treatment, 'black'),
                    ecolor=colors.get(treatment, 'black'), elinewidth=1.5, capsize=3, label=treatment if i == 0 else "_nolegend_") # Only label first point set per treatment
        # Add scatter points on top for better visibility
        ax.scatter(means, np.array(valid_y_positions) + offsets[i], color=colors.get(treatment, 'black'), s=50, label=treatment)


    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Z score')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_measures_to_plot)
    ax.invert_yaxis() # Often preferred for this type of plot
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5)) # Adjust legend position
    ax.grid(axis='x', linestyle='--', color='gray', alpha=0.5)
    plt.xlim(xlim)
    plt.title(title)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    # Save the plot instead of just showing it
    plot_filename = f"zscore_comparison_{'_vs_'.join(treatments)}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, plot_filename), bbox_inches='tight')
    print(f"Saved plot to {os.path.join(OUTPUT_DIR, plot_filename)}")
    # plt.show() # Optionally show plot after saving

# --- Main Execution ---

print("1. Processing Tracking Data...")
fly_tracking_summary = load_and_process_tracking_data(TREATMENTS, PATH_TRACKINGS_ROOT, PATH_DISTANCES_ROOT)
if fly_tracking_summary.empty:
    print("Error: No tracking data loaded. Exiting.")
    exit()

print("2. Calculating Aggregated Tracking Stats...")
tracking_stats_agg = calculate_aggregated_stats(fly_tracking_summary[TRACKING_COLS_AGGREGATED])
tracking_stats_agg.to_excel(os.path.join(OUTPUT_DIR, 'tracking_stats_aggregated.xlsx'))
print(f"Saved aggregated tracking stats to {os.path.join(OUTPUT_DIR, 'tracking_stats_aggregated.xlsx')}")

print("3. Loading Network Measures...")
# Load raw measures and also calculate z-scores relative to pseudo controls
raw_network_measures, network_measures_pseudo_zscores = load_network_measures(
    TREATMENTS, PATH_GLOBAL_MEASURES_ROOT, NETWORK_MEASURE_COLS
)
if raw_network_measures.empty:
    print("Warning: No network measures loaded.")
    # Decide if you want to continue without network measures or exit
    # exit()

print("4. Calculating Aggregated Network Stats...")
if not raw_network_measures.empty:
    network_stats_agg = calculate_aggregated_stats(raw_network_measures)
    network_stats_agg.to_excel(os.path.join(OUTPUT_DIR, 'network_stats_aggregated.xlsx'))
    print(f"Saved aggregated network stats to {os.path.join(OUTPUT_DIR, 'network_stats_aggregated.xlsx')}")

    # Save pseudo z-scored stats as well
    network_pseudo_zscores_agg = calculate_aggregated_stats(network_measures_pseudo_zscores)
    network_pseudo_zscores_agg.to_excel(os.path.join(OUTPUT_DIR, 'network_stats_pseudo_zscores_aggregated.xlsx'))
    print(f"Saved aggregated pseudo z-scored network stats to {os.path.join(OUTPUT_DIR, 'network_stats_pseudo_zscores_aggregated.xlsx')}")


print("5. Merging Tracking and Network Data (per group)...")
# Calculate mean tracking summary per group *before* joining
fly_tracking_summary_grouped = fly_tracking_summary.groupby(level=['Treatment', 'Group']).mean()

# Join tracking summary (grouped) with raw network measures (ungrouped)
# Use 'outer' join if some groups might miss one type of data, 'inner' if both must exist
merged_df = fly_tracking_summary_grouped.join(raw_network_measures, how='inner') # Use 'inner' join

# Rename columns for plotting
merged_df.rename(columns=COLUMN_NAME_MAP, inplace=True)

# Ensure desired columns exist after merge and rename
final_columns_present = [col for col in DESIRED_PLOT_ORDER if col in merged_df.columns]


print("6. Calculating Z-Scores for Plotting...")
# --- Plotting Example: CS_10D vs Cs_5DIZ ---
selected_treatments_plot1 = ['CS_10D', 'Cs_5DIZ']
plot_z_scores_1 = calculate_plot_z_scores(merged_df, selected_treatments_plot1, final_columns_present)


# %%
print("7. Generating Plot...")
plot_means_with_error_bars(
    plot_z_scores_1,
    plot_order=final_columns_present, # Use columns actually present
    colors=COLORS,
    error_type='sem', # Standard error of the mean
    xlim=[-2.5, 2.5], # Adjusted xlim slightly
    title=f"Comparison: {selected_treatments_plot1[0]} vs {selected_treatments_plot1[1]}"
)

# --- Plotting Example 2: Compare 3 treatments ---
# selected_treatments_plot2 = ['LDA_5DIZ', 'OCT_5DIZ', 'LDA_OCT_5DIZ']
# plot_z_scores_2 = calculate_plot_z_scores(merged_df, selected_treatments_plot2, final_columns_present)
# plot_means_with_error_bars(
#     plot_z_scores_2,
#     plot_order=final_columns_present,
#     colors=COLORS,
#     error_type='sem',
#     xlim=[-2.5, 2.5],
#     title=f"Comparison: LDA vs OCT vs LDA+OCT (5DIZ)"
# )


print("Processing complete.")