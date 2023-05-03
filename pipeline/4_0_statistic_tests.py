import package_functions as hf
import scipy
from scipy import stats
import pandas as pd
DATA_PATH = '../2_pipeline/1_0_undirected_singleedge_graph/out/'
SAVE_PATH = '../3_output/'


experiments = hf.load_files_from_folder(DATA_PATH, file_format='.gml')

# t_statistic, p_value = scipy.stats.ttest_ind(a, b)
# hf.together_measeures_coc_ctrl_distribution(experiments)

df = hf.stat_test(experiments)

# df = df.round(4)

# df.to_excel(SAVE_PATH + 'stat_tests.xlsx')