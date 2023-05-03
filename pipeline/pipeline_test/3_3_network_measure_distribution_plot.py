import package_functions as hf


DATA_PATH = '../2_pipeline/1_0_undirected_singleedge_graph/out/'
SAVE_PATH = '../3_output/'


experiments = hf.load_files_from_folder(DATA_PATH, file_format='.gml')

hf.compare_measures(experiments)

#hf.together_measeures_coc_ctrl_distribution(experiments)

