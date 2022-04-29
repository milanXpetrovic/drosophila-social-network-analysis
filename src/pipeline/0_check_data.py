## TO START ENTER PATH OR POPULATION ID
#%%

def load_path_from_yaml()




#%%
check_if_dir_is_empty()

load_paths_to_individuals()

check_nans()
"""
Check nan valuse in specific columns if exists.
"""

check_if_enoug_rows()
"""
check if there is Fps * seconds or fps*minutes*60 rows in data files
"""

check_if_columns_missing()



main()
"""
In yaml sets value of valid data to TRUE if all above foos work
"""
RETURNS valid_data=True