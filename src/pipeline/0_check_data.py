# TO START ENTER PATH OR POPULATION ID
# %%
import yaml
import my_module as mm


CONFIG = '../configs/main.yaml'


with open(CONFIG) as f:
    config = yaml.load(f, Loader=SafeLoader)


load_path_from_yaml()


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

valid_columns = mm.check_if_valid_columns(
    config['raw_data_path'], config['file_extension'], config['validation_columns'])


main()
"""
In yaml sets value of valid data to TRUE if all above foos work
"""
RETURNS valid_data = True
