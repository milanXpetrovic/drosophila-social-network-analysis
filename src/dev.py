#%%
import my_module as mm
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

import warnings
warnings.filterwarnings("ignore")
import glob
import os

PATH = '/home/milky/my_module/data'
FILE_TYPE = '*.csv'

def natural_sort(l): 
    """
    Description

    Parameters
    ----------
    variable : type
        variable description

    Returns
    -------
    variable : type
        variable description

    """

    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)] 
    
    return sorted(l, key = alphanum_key)

def load_files_from_folder(path, file_type):
    """
    """
    if not os.listdir(path):
        sys.exit('{} is empty'.format(path))
    
    files_dict= {}
        
    for root, _, files in os.walk(path):
        sorted_files = natural_sort(files)
        for file in sorted_files:
            if file_type in file:
                files_dict.update({file : os.path.join(root, file)})
            else:
                sys.exit('There is no {} files in {}'.format(file_type, path))

    return files_dict

files = load_files_from_folder(PATH, FILE_TYPE)



# PATH = 'F:/9_experiments/25-03-2022/2022-03-25_11-46/tracking/'

# individuals = mm.load_files_from_folder(PATH)


# total_x = pd.Series()
# total_y = pd.Series()

# for path in individuals.values():
    
#     df = pd.read_csv(path)
    
#     total_x = pd.concat([total_x, df["X#wcentroid (cm)"]])
#     total_y = pd.concat([total_y, df["Y#wcentroid (cm)"]])

# x = total_x.to_numpy()
# y = total_y.to_numpy()

# values = np.vstack((x, y)).T



# #%%
# from pylab import *
# import numpy as np
# from scipy.interpolate import griddata

# #create 5000 Random points distributed within the circle radius 100
# max_r = 100
# max_theta = 2.0 * np.pi
# number_points = 5000
# points = np.random.rand(number_points,2)*[max_r,max_theta]

# #Some function to generate values for these points, 
# #this could be values = np.random.rand(number_points)
# values = points[:,0] * np.sin(points[:,1])* np.cos(points[:,1])

# #now we create a grid of values, interpolated from our random sample above
# theta = np.linspace(0.0, max_theta, 100)
# r = np.linspace(0, max_r, 200)
# grid_r, grid_theta = np.meshgrid(r, theta)
# data = griddata(points, values, (grid_r, grid_theta), method='cubic',fill_value=0)

# #Create a polar projection
# ax1 = plt.subplot(projection="polar")
# ax1.pcolormesh(theta,r,data.T)
# plt.show()










































