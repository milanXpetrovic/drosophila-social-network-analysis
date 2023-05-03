import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

import package_functions as hf

import warnings
warnings.filterwarnings("ignore")

def myplot(x, y, s, bins=500):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


DATA_PATH = '../2_pipeline/0_0_preproc_data/out'
# DATA_PATH = r'F:/0_fax/DM_dataset/raw_trackings_pop'
SAVE_PATH = ''

experiments = hf.load_multiple_folders(DATA_PATH)
        
x_all_ctrl = pd.Series() 
y_all_ctrl = pd.Series() 
x_all_coc = pd.Series() 
y_all_coc = pd.Series() 
       

for pop_name, path in experiments.items():  
  x_pop_all = pd.Series()
  y_pop_all = pd.Series()

  fly_dict=hf.load_files_from_folder(path)
 
  for fly_name, path in fly_dict.items():  
    df = pd.read_csv(path, usecols=['pos_x', 'pos_y'])
    
    if pop_name.startswith('CTRL'):
        x_all_ctrl = pd.concat([x_all_ctrl, df.pos_x], axis=0)
        y_all_ctrl = pd.concat([y_all_ctrl, df.pos_y], axis=0)
      
    else:
        x_all_coc = pd.concat([x_all_coc, df.pos_x], axis=0)
        y_all_coc = pd.concat([y_all_coc, df.pos_y], axis=0)


x = x_all_coc
y = y_all_coc
fig, axs = plt.subplots(2, 2)    
# sigmas = [0, 16, 32, 64]
# for ax, s in zip(axs.flatten(), sigmas):
#     if s == 0:
#         ax.plot(x, y, 'k.', markersize=0.01)
#         ax.set_aspect('equal', adjustable='box')
#         ax.set_title("Scatter plot")
#     else:
#         img, extent = myplot(x, y, s)
#         ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
#         ax.set_title("Smoothing with  $\sigma$ = %d" % s)

ax = plt.subplot()
img, extent = myplot(x, y, 16)
im = ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
# ax.set_title("Smoothing with  $\sigma$ = %d" % s)
plt.colorbar(im)

fig.set_size_inches(7, 7)

plt.tight_layout()
plt.title('COC')
plt.savefig('../3_output/COC_heatmap.png', dpi=350)
plt.show()  


x = x_all_ctrl
y = y_all_ctrl
fig, axs = plt.subplots(2, 2)    
# sigmas = [0, 16, 32, 64]
# for ax, s in zip(axs.flatten(), sigmas):
#     if s == 0:
#         ax.plot(x, y, 'k.', markersize=0.01)
#         ax.set_aspect('equal', adjustable='box')
#         ax.set_title("Scatter plot")
#     else:
#         img, extent = myplot(x, y, s)
#         ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
#         ax.set_title("Smoothing with  $\sigma$ = %d" % s)

ax = plt.subplot()
img, extent = myplot(x, y, 16)
im = ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
plt.colorbar(im)
# ax.set_title("Smoothing with  $\sigma$ = %d" % s)

fig.set_size_inches(7, 7)

plt.tight_layout()
plt.title('CTRL')
plt.savefig('../3_output/CTRL_heatmap.png', dpi=350)
plt.show()  
