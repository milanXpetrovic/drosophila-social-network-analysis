# %%
from re import X
import sys
import yaml
import math
import numpy as np
import pandas as pd
import package_functions as pf


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


print(rotate((0,0), (-34.2, -15.9), 1.328))
#%%

CONFIG = '../config.yaml'

with open(CONFIG) as f:
    config = yaml.safe_load(f)

INPUT_DATA = '../data/preproc/0_0_preproc_data'
SAVE_PATH = '../data/preproc/0_2_distances_between_flies_matrix'

experiments = pf.load_multiple_folders(INPUT_DATA)
for pop_name, path in experiments.items():

    files = pf.load_dfs_to_list(path)

    df_angles = pd.DataFrame()

    for i in range(len(files)):
        df1 = files[i]

        ## OVDJE BI MOGAO ZA SVAKU MUHU BROJATI ODMAH SVE KUTEVE 
        ## KOJI IMAJU DISTANCU MANJU OD NPR 20 BODY LENGTHS
        ## COUNTER NA 360/5 * 10 
        ## DISKRETIZIRATI KUTOVE U BINOVE PO 5
        ## 360 JE DEGREE, ZNACI 72 BINA
        ## 10 JE BODY LENGTHSA
        
        for j in range(len(files)):
            if i == j:
                pass
            else:
                name = 'fly'+str(i+1) + ' ' + 'fly'+str(j+1)
                df2 = files[j]

                df = pd.concat([df1['ori'], df2['ori']], axis=1)
                df.columns = ['ori1', 'ori2']

                df['x1'] = (df1['pos_x']-df1['pos_x'])
                df['y1'] = (df1['pos_y']-df1['pos_y'])
                df['x2'] = (df2['pos_x']-df1['pos_x'])
                df['y2'] = (df2['pos_y']-df1['pos_y'])

                ox = df['x1']
                oy = df['y1']
                px = df['x2']
                py = df['y2']

                angle = -df['ori1']
   
                df['qx2'] = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
                df['qy2'] = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
                df['qx2'] = df['qx2']
                df['qy2'] = df['qy2']

                df['x_axis_dif'] = (df['x1'] - df['x2']).abs()
                df['y_axis_dif'] = (df['y1'] - df['y2']).abs()

                df['distance'] = np.sqrt(np.square(df['x_axis_dif']) +
                            np.square(df['y_axis_dif']))

                df['qx_axis_dif'] = (df['x1'] - df['qx2']).abs()
                df['qy_axis_dif'] = (df['y1'] - df['qy2']).abs()

                df['qdistance'] = np.sqrt(np.square(df['x_axis_dif']) +
                            np.square(df['y_axis_dif']))

                df['angle'] = np.arcsin(df['qy_axis_dif']/df['qdistance'])

                df['angle'] =  np.rad2deg(df['angle'])

                df['angle5'] = round(df['angle']/5.0)*5.0
                df_angles[name] = df['angle']

                
                res = df[df.qdistance <= 10*8.5]
                res = res[['qdistance', 'angle5']]
                res['qdistance'] = (res['qdistance']/8.5).round()

                if len(res):
                    import sys
                    sys.exit()

                ## TREBA PLOTATI U KRUG OVU DISTRIBUCIJU 

    import sys
    sys.exit()
    
# %%

print(len(files))

res = df_angles[['fly1 fly2', 'fly2 fly1']]
res = res.head()