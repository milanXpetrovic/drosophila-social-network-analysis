import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from statistics import mean 

import package_functions as hf

DATA_PATH = '../2_pipeline/2_1_get_fly_distance_traveled/out'
SAVE_PATH = '../3_output/'

experiments = hf.load_files_from_folder(DATA_PATH)
columns = [name.replace('.csv', '') for name in experiments.keys()]
total = pd.DataFrame()

for pop_name, path in experiments.items():  
    df = pd.read_csv(path, index_col=0)
    total = pd.concat([total,df], axis=1)
    
total.columns = columns
df = total

coc_columns = [col for col in df if col.startswith('COC')]
ctrl_columns = [col for col in df if col.startswith('CTRL')]
df_coc = df[coc_columns].T.values.tolist()
df_ctrl = df[ctrl_columns].T.values.tolist()
df_coc = [[x for x in y if not np.isnan(x)] for y in df_coc]
df_ctrl = [[x for x in y if not np.isnan(x)] for y in df_ctrl]   

average_coc = mean([mean(e) for e in df_coc])
average_ctrl = mean([mean(e) for e in df_ctrl])

all_pop = [[x for x in y if not np.isnan(x)] for y in df.T.values.tolist()]
plt.boxplot(all_pop)

plt.axhline(y=average_ctrl, color='blue', label='Mean CTRL')
plt.axhline(y=average_coc, color='red', label='Mean COC')
plt.axvspan(0.5, len(df_coc)+0.5, alpha=0.05, color='red', label='COC popultaions')
plt.legend()
plt.title('COC vs CTRL distances walked distribution')

plt.savefig(SAVE_PATH + 'distances_walked.png', dpi=350)
plt.show()
plt.clf()


# coc_list = [j for i in df_coc for j in i]
# ctrl_list = [j for i in df_ctrl for j in i]

# plt.boxplot(all_pop)
# plt.axvspan(1.5, 2.5, alpha=0.05, color='red', label='COC popultaions')
# plt.legend()
# # plt.savefig('ctrl_vs_coc.png', dpi=350)
# plt.show()
# plt.clf()

# average_coc = [mean(e) for e in all_pop[0:6]]
# average_ctrl = [mean(e) for e in all_pop[6:]]
# all_pop3 = [average_ctrl, average_coc]

# plt.boxplot(all_pop3)
# plt.axvspan(1.5, 2.5, alpha=0.05, color='red', label='COC popultaions')
# plt.legend()
# plt.show()
