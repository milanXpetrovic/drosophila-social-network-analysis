import pandas as pd 
import networkx as nx

import sys
sys.path.append(r"F:\1_coding_projects\my_module")

import module.import_and_preproc as toolkit
import module.my_module as hf
import module.networks_tools as netkit

DATA_PATH = '../2_pipeline/1_0_undirected_singleedge_graph/out/'
SAVE_PATH = '../3_output/'

experiments = hf.load_files_from_folder(DATA_PATH, file_format='.gml')
         

def do_stuff(total):
    res = {}
    for index, row in total.iterrows():
        d = {}
        row = dict(row)
        
        d = {}
        coc_values = []
        ctrl_values = []
        
        for pop_name, values in row.items():
        
            if pop_name.startswith('COC'):
                coc_values.append(values)
    
            else:
                ctrl_values.append(values)
        
        d.update({'COC': coc_values})
        d.update({'CTRL': ctrl_values})
        
        res.update({index: d})
    
    return res


total = pd.DataFrame()
for pop_name, path in experiments.items():
    df = pd.DataFrame()
    
    g = nx.read_gml(path)
    
    remove = [node for node,degree in dict(g.degree()).items() if degree < 0]
    g.remove_nodes_from(remove)
    
    
    df = hf.graph_global_measures(g, pop_name)
    
    total = pd.concat([total, df], axis=1)

res = do_stuff(total)

total = hf.stat_test(res)
total = total.round(decimals=3)

total.to_csv(SAVE_PATH+'global_measures_ttest.csv')
total.to_latex(SAVE_PATH+'global_measures_ttest.tex')  





















# total = {}
# for foo_name, foo in graph_functions:
#     # graphs_d = {exp_name: mmm.remove_nodes_with_degree_less_than(g, 4) for exp_name, g in graphs_d.items()}
    
#     values = {exp_name: foo(g) for exp_name, g in graphs_d.items()}
    
#     values = hf.group_values(values)
    
#     total.update({foo_name: values})

# total = hf.stat_test(total)

# total = hf.order_columns(total)
# total = total.round(decimals=3)
# total = total.loc[:, 'median_COC':'std_CTRL']

# total.to_latex(SAVE_PATH+'global_graph_measures_no_single.tex')

# total.to_excel(SAVE_PATH+'global_graph_measures_no_single.xlsx')

#ADD
#'reciprocity', 'fragmentation'


