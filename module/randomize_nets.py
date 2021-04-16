
# def create_n_samples_of_random_graph(g, pop_name, n_samples, FPS):
#     total = pd.DataFrame()
#     for i in range(n_samples):
#         df = pd.DataFrame()
      
#         random_multi_g = generate_random_multigraph(g)
#         random_weighted_g = convert_multigraph_to_weighted(random_multi_g, FPS)
        
#         df = graph_global_measures(random_weighted_g, 'graph_' + str(i))
        
#         total = pd.concat([total, df], axis=1)  
    
#     graph_cols = [col for col in df if col.startswith('graph')]
    
#     df = pd.DataFrame()
#     #df['median'] = total.loc[:, graph_cols].median(axis=1)
#     df[pop_name.replace('.gml', '') + '_mean_random'] = total.loc[:, graph_cols].mean(axis=1)
#     # df['std'] = total.loc[:, graph_cols].std(axis=1)
    
#     return df
#             else:
#                 G[node_1][node_2]['count'] +=1
#                 G[node_1][node_2]['duration'] = old_duration + new_duration
                
#                 count = G[node_1][node_2]['count']
#                 duration = G[node_1][node_2]['duration']
                
#                 G[node_1][node_2]['frequency'] = float((count/duration)/0.5)
#                 G[node_1][node_2]['averaging'] = float(1/599.5)*(float(duration/count)-0.5)
#                 # created_edges +=1
             
#         else:
#             duration = random.uniform(0.5, 600)
#             count = 1  
            
#             frequency = float((count/duration)/0.5)
#             averaging = float(1/599.5)*(float(duration/count)-0.5)
          
#             G.add_edge(node_1, node_2,
#                     duration=duration,
#                     count=count,
#                     frequency=frequency,
#                     averaging=averaging)
#             # created_edges +=1       
            
#     return G


def generate_random_multigraph(g):
    edges = list(g.edges(data=True))
    weights = [weight.get('duration') for u, v, weight in edges]
    random_mg = nx.MultiGraph() 
    
    random_mg.add_nodes_from(g.nodes())
    
    while random_mg.number_of_edges()< len(g.edges()):  
        node_1, node_2 = random.sample(range(1, len(g.nodes())+1), 2)
        weight = weights.pop(0)
        random_mg.add_edge(node_1, node_2, duration=weight)
        
    return random_mg




def create_n_samples_of_random_graph(g, pop_name, n_samples, FPS):
    total = pd.DataFrame()
    for i in range(n_samples):
        df = pd.DataFrame()
      
        random_multi_g = generate_random_multigraph(g)
        random_weighted_g = convert_multigraph_to_weighted(random_multi_g, FPS)
        
        df = graph_global_measures(random_weighted_g, 'graph_' + str(i))
        
        total = pd.concat([total, df], axis=1)  
    
    graph_cols = [col for col in df if col.startswith('graph')]
    
    df = pd.DataFrame()
    #df['median'] = total.loc[:, graph_cols].median(axis=1)
    df[pop_name.replace('.gml', '') + '_mean_random'] = total.loc[:, graph_cols].mean(axis=1)
    # df['std'] = total.loc[:, graph_cols].std(axis=1)
    
    return df