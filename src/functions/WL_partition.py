import copy
import pprint
import itertools
import hashlib
import networkx as nx
import numpy as np

from collections import Counter

def base_WL(G_, k, verbose, n_set, initial_colors_func, find_neighbors_func):    
    if verbose:
        print('-----------------------------------')
        print('Starting the execution for the graph')
    G, n = n_set(G_)
    colors = initial_colors_func(n)

    old_colors = copy.deepcopy(colors)

    colors_list = [copy.deepcopy(colors)]
    
    if verbose:
        print(f'Initial Color hashes: \n {colors} \n')

    for i in range(len(n)):

        for node in n:
            neigh_colors = [old_colors[i][0] for i in find_neighbors_func(G, n, node)]
            neigh_colors.sort()
            neigh_colors = "-".join(neigh_colors)
            
            colors[node].extend([neigh_colors])
            #colors[node].sort()

        # Update with the hash
        if verbose:
            print(f'Colors before hashes at iteration {i}: {colors} \n')
            print(f"Neighbors : ")
            for node in n:
                print(f"{node} : {find_neighbors_func(G, n, node)}")
        colors = {i: [hashlib.sha224("".join(colors[i]).encode('utf-8')).hexdigest()] for i in colors.keys()}
                
        if verbose:
            print(f'Colors hashes at iteration {i}: \n {colors} \n')
            print(f'Histogram: \n {sorted(Counter([item for sublist in colors.values() for item in sublist]).items())} \n')
        
        if list(Counter([item for sublist in colors.values() for item in sublist]).values()) == list(Counter([item for sublist in old_colors.values() for item in sublist]).values()) and i != 0:
            if verbose:
                print(f'Converged at iteration {i}!')
            break
        
        old_colors = copy.deepcopy(colors)
        colors_list.append(copy.deepcopy(colors))

    canonical_form = sorted(Counter([item for sublist in colors.values() for item in sublist]).items())
    if verbose:
        print(f'Canonical Form Found: \n {canonical_form} \n')

    return canonical_form, colors_list


def WL(G, k=2, verbose=False):
    def n_set(G):
        G = nx.convert_node_labels_to_integers(G)
        return G, list(G.nodes())
    
    def set_initial_colors(n):
        return {i: [hashlib.sha224("1".encode('utf-8')).hexdigest()] for i in n}
    
    def find_neighbors(G, n, node):
        return G.neighbors(node)
    
    return base_WL(G, k, verbose, n_set, set_initial_colors, find_neighbors)

def kWL(G, k, verbose=False):
    def n_set(G):
        G = nx.convert_node_labels_to_integers(G)
        V = list(G.nodes())
        V_k = [comb for comb in itertools.combinations(V, k)]
        return G, V_k

    def set_initial_colors(n):
        pairs = list(itertools.combinations(np.arange(k),2))
        hash_dict = {}
        for i in n:
            str_ = []
            for p in pairs:
                node1 = i[p[0]]
                node2 = i[p[1]]
                if G.has_edge(node1,node2):
                    str_.append('E')
                else:
                    str_.append('0')
            hash_dict[i] = [hashlib.sha224(str(str_).encode('utf-8')).hexdigest()]

        return hash_dict

    def find_neighbors(G, V_k, node):
        if verbose:
            print([n for n in V_k if len(set(n) - set(V_k[V_k.index(node)])) == 1])
        return [n for n in V_k if len(set(n) - set(V_k[V_k.index(node)])) == 1]

    return base_WL(G, k, verbose, n_set, set_initial_colors, find_neighbors)


def partition_graph(G,k, verbose = False):
    if k==1:
        cform, c_list = WL(G,k=1,verbose=verbose)
    elif k==2:
        cform, c_list= kWL(G,k=2,verbose=verbose)
    else:
        raise NotImplementedError
    
    N = len(G)

    processed = []
    for c in c_list:
        processed.append({})
        for k_,v in c.items():
            processed[-1][k_] = v[0]

    if k==1:
        meta_processed = processed
    else: # key is a tuple (edges) - need another round of hashing
        meta_processed = []
        for proc in processed:
            cols = {}
            for n in range(N):
                meta_nodes = [v for k_,v in proc.items() if n in k_]
                meta_nodes.sort()
                col_ = hashlib.sha224("".join(meta_nodes).encode('utf-8')).hexdigest()
                cols[n] = col_
            meta_processed.append(cols)

    #Transforming into index partition
    partition = []
    for s in meta_processed:
        s_u = np.unique(list(s.values()))
        partition_ = []
        for s_u_ in s_u:
            partition_.append([k_ for k_,v in s.items() if v == s_u_])
        partition.append(partition_)

    #Dropping the partition that repeat
    prev_p = set()
    filtered_partition = []
    for p_ in partition:
        new_set = set([tuple(i) for i in p_])
        if new_set == prev_p:
            break
        filtered_partition.append(p_)
        prev_p = new_set

    return filtered_partition


if __name__ == "__main__":
    G = nx.Graph()
    G_edge_list = [(0, 1), (1, 2), (2, 0),(2,3),(3,4),(4,5),(5,3),(0,5),(4,6),(6,7)]
    G.add_edges_from(G_edge_list)
    
    print("1-WL")
    print(partition_graph(G, k=1))

    print("2-WL")
    print(partition_graph(G, k=2))