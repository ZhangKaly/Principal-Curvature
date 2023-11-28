
import torch
import torch_geometric
from torch_geometric.datasets import TUDataset, FakeDataset
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
import itertools
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

# Getting all the paths of length k

def get_paths_length_k(A, k):
    """
    Returns the paths of length k in the graph A
    Shape : Num paths x k
    """
    last_A = A.clone()
    k_ = 1
    while k > k_:
        k_ +=1
        if k_ == 2:
            last_A = torch.einsum('ij,jk->ijk', A, A)
        elif k_==3:
            last_A = torch.einsum('ijk,kl->ijkl', last_A, A)
        elif k_ == 4:
            last_A = torch.einsum('ijkl,lm->ijklm', last_A, A)

    #paths = torch.unique(torch.stack(torch.where(last_A!=0),0).T.sort(1)[0],dim=0)
    paths = torch.stack(torch.where(last_A!=0),0).T
    return paths


# Create new node for each different path and link them to the original nodes
def update_graph(paths, nodes_dict, edge_index, path_length = 1):
    num_nodes = len(nodes_dict)
    for path in paths:
        nodes_dict[tuple(path.tolist())] = num_nodes
                
        p_length = path_length
        
        end_nodes_l = []
        while p_length>0:

            for i, j in itertools.combinations(range(len(path) + 1), 2):
                node = tuple(path.tolist()[i:j])
                #print(stuff[i:j])

            #end_nodes_ = itertools.combinations(path.tolist(), p_length)
            #for node in end_nodes_:
                end_nodes_l.append(nodes_dict[node])
            p_length -= 1

        source_nodes = torch.Tensor([num_nodes]).repeat(len(end_nodes_l)).long()
        
        new_edges_0 = torch.stack([source_nodes, torch.Tensor(end_nodes_l).long()],0)
        new_edges_1 = torch.stack([torch.Tensor(end_nodes_l).long(), source_nodes],0)
        new_edges = torch.cat([new_edges_0, new_edges_1],1)
        
        edge_index = torch.cat([edge_index, new_edges],1)
        num_nodes += 1
    return edge_index, nodes_dict


def path_transform(data):
    edge_index = data.edge_index # Getting the edge index of the graph

    A = torch_geometric.utils.to_dense_adj(data.edge_index)[0] # Creates adjacency matrix

    nodes_dict = {(i,): i for i in range(A.shape[0])} # Dictionary to keep track of the nodes

    paths = get_paths_length_k(A, 1) # Get all the paths of length 2
    edge_index_updated, nodes_dict = update_graph(paths, nodes_dict, edge_index, path_length = 1) # Append these paths to the graph

    paths = get_paths_length_k(A, 2) # Get all the paths of length 2
    edge_index_updated, nodes_dict = update_graph(paths, nodes_dict, edge_index, path_length = 2) # Append these paths to the graph

    data.x = torch.zeros((len(nodes_dict), 4))
    data.edge_index = edge_index_updated

    return data