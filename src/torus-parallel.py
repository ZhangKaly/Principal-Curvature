import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pandas as pd
from numpy import savetxt

def generate_torus_point_cloud(num_points = 5000, R = 3, r = 1):
    # Generate random angles for theta and phi
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, 2*np.pi, num_points)

    # Compute the torus points
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    return np.column_stack((x, y, z))


def sub_vectors_between(set_vectors, a, b):
    # this function selects a set of vectors whose entries are between a and b
    
    filtered_vector = [[x for x in v if a < x < b] for v in set_vectors ]
    return filtered_vector        

def list_vector_indices_upto(list_indices, list_of_nums):
    result_list = [[list_indices[i][j] for j in range(list_of_nums[i] + 1)] for i in range(len(list_indices))]
    return result_list

def list_vector_of_index(set_vectors, list_indices):
    
    list_result_vectors = [np.array([set_vectors[i] for i in list_indices[j]]) for j in range(len(list_indices))]
    return list_result_vectors 



def find_basis(point_cloud, x, epsilon_PCA = 0.1, dim = 2, tau_ratio = 1.5):
    #point_cloud: the manifold 
    #x: np.array of shape 1 by p, the point where the curvature is evaluated at, e.g., [[1, 2, 3]]
    #epsilon: the radius of local PCA
    #dim: the dimension of the manifold
    #tau_ratio: the ratio is tau radius (where we evaluate the curvature)/ epsilon_sqrt
    epsilon_sqrt = np.sqrt(epsilon_PCA)
    tau = tau_ratio * epsilon_sqrt

    # Number of neighbors to find, we take 5% of the total population
    k = int(0.2 * point_cloud.shape[0])
    
    # Create a NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(point_cloud)

    # Find k nearest neighbors
    dist_i, indx_i = nbrs.kneighbors(x)
    # Find epsilon neighborhood
    dist_epsilon = sub_vectors_between(dist_i, 0, epsilon_sqrt)
    len_dist_epsilon = [len(v) for v in dist_epsilon]
    epsilon_neighborhood = list_vector_of_index(point_cloud, list_vector_indices_upto(indx_i, len_dist_epsilon))[0]
    # Find tau neighborhood
    dist_tau = sub_vectors_between(dist_i, 0, tau)
    len_dist_tau = [len(v) for v in dist_tau]
    tau_neighborhood = list_vector_of_index(point_cloud, list_vector_indices_upto(indx_i, len_dist_tau))[0]
    num = len(tau_neighborhood)
    
    distances, indices = nbrs.kneighbors(tau_neighborhood)
    
    distances_epsilon = sub_vectors_between(distances, 0, epsilon_sqrt) # this is the list of distances in the epsilon
    list_len_dist_epsilon = [len(v) for v in distances_epsilon] #this gives the list of lengths in the distance_epsilon 
    
    tau_epsilon_neighborhood = list_vector_of_index(point_cloud, list_vector_indices_upto(indices, list_len_dist_epsilon))
    list_X_i = [tau_epsilon_neighborhood[i][1:] - tau_neighborhood[i] for i in range(num)]
    
    
    #list_D_i = [np.diag(np.sqrt(np.exp(- np.array(distances_epsilon[i]) ** 2 / epsilon_PCA))) for i in range(num)]
    list_D_i = [np.diag(np.sqrt(np.exp( - 5 * np.array(distances_epsilon[i]) ** 2 / epsilon_PCA))) for 
                i in range(num)]
    list_B_i = [list_X_i[j].T @ list_D_i[j] for j in range(num)]
    O = []
    for q in range(num):
        U, S, VT = np.linalg.svd(list_B_i[q], full_matrices = False)
        O_i = U[:dim, :]
        O.append(O_i)
        
    return epsilon_neighborhood, tau_neighborhood, tau_epsilon_neighborhood, O

def compute_curvature(point_cloud, query_point, epsilon_PCA = 0.1, dim = 2, tau_ratio = 1.5):
    
    ep_neighbor, tau_neighbor, tau_epsilon_neighbor, O = find_basis(point_cloud, query_point, epsilon_PCA = epsilon_PCA, dim = dim, tau_ratio = tau_ratio)
    
    transport_maps = np.zeros((len(tau_neighbor), len(tau_neighbor), dim, dim))
    for i in range(len(tau_neighbor)):
        for j in range(len(tau_neighbor)):
            U, S, VT = np.linalg.svd(O[i] @ O[j].T, full_matrices = False)
            O_ij = U @ VT
            transport_maps[i,j] = O_ij
            
    tensor_av = []

    O_init = O[0]
    for i in np.arange(1, len(tau_neighbor)):
        for j in np.arange(i + 1, len(tau_neighbor)):
            O_fin = O_init.T @ transport_maps[0, i] @ transport_maps[i, j] @ transport_maps[j, 0]
            v_init = O_init[0]
            v_fin = O_fin.T[0]
            
            cosin = (v_init @ v_fin.T) / (np.linalg.norm(v_init) * np.linalg.norm(v_fin))
            angle = np.arccos(cosin)
            area = np.linalg.norm(np.cross(tau_neighbor[i] - tau_neighbor[0], tau_neighbor[j] -tau_neighbor[0])) / 2
            
            #tensor = angle / area
            tensor = (2 * np.pi - angle) / area 
            
            tensor_av.append(tensor)
    tensor_av = np.exp(sum(tensor_av)/len(tensor_av) * 0.0001)
  
    
            
    return transport_maps, tensor_av         
    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the range of evaluation IDs.")
    parser.add_argument("--eval_id_start", type=int, required=True, help="Starting ID for evaluation")
    parser.add_argument("--eval_id_end", type=int, required=True, help="Ending ID for evaluation")
    args = parser.parse_args()
    eval_id_start = args.eval_id_start
    eval_id_end = args.eval_id_end

    # ellipsoid = generate_ellipsoid_cloud(1, 2, 0.5, num_points=5000, seed=42)
    # savetxt('ellipsoid_cloud_ratio_4.csv', ellipsoid, delimiter=',')
    torus = np.loadtxt('torus_cloud.csv', delimiter=',')
    # num_eval = int(len(ellipsoid)/5)
    # num_eval = 5000
    num_eval = eval_id_end - eval_id_start
    assert num_eval > 0
    assert eval_id_start >= 0
    assert eval_id_end <= 5000
    print(f"evaluating points {eval_id_start}~{eval_id_end}")
    # curvature = []
    # for i in tqdm(range(num_eval)):
    #     a, b = compute_curvature(ellipsoid, np.expand_dims(ellipsoid[i], axis=0), epsilon_PCA = 0.1, tau_ratio = 4)
    #     curvature.append(b)

    ## run in parallel
    from concurrent.futures import ProcessPoolExecutor

    def compute_curvature_for_point(i):
        j = i + eval_id_start
        print(f"point {j} started.")
        res = compute_curvature(torus, np.expand_dims(torus[j], axis=0), epsilon_PCA = 0.1, tau_ratio = 4)[1]
        print(f"point {j} finished.")
        return res

    with ProcessPoolExecutor() as executor:
        # curvature = list(tqdm(executor.map(compute_curvature_for_point, range(num_eval)), total=num_eval))
        curvature = [x for x in tqdm(executor.map(compute_curvature_for_point, range(num_eval)), total=num_eval)]

    v = np.array(curvature).T
    savetxt(f'curvature_torus_from_{eval_id_start}_to_{eval_id_end}.csv', v, delimiter=',')