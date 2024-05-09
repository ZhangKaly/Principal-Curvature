import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pandas as pd
import skdim
from scipy.stats import pearsonr
from numpy import savetxt
from sklearn.metrics import mean_squared_error


def generate_torus_point_cloud(num_points = 5000, R = 3, r = 1):
    # Generate random angles for theta and phi
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, 2*np.pi, num_points)

    # Compute the torus points
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    x_c = R * np.cos(theta)
    y_c = R * np.sin(theta)
    z_c = np.zeros(x.shape)
    
    K = np.cos(phi)/(r * (R + r * np.cos(phi))) 
    
    

    return np.column_stack((x, y, z)), np.column_stack((x_c, y_c, z_c)), K

# Create torus parameters
R = 1 # Major radius
r = 0.375  # Minor radius
num_samples = 5000



# Generate a torus point cloud with 1000 points and radius 1
torus, torus_centers, torus_K = generate_torus_point_cloud(num_points = num_samples, R = R, r = r)

torus += np.random.normal(0, 0.2, torus.shape)

def epsilon_and_tau(point_cloud, query):
    ratio_all = []
    for i in range(1, 250):
        epsilon_PCA = 0.01 * i
        nbrs = NearestNeighbors(n_neighbors=2500, algorithm='ball_tree').fit(point_cloud)
        ep_dist, ep_idx = nbrs.radius_neighbors(query, epsilon_PCA, return_distance=True, sort_results = True)

        pca_nbrs = point_cloud[ep_idx[0]]
        Xi = pca_nbrs - query
        Di = np.diag(np.sqrt(np.exp(-1*np.array(ep_dist[0]) ** 2 / epsilon_PCA)))
        Bi = Xi.T @ Di
    
        U, S, VT = np.linalg.svd(Bi.T, full_matrices = True)
        if len(S)>= 2: 
            ratio = (S[0]+ S[1])/sum(S)
        else:
            ratio = 1.0
        ratio_all.append(ratio)
    tau = 0.01 * np.argmin(np.array(ratio_all) - min(ratio_all))
    eps_PCA = 0.01 * np.argmin(np.abs(np.array(ratio_all) - 0.75))
    return eps_PCA, tau

def find_basis(point_cloud, x,  extrin_dim = 3):
    #point_cloud: the manifold 
    #x: np.array of shape 1 by p, the point where the curvature is evaluated at, e.g., [[1, 2, 3]]
    #epsilon: the radius of local PCA
    #dim: the dimension of the manifold
    #tau_ratio: the ratio is tau radius (where we evaluate the curvature)/ epsilon_sqrt
    
    # Find transport neighborhood
    k = int(0.05 * point_cloud.shape[0])
    
    epsilon_PCA, tau = epsilon_and_tau(point_cloud, x)


    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(point_cloud)
    ep_dist, ep_idx = nbrs.radius_neighbors(x, epsilon_PCA, return_distance=True, sort_results = True)
    
       
    
    #tau_dist, tau_idx = nbrs.kneighbors(x, k, return_distance=True)
    tau_dist, tau_idx = nbrs.radius_neighbors(x, tau, return_distance=True, sort_results = True)
    
    tau_nbrs = point_cloud[tau_idx[0]]
    
    
    pca_nbrs = point_cloud[ep_idx[0]]
    Xi = pca_nbrs - x
    Di = np.diag(np.sqrt(np.exp(- np.array(ep_dist[0]) ** 2 / epsilon_PCA)))
    Bi = Xi.T @ Di
    
    U, S, VT = np.linalg.svd(Bi.T, full_matrices = False)
    O = VT[:extrin_dim, :]
    
    return tau_nbrs[1:], tau_dist[0][1:], tau, O,
   
    
        
    

#this is the new one for tau automatically tuned
def compute_curvature_adaptive(point_cloud, query_point, extrin_dim = 3, use_cross = True):
    
    tau_nbrs, tau_dist, tau, O = find_basis(point_cloud, query_point, extrin_dim = extrin_dim)
 
    if use_cross:
        O2 = np.cross(O[0], O[1])
    else:
        O2 = O[2]

    ti = tau_nbrs - query_point[0]
    norms = np.square(ti).sum(axis=1)
    tensor_all = 2 * (O2 * ti).sum(axis=1) / norms
    
    
    max_min_num = int(0.25 * len(tau_nbrs))
    #max_min_num = 250
    
    max_indices = np.argsort(tensor_all)[-max_min_num: ]
    max_cur = tensor_all[max_indices]
    
    min_indices = np.argsort(tensor_all)[:max_min_num]
    min_cur = tensor_all[min_indices]
    
    max_cur_weight = np.sqrt(np.exp(-0.5 * np.array(tau_dist[max_indices]) ** 2 / np.sqrt(tau)))
    min_cur_weight = np.sqrt(np.exp(-0.5 * np.array(tau_dist[min_indices]) ** 2 / np.sqrt(tau)))
    
    principal_cur1 = sum(max_cur_weight * max_cur)/sum(max_cur_weight)
    principal_cur2 = sum(min_cur_weight * min_cur)/sum(min_cur_weight)
    #principal_cur1 = sum(max_cur)/len(max_cur)
    #principal_cur2 = sum(min_cur)/len(min_cur)
    
    return principal_cur1 * principal_cur2 
 

num_eval = int(len(torus))
gaussian_cur = []
for i in tqdm(range(num_eval)):
    g = compute_curvature_adaptive(torus, torus[i].reshape(1, -1))
    gaussian_cur.append(g)
    
v = np.array(gaussian_cur).T
savetxt('torus_noise2.csv', v, delimiter=',')

corr, _ = pearsonr(torus_K , v)
rmse = np.sqrt(mean_squared_error(torus_K, v))

stat = np.array([[rmse, corr]])

savetxt('stats_torus_noise2.csv', stat, delimiter=',')




    


