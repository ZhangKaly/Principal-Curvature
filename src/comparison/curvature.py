# import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
# import torch
# import pandas as pd
from sklearn.neighbors import NearestNeighbors
# from tqdm import tqdm
# import pandas as pd
import skdim

# def generate_torus_point_cloud(num_points = 5000, R = 3, r = 1):
#     # Generate random angles for theta and phi
#     theta = np.random.uniform(0, 2*np.pi, num_points)
#     phi = np.random.uniform(0, 2*np.pi, num_points)

#     # Compute the torus points
#     x = (R + r * np.cos(phi)) * np.cos(theta)
#     y = (R + r * np.cos(phi)) * np.sin(theta)
#     z = r * np.sin(phi)
    
#     K = np.cos(phi)/(r * (R + r * np.cos(phi))) 
    
#     return np.column_stack((x, y, z)), K

def find_basis(point_cloud, x,  extrin_dim = 3, epsilon_PCA = 0.1, tau_radius = 0.4, return_dim = False):
    #point_cloud: the manifold 
    #x: np.array of shape 1 by p, the point where the curvature is evaluated at, e.g., [[1, 2, 3]]
    #epsilon: the radius of local PCA
    #dim: the dimension of the manifold
    #tau_ratio: the ratio is tau radius (where we evaluate the curvature)/ epsilon_sqrt
    
    # Find transport neighborhood
    k = int(0.05 * point_cloud.shape[0])

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(point_cloud)
    ep_dist, ep_idx = nbrs.radius_neighbors(x, epsilon_PCA, return_distance=True)
    
    tau_dist, tau_idx = nbrs.radius_neighbors(x, tau_radius, return_distance=True)
    sorted_neighbors = sorted(zip(tau_dist[0], tau_idx[0]))
    sorted_dist, sorted_ind = zip(*sorted_neighbors)
    tau_nbrs = point_cloud[list(sorted_ind)]
    
    
    pca_nbrs = point_cloud[ep_idx[0]]
    Xi = pca_nbrs - x
    Di = np.diag(np.sqrt(np.exp(- np.array(ep_dist[0]) ** 2 / epsilon_PCA)))
    Bi = Xi.T @ Di
    
    U, S, VT = np.linalg.svd(Bi.T, full_matrices = False)
    O = VT[:extrin_dim, :]
    
    if return_dim:
        x_idx = sorted_ind[0]
        lpca = skdim.id.lPCA().fit_pw(point_cloud, n_neighbors = 10, n_jobs = 1)
        dim = lpca.dimension_pw_[x_idx]
        return tau_nbrs, dim, O
    else:
        return tau_nbrs, O
    
        
def compute_sectional_curvature(point_cloud, query_point, extrin_dim = 3, 
                                epsilon_PCA = 0.1, tau_radius = 0.4, max_min_num = 10, use_cross=False):
    
    tau_nbrs, O = find_basis(point_cloud, query_point, extrin_dim = extrin_dim,
                             epsilon_PCA = epsilon_PCA, tau_radius = tau_radius)

    if use_cross:
        O2 = np.cross(O[0], O[1])
    else:
        O2 = O[2]

    ti = tau_nbrs[1:] - tau_nbrs[0]
    norms = np.square(ti).sum(axis=1)
    tensor_all = 2 * (O2 * ti).sum(axis=1) / norms
    # tensor_all = []
    # for i in np.arange(1, len(tau_nbrs)):
    #     tensor = 2 * (sum(O2 *  (tau_nbrs[i] - tau_nbrs[0])))/np.linalg.norm(tau_nbrs[i] - tau_nbrs[0])**2
    #     tensor_all.append(tensor)

    if max_min_num < 1:
        min_quantile = max_min_num
        max_cur = np.quantile(tensor_all, 1-min_quantile)
        min_cur = np.quantile(tensor_all, min_quantile)
    else:
        max_cur = sum(sorted(tensor_all, reverse=True)[:max_min_num])/max_min_num    
        min_cur = sum(sorted(tensor_all)[:max_min_num])/max_min_num
    
    return max_cur * min_cur

def generate_ellipsoid_cloud(a, b, c, num_points = 5000):
    """Generate a random point on an ellipsoid defined by a,b,c"""
    
    theta = np.random.uniform(0, 2*np.pi, num_points)
    v = np.random.rand(num_points)
    phi = np.arccos(2.0 * v - 1.0)
    sinTheta = np.sin(theta);
    cosTheta = np.cos(theta);
    sinPhi = np.sin(phi);
    cosPhi = np.cos(phi);
    rx = a * sinPhi * cosTheta;
    ry = b * sinPhi * sinTheta;
    rz = c * cosPhi;
    return np.column_stack((rx, ry, rz))

