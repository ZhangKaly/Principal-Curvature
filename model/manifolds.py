import numpy as np
import pandas as pd

def ellipsoid_ground_truth(point_cloud, a, b, c):
    #point_cloud: N by 3 array
    cur = []
    for p in point_cloud:
        k = 1 / (a **2 * b**2 * c**2 * (p[0]**2 / a**4 + p[1]**2 / b**4 + p[2]**2 / c**4) **2)
        cur.append(k)
    return cur

def hyperboloid_ground_truth(point_cloud, a, b, c):
    #point_cloud: N by 3 array
    cur = []
    for p in point_cloud:
        k = - c **6 / (c **4 + a**2 * p[2]**2 + b**2 * p[2]**2) **2
        cur.append(k)
    return cur
        
def generate_torus_cloud(num_points = 5000, R = 3, r = 1, seed=42):
    # Generate random angles for theta and phi
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, 2*np.pi, num_points)

    # Compute the torus points
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    K = np.cos(phi)/(r * (R + r * np.cos(phi))) 

    return np.column_stack((x, y, z)), K

def generate_hyperboloid_cloud(a, b, c, num_points=5000, seed=42):
    np.random.seed(seed)

    u = np.random.uniform(-1.5,1.5,num_points)
    v = np.random.uniform(0, 2*np.pi, num_points)

    x = a*np.sqrt(1 + np.square(u)) * np.cos(v)
    y = b*np.sqrt(1 + np.square(u)) * np.sin(v)
    z = c*u

    K = - c**2 / np.square(c**2 + (a**2 + c**2) * np.square(u))
    return np.column_stack((x, y, z)), K

def generate_ellipsoid_cloud(a, b, c, num_points = 5000, seed=42):
    """Generate a random point on an ellipsoid defined by a,b,c"""
    np.random.seed(seed)
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
    ellipsoid = np.column_stack((rx, ry, rz))
    K = ellipsoid_ground_truth(ellipsoid, a, b, c)
    return ellipsoid, K

def generate_point_cloud(a, b, c, r, R, manifold_type, num_points, seed=42):
    if manifold_type == 'torus':
        point_cloud, K = generate_torus_cloud(num_points, R, r, seed) 
    elif manifold_type == 'hyperboloid':
        point_cloud, K = generate_hyperboloid_cloud(a, b, c, num_points, seed)
    elif manifold_type == 'ellipsoid':
        point_cloud, K = generate_ellipsoid_cloud(a, b, c, num_points, seed)
    else:
        raise ValueError('manifold_type not recognized')
    return point_cloud, K