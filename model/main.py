"""
THIS is still in progress.
TODO 1. add selection of manifold type to arguments
TODO 3. add support for removing irregular points
"""

import numpy as np
from tqdm import tqdm
import argparse
from manifolds import generate_point_cloud
import pathlib
from curvature import compute_curvature

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the range of evaluation IDs.")
    parser.add_argument("--eval_id_start", type=int, required=True, help="Starting ID for evaluation")
    parser.add_argument("--eval_id_end", type=int, required=True, help="Ending ID for evaluation")
    parser.add_argument("--if_generate_cloud", type=bool, required=True, help="Whether to generate the point cloud or not")
    parser.add_argument("--num_points", type=int, required=False, default=5000, help="Number of points in the point cloud")
    
    parser.add_argument("--file_path", type=str, required=False, default='.', help="result path")


    parser.add_argument("--manifold_type", type=str, required=True, help="Manifold type")

    parser.add_argument("--a", type=float, required=False, default=0.9, help="Semi-major axis")
    parser.add_argument("--b", type=float, required=False, default=1.25, help="Semi-minor axis")
    parser.add_argument("--c", type=float, required=False, default=0.9, help="Semi-minor axis")

    parser.add_argument("--r", type=float, required=False, default=0.375, help="small radius")
    parser.add_argument("--R", type=float, required=False, default=1, help="large radius")

    parser.add_argument("--seed", type=int, required=False, default=42, help="seed for random number generator")
    parser.add_argument("--epsilon_PCA", type=float, required=False, default=0.2, help="epsilon for PCA")
    parser.add_argument("--tau_ratio", type=float, required=True, help="Ratio of tau to be used for the curvature computation")

    parser.add_argument("--init", type=str, required=False, default=False, help="if true do not compute curvature")
    parser.add_argument("--noise", type=float, required=True, default=0.0, help="std of Gaussian noise to be added to the point cloud")

    args = parser.parse_args()
    eval_id_start = args.eval_id_start
    eval_id_end = args.eval_id_end
    tau_ratio = args.tau_ratio
    if_generate_cloud = args.if_generate_cloud
    num_points = args.num_points
    a = args.a
    b = args.b
    c = args.c
    r = args.r
    R = args.R
    seed = args.seed
    epsilon_PCA = args.epsilon_PCA
    manifold_type = args.manifold_type
    file_path = args.file_path
    # make the directory for the result
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
    init = args.init
    if init:
        if_generate_cloud = True
    noise = args.noise
    # print all the arguments
    print("--------------")
    print("Parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------")

    if if_generate_cloud:
        point_cloud_noiseless, K = generate_point_cloud(a=a, b=b, c=c, r=r, R=R, manifold_type=manifold_type, num_points=num_points, seed=42)
        np.random.seed(seed)
        point_cloud = point_cloud_noiseless + noise * np.random.multivariate_normal([0, 0, 0], np.identity(3), num_points)
        np.savetxt(file_path + f'/point_cloud_noiseless.csv', point_cloud_noiseless, delimiter=',')
        np.savetxt(file_path + f'/point_cloud.csv', point_cloud, delimiter=',')
        np.savetxt(file_path + f'/K.csv', K, delimiter=',')
    else:
        point_cloud = np.loadtxt(file_path + f'/point_cloud.csv', delimiter=',')

    if not init:
        num_eval = eval_id_end - eval_id_start
        assert num_eval > 0
        assert eval_id_start >= 0
        assert eval_id_end <= point_cloud.shape[0]
        print(f"evaluating points {eval_id_start}~{eval_id_end}")

        ## run in parallel
        from concurrent.futures import ProcessPoolExecutor

        def compute_curvature_for_point(i):
            j = i + eval_id_start
            print(f"point {j} started.")
            res = compute_curvature(point_cloud, np.expand_dims(point_cloud[j], axis=0), epsilon_PCA = 0.1, tau_ratio = tau_ratio)[1]
            print(f"point {j} finished.")
            return res

        with ProcessPoolExecutor() as executor:
            curvature = [x for x in tqdm(executor.map(compute_curvature_for_point, range(num_eval)), total=num_eval)]

        v = np.array(curvature).T
        np.savetxt(file_path + f'/curvature_from_{eval_id_start}_to_{eval_id_end}.csv', v, delimiter=',')