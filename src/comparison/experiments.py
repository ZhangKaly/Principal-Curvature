import sys
# import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
from tqdm import tqdm
import pandas as pd
# import skdim
from .point_clouds import generate_point_cloud
from .curvature import compute_sectional_curvature
from .abby_curvature import scalar_curvature_est
from .diffusion_curvature import diffusion_curvature

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import pathlib

def compute_curvatures(point_cloud, gt_curvature, epsilon_PCA=0.2, tau_radius=1, max_min_num=100, use_cross=False):
    gt_curvature = np.array(gt_curvature)
    num_eval = int(len(point_cloud))
    our_curvature = []
    for i in tqdm(range(num_eval)):
        try:
            b = compute_sectional_curvature(point_cloud, point_cloud[i].reshape(1, -1), 
                                        epsilon_PCA =epsilon_PCA, tau_radius=tau_radius, max_min_num=max_min_num, use_cross=use_cross)
        except IndexError:
            print(f"Index Error")
            b = np.nan
        # b = compute_sectional_curvature(point_cloud, point_cloud[i].reshape(1, -1), 
                                        # epsilon_PCA =epsilon_PCA, tau_radius=tau_radius, max_min_num=max_min_num, use_cross=use_cross)
        our_curvature.append(b)
    our_curvature = np.array(our_curvature)
    n = 2
    sce_point_cloud = scalar_curvature_est(n, point_cloud)
    ab_curvature = sce_point_cloud.estimate(rmax=np.pi/2)
    ab_curvature = np.array(ab_curvature)
    dc_curvature = diffusion_curvature(point_cloud)
    return dict(
        gt_curvature=gt_curvature,
        our_curvature=our_curvature,
        ab_curvature=ab_curvature,
        dc_curvature=dc_curvature
    )

def plot_curvatures(point_cloud, curvature_dict, save=False, save_path='comparison.pdf'):
    gt_curvature, our_curvature, ab_curvature, dc_curvature = curvature_dict['gt_curvature'], curvature_dict['our_curvature'], curvature_dict['ab_curvature'], curvature_dict['dc_curvature']
    fig = plt.figure(figsize=(12, 3))
    v = our_curvature
    ax1 = fig.add_subplot(141, projection='3d')
    scatter = ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=0.5, c=v)
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.set_title("ours")
    fig.colorbar(scatter, ax=ax1, shrink=0.5)

    v = dc_curvature
    ax2 = fig.add_subplot(142, projection='3d')
    scatter = ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=0.5, c=v)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.set_title("Diffusion Curvature")
    fig.colorbar(scatter, ax=ax2, shrink=0.5)

    v = ab_curvature
    ax3 = fig.add_subplot(143, projection='3d')
    scatter = ax3.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=0.5, c=v)
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.set_title("H. & B.")
    fig.colorbar(scatter, ax=ax3, shrink=0.5)

    v = gt_curvature
    ax4 = fig.add_subplot(144, projection='3d')
    scatter = ax4.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=0.5, c=v)
    ax4.set_aspect('equal')
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_zticks([])
    ax4.set_title("Ground Truth curvature")
    fig.colorbar(scatter, ax=ax4, shrink=0.5)

    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=300)
    plt.show()

def compare_metrics(curvature_dict):
    gt_curvature, our_curvature, ab_curvature, dc_curvature = curvature_dict['gt_curvature'], curvature_dict['our_curvature'], curvature_dict['ab_curvature'], curvature_dict['dc_curvature']
    corr_dict = dict(
        our=np.corrcoef(our_curvature, gt_curvature)[0, 1],
        ab=np.corrcoef(ab_curvature, gt_curvature)[0, 1],
        dc=np.corrcoef(dc_curvature, gt_curvature)[0, 1]
    )
    mse_dict = dict(
        our=np.sqrt(np.mean((our_curvature - gt_curvature)**2)),
        ab=np.sqrt(np.mean((ab_curvature - gt_curvature)**2)),
        dc=np.sqrt(np.mean((dc_curvature - gt_curvature)**2))
    )
    r2_dict = dict(
        our=r2_score(gt_curvature, our_curvature),
        ab=r2_score(gt_curvature, ab_curvature),
        dc=r2_score(gt_curvature, dc_curvature)
    )
    result_df = pd.DataFrame([corr_dict, mse_dict, r2_dict], index=['corrcoef', 'rmse', 'R2'])
    return result_df

def do_experiment(manifold_type, save, save_path, noise=0.0, **kwargs):
    point_cloud, gt_curvature = generate_point_cloud(
        a=1.5,
        b=0.9,
        c=0.9, 
        r=0.375, 
        R=1, 
        manifold_type=manifold_type, 
        num_points=5000, 
        seed=42
    )
    np.random.seed(42)
    point_cloud += np.random.normal(0, noise, point_cloud.shape)
    res_dict = compute_curvatures(point_cloud, gt_curvature, **kwargs)
    plot_curvatures(point_cloud, res_dict, save=save, save_path=f"{save_path}/plot.pdf")
    res_df = compare_metrics(res_dict)
    if save:
        res_df.to_csv(f"{save_path}/metrics.csv")
    return res_df

def make_mfd(manifold_type, save_path, noise=0.0, seed=42):
    point_cloud, gt_curvature = generate_point_cloud(
        a=1.5,
        b=0.9,
        c=0.9, 
        r=0.375, 
        R=1, 
        manifold_type=manifold_type, 
        num_points=5000, 
        seed=seed
    )
    np.random.seed(seed)
    point_cloud += np.random.normal(0, noise, point_cloud.shape)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_path}/point_cloud.npy", point_cloud)
    np.save(f"{save_path}/gt_curvature.npy", gt_curvature)

def run_ours(point_cloud, epsilon_PCA=0.2, tau_radius=1, max_min_num=100, use_cross=True):
    num_eval = int(len(point_cloud))
    our_curvature = []
    for i in tqdm(range(num_eval)):
        try:
            b = compute_sectional_curvature(point_cloud, point_cloud[i].reshape(1, -1), 
                                        epsilon_PCA =epsilon_PCA, tau_radius=tau_radius, max_min_num=max_min_num, use_cross=use_cross)
        except IndexError:
            print(f"Index Error")
            b = np.nan
        our_curvature.append(b)
    our_curvature = np.array(our_curvature)
    return our_curvature

def run_ab(point_cloud, n=2):
    sce_point_cloud = scalar_curvature_est(n, point_cloud)
    ab_curvature = sce_point_cloud.estimate(rmax=np.pi/2)
    ab_curvature = np.array(ab_curvature)
    return ab_curvature

def run_dc(point_cloud):
    dc_curvature = diffusion_curvature(point_cloud)
    return dc_curvature

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig):
    if cfg.logger.use_wandb:
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run = wandb.init(
            entity=cfg.logger.entity,
            project=cfg.logger.project,
            tags=cfg.logger.tags,
            reinit=True,
            config=config,
            settings=wandb.Settings(start_method="spawn"),
        )
    # import pdb; pdb.set_trace()
    # for noise in cfg.path.noises:
    #     for name in cfg.path.names:
    noise = cfg.path.noise
    name = cfg.path.name
    data_path = f"{cfg.path.root}/data/{name}/{noise}"
    point_cloud = np.load(f"{data_path}/point_cloud.npy")
    gt_curvature = np.load(f"{data_path}/gt_curvature.npy")
    if cfg.method.name == 'ab':
        ab_curvature = run_ab(point_cloud)
        # np.save(f"{data_path}/ab_curvature.npy", ab_curvature)
        if cfg.logger.use_wandb:
            wandb.log({'coef': np.corrcoef(ab_curvature, gt_curvature)[0, 1]})
            wandb.log({'rmse': np.sqrt(np.mean((ab_curvature - gt_curvature)**2))})
            wandb.log({'R2': r2_score(gt_curvature, ab_curvature)})
            wandb.log({'curvature': ab_curvature.tolist()})

    elif cfg.method.name == 'dc':
        dc_curvature = run_dc(point_cloud)
        # np.save(f"{data_path}/dc_curvature.npy", dc_curvature)
        if cfg.logger.use_wandb:
            wandb.log({'coef': np.corrcoef(dc_curvature, gt_curvature)[0, 1]})
            wandb.log({'rmse': np.sqrt(np.mean((dc_curvature - gt_curvature)**2))})
            wandb.log({'R2': r2_score(gt_curvature, dc_curvature)})
            wandb.log({'curvature': dc_curvature.tolist()})
    elif cfg.method.name == 'ours':
        try:
            our_curvature = run_ours(point_cloud, epsilon_PCA=cfg.our_param.epsilon_PCA, tau_radius=cfg.our_param.tau_radius, max_min_num=cfg.our_param.max_min_num, use_cross=cfg.our_param.use_cross)
            # np.save(f"{data_path}/our_curvature.npy", our_curvature)
            if cfg.logger.use_wandb:
                wandb.log({'coef': np.corrcoef(our_curvature, gt_curvature)[0, 1]})
                wandb.log({'rmse': np.sqrt(np.mean((our_curvature - gt_curvature)**2))})
                wandb.log({'R2': r2_score(gt_curvature, our_curvature)})
                wandb.log({'curvature': our_curvature.tolist()})
        except IndexError:
            print(f"Error in {data_path}")
            if cfg.logger.use_wandb:
                wandb.log({'coef': np.nan})
                wandb.log({'rmse': np.nan})
                wandb.log({'R2': np.nan})
                wandb.log({'curvature': np.nan})

    if cfg.logger.use_wandb:
        run.finish()

if __name__ == "__main__":
    main()