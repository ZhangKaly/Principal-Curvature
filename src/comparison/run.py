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
from point_clouds import generate_point_cloud
from curvature import compute_sectional_curvature
from abby_curvature import scalar_curvature_est
from diffusion_curvature import diffusion_curvature

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import pathlib

from experiments import run_ab, run_dc, run_ours, get_non_na

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
        our_curvature = run_ours(point_cloud, epsilon_PCA=cfg.our_param.epsilon_PCA, tau_radius=cfg.our_param.tau_radius, max_min_num=cfg.our_param.max_min_num, use_cross=cfg.our_param.use_cross)
        # np.save(f"{data_path}/our_curvature.npy", our_curvature)
        if cfg.logger.use_wandb:
            wandb.log({'curvature': our_curvature.tolist()})
            non_na_percent = np.mean(~np.isnan(our_curvature))
            wandb.log({'non_na_percent': non_na_percent})
            our_curvature, gt_curvature = get_non_na(our_curvature, gt_curvature)
            wandb.log({'coef': np.corrcoef(our_curvature, gt_curvature)[0, 1]})
            wandb.log({'rmse': np.sqrt(np.mean((our_curvature - gt_curvature)**2))})
            wandb.log({'R2': r2_score(gt_curvature, our_curvature)})

    if cfg.logger.use_wandb:
        run.finish()

if __name__ == "__main__":
    main()