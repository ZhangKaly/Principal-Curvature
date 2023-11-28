import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from src.utils.path_transform import path_transform
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch
import os

class GeoDataPL(pl.LightningDataModule):
    def __init__(self, name, batch_size, data_dir, pre_transform, seed = 421, num_workers = 0):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

        if pre_transform is None:
            data_path = os.path.join(self.data_dir, "original")
        else:
            data_path = os.path.join(self.data_dir, "path_extension")

        self.ds = TUDataset(root = data_path, name = self.name, pre_transform = pre_transform)
        self.num_classes = len(torch.unique(self.ds._data.y))
        self.input_dim = self.ds._data.x.shape[1]

        self.seed = seed

    def setup(self, stage = None):

        train_idx, test_idx = train_test_split(list(range(len(self.ds))), test_size = 0.2, random_state = self.seed)
        train_idx, val_idx = train_test_split(list(range(len(self.ds))), test_size = 0.2, random_state = self.seed) 

        self.train_ds = Subset(self.ds,train_idx)
        self.val_ds = Subset(self.ds,val_idx)
        self.test_ds = Subset(self.ds,test_idx)



    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size = self.batch_size, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size = self.batch_size, num_workers = self.num_workers)

