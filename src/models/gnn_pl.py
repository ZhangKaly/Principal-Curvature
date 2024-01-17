# PL GNN Model
import torch
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class GNNModel(pl.LightningModule):
    def __init__(self, input_dim, num_classes, lr = 0.001):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        hidden_dim = 16
        self.model = GCN(input_dim, hidden_dim)
        self.pooling_layer = torch_geometric.nn.pool.global_mean_pool
        self.loss = torch.nn.CrossEntropyLoss()
        self.classifier = torch.nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,num_classes))

        self.test_step_outputs = []   # save outputs in each batch to compute metric overall epoch
        self.test_step_targets = [] 

    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        x_ = self(batch)
        h = self.pooling_layer(x_, batch.batch)
        y_hat = self.classifier(h)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        x_ = self(batch)
        h = self.pooling_layer(x_, batch.batch)
        y_hat = self.classifier(h)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, edge_index, y = batch.x, batch.edge_index, batch.y
        x_ = self(batch)
        h = self.pooling_layer(x_, batch.batch)
        y_hat = self.classifier(h)
        loss = self.loss(y_hat, y)
        accuracy = torch.sum(y_hat.argmax(dim = 1) == y).item() / len(y)

        #self.log("test_accuracy", accuracy)
        self.log("test_loss", loss)

        self.test_step_outputs.extend(y_hat)
        self.test_step_targets.extend(y)

        return loss
    
    def on_test_epoch_end(self) -> None:
        test_all_outputs = self.test_step_outputs
        test_all_targets = self.test_step_targets

        y_pred = torch.stack(test_all_outputs, dim = 0)
        y_true = torch.stack(test_all_targets, dim = 0)

        accuracy = torch.sum(y_pred.argmax(dim = 1) == y_true).item() / len(y_true)
        
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # free up the memory
        # --> HERE STEP 3 <--
        self.test_step_outputs.clear()
        self.test_step_outputs.clear()
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def predict(self, x, edge_index):
        return self(x, edge_index)
    
