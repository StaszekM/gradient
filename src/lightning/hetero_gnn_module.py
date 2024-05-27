from typing import Any, List
import pytorch_lightning as pl

from src.baseline_models.HeteroGNN import HeteroGNN
import torch.optim as optim
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score


class HeteroGNNModule(pl.LightningModule):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_conv_layers: int,
        hetero_data: Any,
        lin_layer_sizes: List[int],
        add_batch_norm: bool,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()

        self.model = HeteroGNN(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_conv_layers=num_conv_layers,
            lin_layer_sizes=lin_layer_sizes,
            add_batch_norm=add_batch_norm,
            hetero_data=hetero_data,
        )
        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters(logger=True, ignore="hetero_data")

        self._initialize_lazy_modules(hetero_data)

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Any:
        return optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        loss = F.cross_entropy(y_hat["hex"], batch["hex"].y)
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, bath_idx):
        y_hat = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        val_loss = F.cross_entropy(y_hat["hex"], batch["hex"].y)
        self.log("val_loss", val_loss, batch_size=1, on_step=True)

    def test_step(self, batch, batch_idx):
        y_hat = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        loss = F.cross_entropy(y_hat["hex"], batch["hex"].y)
        y_hat = torch.softmax(y_hat["hex"], dim=-1)
        auc = roc_auc_score(
            batch["hex"].y.cpu().numpy(),
            y_hat[:, 1].cpu().numpy(),
            average="micro",
        )
        accuracy = (y_hat.argmax(dim=-1) == batch["hex"].y).sum().item() / len(
            batch["hex"].y
        )

        self.log("test_auc", auc, batch_size=1, on_epoch=True)
        self.log("test_loss", loss, batch_size=1, on_epoch=True)
        self.log("test_accuracy", accuracy, batch_size=1, on_epoch=True)

    def _initialize_lazy_modules(self, batch):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
