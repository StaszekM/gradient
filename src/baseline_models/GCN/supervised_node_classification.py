import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch_geometric.data import Data
from typing import List, Optional, Tuple
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SupervisedNodeClassificationGNN(pl.LightningModule):
    """Supervised node classification for a given GNN model."""

    def __init__(self, gnn: nn.Module, emb_dim: int, num_classes: int):
        super().__init__()

        self._gnn = gnn.to(device)

        self._classification_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(emb_dim, num_classes),
            nn.LogSoftmax(dim=1),
        ).to(device)

        self._loss_fn = nn.NLLLoss()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self._gnn(x, edge_index, edge_weight)

    def training_step(self, batch: List[Data], batch_idx: int) -> torch.Tensor:
        data = batch[0]

        y_pred, y, auc = self._common_step(data=data, mask=data.train_mask)

        loss = self._loss_fn(input=y_pred, target=y)

        self.log("step", self.trainer.current_epoch)
        self.log("train/loss", loss.item(), on_epoch=True, on_step=False)
        self.log("train/auc", auc.item(), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: List[Data], batch_idx: int):
        data = batch[0]

        _, _, auc = self._common_step(data=data, mask=data.val_mask)

        self.log("step", self.trainer.current_epoch)
        self.log("val/auc", auc.item(), on_epoch=True, on_step=False)

        return {"auc": auc}

    def test_step(self, batch: List[Data], batch_idx: int):
        data = batch[0]

        _, _, auc = self._common_step(data=data, mask=data.test_mask)

        self.log("step", self.trainer.current_epoch)
        self.log("test/auc", auc.item(), on_epoch=True, on_step=False)

        return {"auc": auc}

    def predict_step(
        self,
        batch: List[Data],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = batch[0]
        try:
            z = self(data.x, data.edge_index, data.weight)
        except:
            z = self(data.x, data.edge_index, None)
        y = data.y

        return z, y

    def _common_step(
        self,
        data: Data,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        try:
            z = self(data.x, data.edge_index, data.weight)
        except:
            z = self(data.x, data.edge_index, None)

        y_pred = self._classification_head(z)[mask]
        y = data.y[mask]

        auc = roc_auc_score(
            y_true=y.detach().cpu().numpy(),
            y_score=y_pred.exp().detach().cpu().numpy(),
            multi_class="ovr",
        )

        return y_pred, y, auc

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
        )

    def visualize_embeddings(z: torch.Tensor, y: torch.Tensor, n_components: int = 2):
        
        z = z.to(device)
        y = y.to(device)

        z_PCA = PCA(n_components=n_components).fit_transform(z.cpu().numpy())
        z_UMAP = umap.UMAP(n_components=n_components).fit_transform(z.cpu().numpy())
        tsne = TSNE(n_components=n_components, n_iter=500)

        z_tsne = tsne.fit_transform(z.cpu().numpy(), y.cpu().numpy())

        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
        sns.scatterplot(x=z_PCA[:, 0], y=z_PCA[:, 1], hue=y.cpu().numpy(), palette="Set2", ax=axs[0])
        axs[0].set(title="PCA")
        sns.scatterplot(x=z_UMAP[:, 0], y=z_UMAP[:, 1], hue=y.cpu().numpy(), palette="Set2", ax=axs[1])
        axs[1].set(title="UMAP")
        sns.scatterplot(x=z_tsne[:, 0], y=z_tsne[:, 1], hue=y.cpu().numpy(), palette="Set3", ax=axs[2])
        axs[1].set(title="tsne")

        plt.show()