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
from typing import Callable


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SupervisedNodeClassificationGNN(pl.LightningModule):
    """    
    Lightning Module for Supervised Node Classification using a Graph Neural Network (GNN).
    
    Attributes:
        _gnn (nn.Module): The underlying Graph Neural Network model.
        _classification_head (nn.Sequential): Classification head for predicting node labels.
        _loss_fn (function): Loss function for training.
    """

    def __init__(self, gnn: nn.Module, emb_dim: int, num_classes: int, loss_fn: Callable = nn.NLLLoss):
        """
        Init SupervisedNodeClassificationGNN.

        Args:
            gnn (nn.Module): The Graph Neural Network model.
            emb_dim (int): Dimension of node embeddings.
            num_classes (int): Number of classes for node classification.
            loss_fn (Callable): Loss function (default: nn.NLLLoss).
        """
        super().__init__()

        self._gnn = gnn.to(device)

        self._classification_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(emb_dim, num_classes),
            nn.LogSoftmax(dim=1),
        ).to(device)

        self._loss_fn = loss_fn()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the GNN model.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph edge indices.
            edge_weight (torch.Tensor): Edge weights.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        return self._gnn(x, edge_index, edge_weight)

    def training_step(self, batch: List[Data], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (List[Data]): List of graph data batches.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Training loss.
        """
        data = batch[0]

        y_pred, y, auc = self._common_step(data=data, mask=data.train_mask)

        loss = self._loss_fn(y_pred, y)

        self.log("step", self.trainer.current_epoch)
        self.log("train/loss", loss.item(), on_epoch=True, on_step=False)
        self.log("train/auc", auc.item(), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: List[Data], batch_idx: int) -> dict:
        """
        Validation step.

        Args:
            batch (List[Data]): List of graph data batches.
            batch_idx (int): Index of the current batch.
        Returns:
            dict: Dictionary containing the computed AUC score for validation.
        """
        data = batch[0]

        _, _, auc = self._common_step(data=data, mask=data.val_mask)

        self.log("step", self.trainer.current_epoch)
        self.log("val/auc", auc.item(), on_epoch=True, on_step=False)

        return {"auc": auc}

    def test_step(self, batch: List[Data], batch_idx: int):
        """
        Test step.

        Args:
            batch (List[Data]): List of graph data batches.
            batch_idx (int): Index of the current batch.
        Returns:
            dict: Dictionary containing the computed AUC score for test.
        """
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
        """
        Prediction step.

        Args:
            batch (List[Data]): List of graph data batches.
            batch_idx (int): Index of the current batch.
            dataloader_idx (Optional[int]): Index of the dataloader (default: None).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted node embeddings and ground truth labels.
        """
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
        """
        Common step for training, validation, and test steps.

        Args:
            data (Data): Graph data.
            mask (torch.Tensor): Mask indicating the subset of nodes to consider.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]: Predicted labels, true labels, and AUC score.
        """
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
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer for training.
        """
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
        )

    def visualize_embeddings(z: torch.Tensor, y: torch.Tensor, n_components: int = 2) -> plt.figure:
        """
        Visualizes node embeddings using PCA, UMAP, and t-SNE.

        Args:
            z (torch.Tensor): Node embeddings.
            y (torch.Tensor): Ground truth labels.
            n_components (int): Number of components for dimensionality reduction (default: 2).
        fig (plt.Figure): Matplotlib figure containing the scatter plots with node embeddings visualizations.
        """
        
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
        sns.scatterplot(x=z_tsne[:, 0], y=z_tsne[:, 1], hue=y.cpu().numpy(), palette="Set2", ax=axs[2])
        axs[2].set(title="tsne")

        return fig