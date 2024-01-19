from torch import nn
from torch_geometric.nn import GCNConv
import torch


class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) Model.
    
    Description:
        This class defines a simple Graph Convolutional Network (GCN) model using PyTorch Geometric. The model consists of
        two graph convolutional layers with ReLU activation functions.
    Attributes:
        conv1: The first GCN layer.
        act1: ReLU activation after the first GCN layer.
        conv2: The second GCN layer.
        act2: ReLU activation after the second GCN layer.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        """
        Init GCNModel.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Dimension of the hidden layer.
            out_dim (int): Output dimension.
        """
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.act2 = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward method of the GCN model.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph edge indices.
            edge_weight (torch.Tensor): Edge weights.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        x = self.act1(self.conv1(x, edge_index, edge_weight))
        x = self.act2(self.conv2(x, edge_index, edge_weight))
        return x