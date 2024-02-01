from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GINConv
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
    
    
class GATModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=1)
        self.act1 = nn.ReLU()
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1)
        self.act2 = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = self.act1(self.conv1(x, edge_index, edge_weight))
        x = self.act2(self.conv2(x, edge_index, edge_weight))
        return x

class GNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        layer_name: str,
        num_layers: int = 2,
    ):
        super().__init__()
        
        assert num_layers >= 2, "num_layers must be >= 2"
        
        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(layer_name, in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(self._make_layer(layer_name, hidden_dim, hidden_dim))
        self.layers.append(self._make_layer(layer_name, hidden_dim, out_dim))
        
        self.act = nn.ReLU()
    
    @staticmethod
    def _make_layer(layer_name: str, in_dim: int, out_dim: int):
        if layer_name == "GCNConv":
            return GCNConv(in_dim, out_dim)
        elif layer_name == "GATConv":
            return GATConv(in_dim, out_dim)
        else:
            raise ValueError(f'Unknown layer type: {layer_name}. Select one from ["GCNConv", "GATConv", "GINConv"]')
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            x = torch.relu(x)
        return x