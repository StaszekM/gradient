from torch import nn as nn
from torch_geometric import data as hd, nn as gnn

from .HeteroDictBatchNorm import HeteroDictBatchNorm


class HeteroGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_conv_layers: int,
        lin_layer_sizes: "list[int]",
        hetero_data: hd.HeteroData,
        add_batch_norm: bool = False,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        nodes, edges = hetero_data.metadata()

        self.has_batch_norm = add_batch_norm

        for _ in range(num_conv_layers):
            conv = gnn.HeteroConv(
                {
                    key: gnn.GATConv(
                        (-1, -1),
                        hidden_channels,
                        add_self_loops=key[0] == key[2],
                        edge_dim=hetero_data[key].num_features,
                    )
                    for key in edges
                },
                aggr="sum",
            )
            self.convs.append(conv)

        sizes: "list[int]" = [hidden_channels, *lin_layer_sizes, out_channels]

        self.lin_layers = gnn.Sequential(
            "dictionary",
            [
                (
                    gnn.HeteroDictLinear(i, j, types=nodes),
                    "dictionary -> dictionary",
                )
                for (i, j) in zip(sizes, sizes[1:])
            ],
        )

        if self.has_batch_norm:
            self.batch_norm = HeteroDictBatchNorm(
                in_channels=hidden_channels, types=nodes
            )

    def forward(self, x_dict, edge_index_dict, edge_atrr_dict):
        value = self._forward(x_dict, edge_index_dict, edge_atrr_dict)
        value = self.lin_layers(value)
        return value

    def get_embeddings(self, x_dict, edge_index_dict, edge_atrr_dict):
        value = self._forward(x_dict, edge_index_dict, edge_atrr_dict)

        layers = self.lin_layers.children()
        for layer in list(layers)[:-1]:
            value = layer(value)

        return value

    def _forward(self, x_dict, edge_index_dict, edge_atrr_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_atrr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if self.has_batch_norm:
            x_dict = self.batch_norm(x_dict)

        x = x_dict
        return x
