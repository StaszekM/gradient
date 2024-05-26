from torch import nn as nn
from torch_geometric import nn as gnn


class HeteroDictBatchNorm(nn.Module):
    def __init__(self, types, **kwargs):
        super().__init__()

        self.batch_norm = nn.ModuleDict({key: gnn.BatchNorm(**kwargs) for key in types})

    def forward(self, dictionary):
        return {key: self.batch_norm[key](dictionary[key]) for key in dictionary.keys()}
