import torch
from torch_geometric.data import HeteroData


class EdgesDict:
    edge_index: torch.Tensor


class EdgesWithAttributesDict(EdgesDict):
    edge_attr: torch.Tensor


class NodesDict:
    x: torch.Tensor


class NodesWithAttributesDict(NodesDict):
    y: torch.Tensor


class CityHeteroData(HeteroData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def hex(self) -> NodesWithAttributesDict:
        return self["hex"]

    @property
    def osmnx_node(self) -> NodesDict:
        return self["osmnx_node"]

    @property
    def hex_connected_to_hex(self) -> EdgesDict:
        return self["hex", "connected_to", "hex"]

    @property
    def osmnx_node_connected_to_osmnx_node(self) -> EdgesWithAttributesDict:
        return self["osmnx_node", "connected_to", "osmnx_node"]

    @property
    def osmnx_node_connected_to_hex(self) -> EdgesWithAttributesDict:
        return self["osmnx_node", "connected_to", "hex"]
