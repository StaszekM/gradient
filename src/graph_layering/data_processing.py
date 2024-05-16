import torch
from src.graph_layering.city_hetero_data import CityHeteroData

from abc import ABC, abstractmethod


class DataProcessor(ABC):
    @abstractmethod
    def transform_inplace(self, hetero_data_objs: "list[CityHeteroData]") -> None:
        pass

    @abstractmethod
    def fit(self, hetero_data_objs: "list[CityHeteroData]") -> None:
        pass


class NormalizationParams:
    hexes_std: torch.Tensor
    hexes_mean: torch.Tensor
    osmnx_nodes_std: torch.Tensor
    osmnx_nodes_mean: torch.Tensor
    layer_1_edges_std: torch.Tensor
    layer_1_edges_mean: torch.Tensor


class Normalizer(DataProcessor):
    def __init__(self) -> None:
        super().__init__()
        self._normalization_params = None

    def fit(self, hetero_data_objs: "list[CityHeteroData]"):
        hexes_X = torch.vstack([data.hex.x for data in hetero_data_objs])
        osmnx_nodes_X = torch.vstack([data.osmnx_node.x for data in hetero_data_objs])
        layer_1_edges_X = torch.vstack(
            [
                data.osmnx_node_connected_to_osmnx_node.edge_attr
                for data in hetero_data_objs
            ]
        )

        hexes_std, hexes_mean = torch.std_mean(hexes_X, dim=0)
        osmnx_nodes_std, osmnx_nodes_mean = torch.std_mean(osmnx_nodes_X, dim=0)
        layer_1_edges_std, layer_1_edges_mean = torch.std_mean(layer_1_edges_X, dim=0)

        self._normalization_params = NormalizationParams()

        self._normalization_params.hexes_std = hexes_std
        self._normalization_params.hexes_mean = hexes_mean
        self._normalization_params.osmnx_nodes_std = osmnx_nodes_std
        self._normalization_params.osmnx_nodes_mean = osmnx_nodes_mean
        self._normalization_params.layer_1_edges_std = layer_1_edges_std
        self._normalization_params.layer_1_edges_mean = layer_1_edges_mean

    def transform_inplace(self, hetero_data_objs: "list[CityHeteroData]"):
        if not self._normalization_params:
            raise ValueError("Normalization params are not fitted")
        hexes_std = self._normalization_params.hexes_std
        hexes_mean = self._normalization_params.hexes_mean
        osmnx_nodes_std = self._normalization_params.osmnx_nodes_std
        osmnx_nodes_mean = self._normalization_params.osmnx_nodes_mean
        layer_1_edges_std = self._normalization_params.layer_1_edges_std
        layer_1_edges_mean = self._normalization_params.layer_1_edges_mean

        for data in hetero_data_objs:
            data.hex.x = (data.hex.x - hexes_mean) / hexes_std
            data.osmnx_node.x = (data.osmnx_node.x - osmnx_nodes_mean) / osmnx_nodes_std
            data.osmnx_node_connected_to_osmnx_node.edge_attr = (
                data.osmnx_node_connected_to_osmnx_node.edge_attr - layer_1_edges_mean
            ) / layer_1_edges_std
