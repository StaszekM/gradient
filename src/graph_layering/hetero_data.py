from typing import Iterable
from torch_geometric.data import HeteroData
import torch

from src.graph_layering.graph_layer_creator import GraphLayerController, SourceType


def create_hetero_data(
    controller: GraphLayerController,
    hexes_attrs_columns_names: Iterable[str],
    osmnx_node_attrs_columns_names: Iterable[str],
    osmnx_edge_attrs_columns_names: Iterable[str],
    virtual_edge_attrs_columns_names: Iterable[str],
) -> HeteroData:
    data = HeteroData()
    edges_between_hexes = controller.get_edges_between_hexes()
    edges_between_source_and_hexes = controller.get_virtual_edges_to_hexes(
        SourceType.OSMNX_NODES
    )

    data["hex"].x = torch.tensor(
        controller.hexes_centroids_gdf[hexes_attrs_columns_names].to_numpy()
    )

    data["osmnx_node"].x = torch.tensor(
        controller.osmnx_nodes_gdf[osmnx_node_attrs_columns_names].to_numpy()
    )

    data["hex", "connected_to", "hex"].edge_index = torch.tensor(
        edges_between_hexes.merge(
            controller.hexes_gdf.reset_index(),
            left_on="u",
            right_on="h3_id",
        )
        .rename(columns={"region_id": "u_region_id"})
        .merge(
            controller.hexes_gdf.reset_index(),
            left_on="v",
            right_on="h3_id",
        )
        .rename(columns={"region_id": "v_region_id"})[["u_region_id", "v_region_id"]]
        .to_numpy()
        .T
    )

    node_to_node_connections = (
        controller.osmnx_edges_gdf.merge(
            controller.osmnx_nodes_gdf.reset_index(), left_on="u", right_on="osmid"
        )
        .rename(columns={"node_id": "u_node_id"})
        .merge(controller.osmnx_nodes_gdf.reset_index(), left_on="v", right_on="osmid")
        .rename(columns={"node_id": "v_node_id"})
    )

    data["osmnx_node", "connected_to", "osmnx_node"].edge_index = torch.tensor(
        node_to_node_connections[["u_node_id", "v_node_id"]].to_numpy().T
    )

    data["osmnx_node", "connected_to", "osmnx_node"].edge_attr = torch.tensor(
        node_to_node_connections[osmnx_edge_attrs_columns_names].to_numpy()
    )

    data["osmnx_node", "connected_to", "hex"].edge_index = torch.tensor(
        edges_between_source_and_hexes[["source_id", "region_id"]].to_numpy().T
    )

    data["osmnx_node", "connected_to", "hex"].edge_attr = torch.tensor(
        edges_between_source_and_hexes[virtual_edge_attrs_columns_names].to_numpy()
    )

    return data
