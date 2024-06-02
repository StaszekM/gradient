from typing import Tuple, cast

import pandas as pd
from shapely import Point
import geopandas as gpd
import osmnx as ox
import osmnx.graph as ox_graph

from src.graph.create_osmnx_graph import OSMnxGraph
from src.organized_datasets_creation.utils import resolve_nominatim_city_name


def create_osmnx_dataframes(
    df_accidents: pd.DataFrame, nominatim_city_name: str
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Creates ready OSMNX nodes and edges GeoDataFrames with accidents data.

    Accidents will be spatially assigned to nearest OSMNX nodes and counted.
    After that, the accidents count will be binarized. If the accidents count is greater than 0, the value will be 1.

    Parameters
    ----------
    df_accidents : pd.DataFrame
        The dataframe with accidents data.
    nominatim_city_name : str
        The Nominatim city name for which the accidents data should be filtered. If this city is not in the data, the function
        will raise an AssertionError.

    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        The first element is a GeoDataFrame with nodes data. The second element is a GeoDataFrame with edges data.
        Nodes data contains the following columns:
            - index: autoincrement 0 to n, index name: node_id
            - osmid: OSMNX node ID
            - accidents_count: for now, binary value (1 if there are any accidents, 0 otherwise)
            - x: longitude
            - y: latitude
            - geometry: point geometry
        The remaining columns are attributes from the OSMNX nodes data.

        Edges data contains the following columns:
            - index: autoincrement 0 to n, index name: edge_id
            - u: OSMNX node ID of the first node of the edge
            - v: OSMNX node ID of the second node of the edge
            - key: edge key (for multiple edges between the same nodes)
            - geometry: linestring geometry
        The remaining columns are attributes from the OSMNX edges data.
    """

    city_name = resolve_nominatim_city_name(nominatim_city_name)

    assert (
        df_accidents["mie_nazwa"].isin([city_name]).any()
    ), f"City {nominatim_city_name} ({city_name}) is not in the data"

    df_accidents = df_accidents.drop(
        df_accidents[(df_accidents["mie_nazwa"] != city_name)].index,
    )
    df_accidents = df_accidents.drop(columns="uczestnicy")

    geometry = [
        Point(xy) for xy in zip(df_accidents["wsp_gps_x"], df_accidents["wsp_gps_y"])
    ]
    gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry=geometry)
    gdf_accidents = gdf_accidents.drop(columns=["wsp_gps_x", "wsp_gps_y"])
    G = ox_graph.graph_from_place(
        nominatim_city_name, network_type="drive", simplify=False
    )
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    assert gdf_edges.crs == gdf_nodes.crs
    osmnx_crs = gdf_edges.crs

    gdf_nodes_geometry_xy = gdf_nodes[["geometry", "x", "y"]]
    gdf_edges_geometry = gdf_edges["geometry"]

    osmnx_graph = OSMnxGraph(gdf_accidents, gdf_nodes, gdf_edges)
    osmnx_graph._aggregate_accidents(aggregation_type="node")
    accidents_binary_y = (osmnx_graph.gdf_nodes["accidents_count"] > 0).astype(int)
    osmnx_graph.create_graph(aggregation_type="node", normalize_y=True)

    osmnx_nodes = osmnx_graph.get_node_attrs()
    osmnx_nodes = osmnx_nodes.merge(
        left_index=True, right_index=True, right=accidents_binary_y
    ).merge(left_index=True, right_index=True, right=gdf_nodes_geometry_xy)
    osmnx_nodes = osmnx_nodes.reset_index(names="osmid").rename_axis("node_id", axis=0)
    osmnx_nodes = gpd.GeoDataFrame(osmnx_nodes, geometry="geometry", crs=osmnx_crs)

    osmnx_edges = osmnx_graph.get_edge_attrs()
    osmnx_edges = osmnx_edges.merge(
        left_index=True, right_index=True, right=gdf_edges_geometry
    )
    osmnx_edges = osmnx_edges.reset_index().rename_axis("edge_id", axis=0)
    osmnx_edges = gpd.GeoDataFrame(osmnx_edges, geometry="geometry", crs=osmnx_crs)

    return osmnx_nodes, osmnx_edges
