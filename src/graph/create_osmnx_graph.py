import geopandas as gpd
from torch_geometric.utils.convert import from_networkx
import numpy as np
import torch
import pandas as pd
import osmnx as ox
import math
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from shapely.geometry import box
from scipy.spatial.distance import cdist
from typing import Union, Literal


class OSMnxGraph:
    def __init__(
        self,
        gdf_accidents: gpd.GeoDataFrame,
        gdf_nodes: gpd.GeoDataFrame,
        gdf_edges: gpd.GeoDataFrame,
        start_acc_distance: float = 100,
    ):
        self.gdf_accidents = gdf_accidents
        self.gdf_nodes = gdf_nodes
        self.gdf_edges = gdf_edges
        self.start_acc_distance = start_acc_distance
        self.graph_nx = None
        self.graph_data = None

    def get_node_attrs(self):
        """Method to extract node features from OSMNx nodes GeoDataFrame. In that case retrieves highway types and street_count
        for each node. Cleaning process included applying CountVectorizer to highway column that represents highway types. Also the
        "ref' column was removed.


        Returns:
            pd.DataFrame: cleaned dataframe that contains node features that consists of highway types and street count for each node.
        """

        if not any(
            col in self.gdf_nodes.columns
            for col in ["geometry", "x", "y", "accidents_count", "ref"]
        ):
            return self.gdf_nodes
        attrs = self.gdf_nodes.drop(
            ["geometry", "x", "y", "accidents_count", "ref"], axis=1, errors="ignore"
        )
        attrs["highway"] = attrs["highway"].replace(0, "unknown")
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
        vectorized_feature = vectorizer.fit_transform(attrs["highway"])
        df_vect_feature = pd.DataFrame(
            vectorized_feature.toarray(), columns=vectorizer.get_feature_names_out()
        )
        df_vect_feature["index"] = attrs.index
        df_vect_feature.set_index("index", inplace=True)
        cleaned_df = pd.merge(attrs, df_vect_feature, left_index=True, right_index=True)
        cleaned_df.fillna(0, inplace=True)
        cleaned_df.drop(["highway", "unknown"], axis=1, inplace=True)
        self.gdf_nodes = cleaned_df
        return cleaned_df

    def get_edge_attrs(self):
        """Method that gets edge attributes. Includes preprocessing:
        * lanes - add default value of 2
        * maxspeed - add default value of 50 as common speed limit in urban area
        * width - add default value of 2.0 as common road width
        * other columns - add default value of "unspecified"

        If column was numeric / bool and contained list the first element was taken. If column was string type the CountVectorizer
        was applied. Then columns that were vectorized was dropped and also "ref' column was removed.


        Returns:
            pd.DataFrame: cleaned DataFrame that contains edge features
        """

        def _get_first_element(lst):
            """Method that gets first element of list if lst is of type "list" """
            if isinstance(lst, list):
                return lst[0]
            else:
                return lst

        attrs = self.gdf_edges
        if not any(
            col in attrs.columns
            for col in [
                "highway",
                "osmid",
                "access",
                "junction",
                "bridge",
                "tunnel",
                "geometry",
            ]
        ):
            return attrs
        else:
            attrs.replace("NaN", np.nan, inplace=True)
            attrs["maxspeed"] = attrs["maxspeed"].fillna(50)
            attrs["width"] = (
                pd.to_numeric(attrs["width"], errors="coerce").fillna(2.0).astype(float)
            )
            attrs["lanes"] = attrs["lanes"].fillna(2)
            attrs = attrs.drop(["ref", "name"], axis=1)
            attrs = attrs.fillna("unspecified")
            attrs["lanes"] = (
                attrs["lanes"].apply(lambda x: _get_first_element(x)).astype(int)
            )
            attrs["reversed"] = attrs["reversed"].apply(lambda x: _get_first_element(x))
            attrs["maxspeed"] = (
                pd.to_numeric(
                    attrs["maxspeed"].apply(lambda x: _get_first_element(x)),
                    errors="coerce",
                )
                .fillna(50)
                .astype(int)
            )
            attrs["reversed"] = attrs["reversed"].map({True: 1, False: 0}).astype(int)
            attrs["oneway"] = attrs["oneway"].map({True: 1, False: 0}).astype(int)
            vect = CountVectorizer(tokenizer=lambda x: x.split())
            cleaned_df = attrs
            for col in ["highway", "access", "junction", "bridge", "tunnel"]:
                attrs[col] = attrs[col].apply(
                    lambda x: " ".join(x) if isinstance(x, list) else str(x)
                )
                vectorized_feature = vect.fit_transform(attrs[col])
                df_feature_count = pd.DataFrame(
                    vectorized_feature.toarray(), columns=vect.get_feature_names_out()
                )
                df_feature_count["index"] = attrs.index
                df_feature_count.set_index("index", inplace=True)
                new_index_tuples = [(u, v, x) for u, v, x in attrs.index]
                new_index = pd.MultiIndex.from_tuples(
                    new_index_tuples, names=["u", "v", "key"]
                )
                df_feature_count.index = new_index
                df_feature_count = df_feature_count.rename(
                    columns={
                        col_nm: col + "_" + col_nm
                        for col_nm in df_feature_count.columns
                    }
                )
                cleaned_df = pd.merge(
                    cleaned_df, df_feature_count, left_index=True, right_index=True
                )
            cleaned_df = cleaned_df.drop(
                [
                    "highway",
                    "osmid",
                    "access",
                    "junction",
                    "bridge",
                    "tunnel",
                    "geometry",
                ],
                axis=1,
            )
            self.gdf_edges = cleaned_df
            return cleaned_df

    def create_graph(
        self,
        aggregation_type: Union[Literal["node"], Literal["edge"]],
        normalize_y=True,
    ):
        """Method that creates graph from OSMNx geodataframes for nodes and edges.

        Args:
            aggregation_type (Union[Literal["node"], Literal["edge"]]): node or edge aggregation type
            normalize_y (bool, optional): If y should be treated as binary classification (if y greater than 0 it is 1
                                            and if 0 then 0). Defaults to True.

        Returns:
            pyg_graph (torch_geometric.data.Data): pytorch geometric graph
        """
        self._aggregate_accidents(aggregation_type)
        self.gdf_nodes.fillna(0, inplace=True)
        self.gdf_edges.fillna(0, inplace=True)
        self.graph_nx = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)
        pyg_graph = from_networkx(self.graph_nx)
        # x and y are node attrs to assign edge attributes use 'pyg_graph.edge_attr'
        if aggregation_type == "node":
            pyg_graph.y = torch.tensor(
                self.gdf_nodes["accidents_count"].values, dtype=torch.long
            )
            attrs = self.get_node_attrs()
        elif aggregation_type == "edge":
            pyg_graph.y = torch.tensor(
                self.gdf_edges["accidents_count"].values, dtype=torch.long
            )
            attrs = self.get_edge_attrs()
        if normalize_y:
            pyg_graph.y = torch.where(
                pyg_graph.y > 0,
                torch.ones_like(pyg_graph.y),
                torch.zeros_like(pyg_graph.y),
            )
        pyg_graph.x = torch.tensor(attrs.values, dtype=torch.float32)
        self.graph_data = pyg_graph
        return pyg_graph

    def show_statistics(self):
        """Shows graph statistics

        Returns:
            dict: graph statistics that includes:
                    * Nodes - nodes count
                    * Edges - edges count
                    * Nodes class - number of nodes class
                    * Directed - whether graph is directed
                    * Graph density - graph density in percent
        """
        data = self.graph_data
        max_edges = (
            data.num_nodes * (data.num_nodes - 1)
            if data.is_directed()
            else data.num_nodes * (data.num_nodes - 1) // 2
        )
        edges = data.num_edges if data.is_directed() else data.num_edges // 2
        return {
            "Nodes": data.num_nodes,
            "Edges": edges,
            "Nodes dim": data.num_node_features,
            "Nodes class": torch.unique(data.y).size(0),
            "Directed": data.is_directed(),
            "Graph density [%]": round((edges / (max_edges) * 100), 3),
        }

    def _get_lat_lon_distance(self, lat, lon, meters):
        """Calculates the coordinates from given latitude and longitude and distance in meters in 4 directions to make a square.

        Args:
            lat (float): latitude
            lon (float): longitude
            meters (float): distance in meters

        Returns:
            NSWE points (floats) : coordinates North, South, West and East from given point in given distance
        """
        earth = 6378.137  # Radius of the Earth in kilometers
        pi = math.pi

        # Conversion factor from meters to degrees
        m = (1 / ((2 * pi / 360) * earth)) / 1000  # 1 meter in degree

        new_latitude_N = lat + (meters * m)
        new_latitude_S = lat - (meters * m)
        new_longitude_E = lon + (meters * m) / math.cos(lat * (pi / 180))
        new_longitude_W = lon - (meters * m) / math.cos(lat * (pi / 180))

        return new_latitude_N, new_latitude_S, new_longitude_W, new_longitude_E

    def _find_nearest_node(self, accident_point, nodes):
        """Finds nearest node from given accident point based on cdist

        Args:
            accident_point (Point): accident point
            nodes (pd.Series): pandas series with node geometry

        Returns:
            first_dist_osmid (int): osmid for node closest to accident point
        """

        def _calculate_distance(coord1, coord2):
            """Calculates cdist between 2 coordinates

            Args:
                coord1 (Point): first coord
                coord2 (Point): second coord

            Returns:
                distance: distance between coordinates
            """
            return cdist([coord1], [coord2])[0, 0]

        accident_point = np.array(accident_point.xy).T[0]
        accident_series = pd.Series([accident_point] * len(nodes), index=nodes.index)
        sorted_nodes = nodes.combine(accident_series, _calculate_distance).sort_values()
        first_dist_osmid = sorted_nodes.index[0]
        return first_dist_osmid

    def _find_nearest_edge(self, accident_point, edges):
        """Finds nearest edge from given accident point

        Args:
            accident_point (Point): accident point
            edges (pd.Series): edges geometry series
        """

        def _calculate_edge_distance(edge, accident_point):
            return edge.distance(accident_point)

        accident_series = pd.Series([accident_point] * len(edges), index=edges.index)
        sorted_edges = edges.combine(
            accident_series, _calculate_edge_distance
        ).sort_values()
        first_dist_osmid = sorted_edges.index[0]
        return first_dist_osmid

    def _aggregate_accidents(
        self, aggregation_type: Union[Literal["node"], Literal["edge"]]
    ):
        """Method to aggregate accidents to node or edge

        Args:
            aggregation_type (Union[Literal[&quot;node&quot;], Literal[&quot;edge&quot;]]): aggregation type

        """
        if aggregation_type == "node":
            self.gdf_nodes["accidents_count"] = 0
        elif aggregation_type == "edge":
            self.gdf_edges["accidents_count"] = 0

        square_edge_length = self.start_acc_distance

        for _, accident in self.gdf_accidents.iterrows():
            # Create square around accident point
            accident_point = accident.geometry
            square_N, square_S, square_E, square_W = self._get_lat_lon_distance(
                accident_point.y, accident_point.x, square_edge_length / 2
            )
            square = box(square_W, square_S, square_E, square_N)

            if aggregation_type == "node":
                # Check for nodes within the square
                nodes_or_edges_within_square = self.gdf_nodes[
                    self.gdf_nodes.intersects(square)
                ]
            elif aggregation_type == "edge":
                # Check for edges within the square
                nodes_or_edges_within_square = self.gdf_edges[
                    self.gdf_edges.intersects(square)
                ]

            # If no nodes/edges found, increase square size and repeat
            while len(nodes_or_edges_within_square) == 0:
                square_edge_length += 100
                square_N, square_S, square_E, square_W = self._get_lat_lon_distance(
                    accident_point.y, accident_point.x, square_edge_length / 2
                )
                square = box(square_W, square_S, square_E, square_N)

                # Check for nodes/edges within the square
                if aggregation_type == "node":
                    nodes_or_edges_within_square = self.gdf_nodes[
                        self.gdf_nodes.intersects(square)
                    ]
                elif aggregation_type == "edge":
                    nodes_or_edges_within_square = self.gdf_edges[
                        self.gdf_edges.intersects(square)
                    ]

            # Find nearest node within the square and update accidents_count column in gdf_nodes/edges
            if aggregation_type == "node":
                nearest_osmid = self._find_nearest_node(
                    accident_point,
                    nodes_or_edges_within_square.geometry.apply(
                        lambda x: np.array(x.xy).T[0]
                    ),
                )
                self.gdf_nodes.at[nearest_osmid, "accidents_count"] += 1
            elif aggregation_type == "edge":
                nearest_osmid = self._find_nearest_edge(
                    accident_point, nodes_or_edges_within_square.geometry
                )
                self.gdf_edges.at[nearest_osmid, "accidents_count"] += 1

            # Reset square edge length for next iteration
            square_edge_length = self.start_acc_distance
