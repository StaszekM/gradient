import geopandas as gpd
from torch_geometric.utils.convert import from_networkx
import numpy as np
import torch
import pandas as pd
import osmnx as ox
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import Union, Literal


class OSMnxGraph:
    def __init__(
        self,
        gdf_events: gpd.GeoDataFrame,
        gdf_nodes: gpd.GeoDataFrame,
        gdf_edges: gpd.GeoDataFrame,
        all_attributes: "dict[str, str]",
        start_acc_distance: float = 100,
        y_column_name: str = "accidents_count",
    ):

        self.gdf_events = gdf_events
        self.gdf_nodes = gdf_nodes
        self.gdf_edges = gdf_edges
        self.start_acc_distance = start_acc_distance
        self.graph_nx = None
        self.graph_data = None
        self.y_column_name = y_column_name
        self.all_attributes = all_attributes

    def get_node_attrs(self):
        """Method to extract node features from OSMNx nodes GeoDataFrame. In that case retrieves highway types and street_count
        for each node. Cleaning process included applying CountVectorizer to highway column that represents highway types. Also the
        "ref' column was removed.


        Returns:
            pd.DataFrame: cleaned dataframe that contains node features that consists of highway types and street count for each node.
        """
        if not any(
            col in self.gdf_nodes.columns
            for col in ["geometry", "x", "y", self.y_column_name, "ref"]
        ):
            return self.gdf_nodes
        attrs = self.gdf_nodes.drop(["x", "y", "ref"], axis=1)
        attrs.replace("NaN", np.nan, inplace=True)
        attrs["highway"] = attrs["highway"].replace(np.nan, "unknown")
        cols_to_add = set(
            [
                item
                for item in self.all_attributes["nodes_highway"]
                if item not in set(attrs["highway"])
            ]
        )
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
        vectorized_feature = vectorizer.fit_transform(attrs["highway"])
        df_vect_feature = pd.DataFrame(
            vectorized_feature.toarray(), columns=vectorizer.get_feature_names_out()
        )
        df_vect_feature.index = attrs.index
        cleaned_df = pd.merge(attrs, df_vect_feature, left_index=True, right_index=True)
        cleaned_df.fillna(0, inplace=True)
        cleaned_df.drop(["highway", "unknown"], axis=1, inplace=True)
        for col in cols_to_add:
            cleaned_df[col] = 0
        self.gdf_nodes = cleaned_df
        return cleaned_df

    def get_edge_attrs(self, convert_to_meters: bool = False):
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

        def _convert_mph_to_kmh(speed):
            """Method that converts speed from mph to km/h"""
            return round(speed * 1.60934)

        def _check_max_speed(speed):
            """Method that checks if speed is in km/h and converts it if it is in mph"""
            if "mph" in speed:
                speed = speed.replace("mph", "")
                speed = int(speed)
                speed = _convert_mph_to_kmh(speed)
            return speed

        def _convert_to_meters(width):
            """Method that converts width from feet to meters"""
            width = width.replace("ft", "") if "ft" in width else width
            width = width.replace("'", ".") if "'" in width else width
            width = width.replace('"', "") if '"' in width else width
            width = float(width)
            return round(width * 0.3048, 2)

        features_groups = ["highway", "access", "junction", "bridge", "tunnel"]
        attrs = self.gdf_edges
        for col in attrs.columns:
            attrs = attrs.explode(col)
        if not any(
            col in attrs.columns
            for col in ["highway", "osmid", "access", "junction", "bridge", "tunnel"]
        ):
            return attrs
        else:
            attrs = attrs.drop(["ref", "name"], axis=1)
            attrs.replace("NaN", np.nan, inplace=True)
            attrs["width"] = (
                attrs["width"].astype(str).apply(_convert_to_meters)
                if convert_to_meters
                else attrs["width"]
            )
            attrs["width"] = pd.to_numeric(attrs["width"], errors="coerce")
            attrs["width"] = (
                attrs["width"].fillna(round(attrs["width"].mean(), 2)).astype(float)
            )
            attrs["length"] = round(attrs["length"], 2).astype(float)
            attrs["lanes"] = (
                attrs["lanes"].fillna(attrs["lanes"].astype(float).mean()).astype(int)
            )
            attrs["maxspeed"] = attrs["maxspeed"].astype(str).apply(_check_max_speed)
            attrs["maxspeed"] = attrs["maxspeed"].replace("nan", np.nan)
            attrs["maxspeed"] = attrs["maxspeed"].fillna(
                round(attrs["maxspeed"].astype(float).mean())
            )
            attrs["maxspeed"] = attrs["maxspeed"].astype(int)
            attrs["reversed"] = attrs["reversed"].map({True: 1, False: 0}).astype(int)
            attrs["oneway"] = attrs["oneway"].map({True: 1, False: 0}).astype(int)
            attrs = attrs.fillna("unspecified")
            vect = CountVectorizer(tokenizer=lambda x: x.split())
            idx = attrs.index
            cleaned_df = attrs.copy()
            cleaned_df.reset_index(drop=True, inplace=True)
            cleaned_df = cleaned_df.drop(
                ["highway", "osmid", "access", "junction", "bridge", "tunnel"], axis=1
            )
            for col in features_groups:
                vectorized_feature = vect.fit_transform(attrs[col])
                df_feature_count = pd.DataFrame(
                    vectorized_feature.toarray(), columns=vect.get_feature_names_out()
                )
                df_feature_count = df_feature_count.reset_index(drop=True)
                columns_list = df_feature_count.columns.tolist()
                columns_to_add = [
                    item
                    for item in self.all_attributes[f"edges_{col}"]
                    if item not in columns_list
                ]
                df_feature_count = df_feature_count.rename(
                    columns={col_nm: col + "_" + col_nm for col_nm in columns_list}
                )
                for col_to_add in columns_to_add:
                    df_feature_count[col + "_" + col_to_add] = 0

                cleaned_df.reset_index(drop=True, inplace=True)
                df_feature_count.reset_index(drop=True, inplace=True)
                cleaned_df = pd.concat([cleaned_df, df_feature_count], axis=1)

            cleaned_df.set_index(idx, inplace=True)
            (
                cleaned_df.drop(["est_width"], axis=1, inplace=True)
                if "est_width" in cleaned_df.columns
                else None
            )
            cols_to_drop = [col for col in cleaned_df.columns if "unspecified" in col]
            for col in cols_to_drop:
                if col in cleaned_df.columns:
                    cleaned_df = cleaned_df.drop(col, axis=1)
            self.gdf_edges = cleaned_df
            return cleaned_df

    def create_graph(
        self,
        element_type: Union[Literal["node"], Literal["edge"]],
        aggregation_method: Union[Literal["sum"], Literal["mean"], Literal["count"]],
        normalize_y=True,
    ):
        """Method that creates graph from OSMNx geodataframes for nodes and edges.

        Args:
            element_type (Union[Literal["node"], Literal["edge"]]): node or edge aggregation type
            aggregation_method (Union[Literal["sum"], Literal["mean"], Literal["count"]): aggregation method
            normalize_y (bool, optional): If y should be treated as binary classification (if y greater than 0 it is 1
                                            and if 0 then 0). Defaults to True.

        Returns:
            pyg_graph (torch_geometric.data.Data): pytorch geometric graph
        """
        self._aggregate(element_type, aggregation_method)
        self.gdf_nodes.fillna(0, inplace=True)
        self.gdf_edges.fillna(0, inplace=True)
        self.graph_nx = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)
        pyg_graph = from_networkx(self.graph_nx)
        # x and y are node attrs to assign edge attributes use 'pyg_graph.edge_attr'
        if element_type == "node":
            pyg_graph.y = torch.tensor(
                self.gdf_nodes[self.y_column_name].values, dtype=torch.long
            )
            attrs = self.get_node_attrs()
        elif element_type == "edge":
            pyg_graph.y = torch.tensor(
                self.gdf_edges[self.y_column_name].values, dtype=torch.long
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

    def _aggregate(
        self,
        element_type: Union[Literal["node"], Literal["edge"]],
        aggregation_method: Union[Literal["sum"], Literal["mean"], Literal["count"]],
    ):
        """Method to aggregate event to node or edge

        Args:
            element_type (Union[Literal[&quot;node&quot;], Literal[&quot;edge&quot;]]): aggregation type
            aggregation_method (Union[Literal[&quot;sum&quot;], Literal[&quot;mean&quot;], Literal[&quot;count&quot;]]): aggregation method

        """
        if element_type == "node":
            elements = self.gdf_nodes
        elif element_type == "edge":
            elements = self.gdf_edges
        else:
            raise ValueError("element_type should be 'node' or 'edge'")

        closest_indexes = elements.geometry.sindex.nearest(
            self.gdf_events.geometry, return_all=False
        )[1]

        total_dict = {}
        total_dict[self.y_column_name] = (
            self.gdf_events[self.y_column_name]
            if self.y_column_name in self.gdf_events.columns
            else 1
        )
        total_dict["element_id"] = elements.iloc[closest_indexes].index
        total = pd.DataFrame(total_dict).set_index("element_id")

        aggregated = total.groupby("element_id")
        if aggregation_method == "count":
            aggregated = aggregated.count()
        elif aggregation_method == "sum":
            aggregated = aggregated.sum()
        elif aggregation_method == "mean":
            aggregated = aggregated.mean()
        else:
            raise ValueError("Unknown aggregation method")

        elements = elements.merge(
            aggregated, left_index=True, right_index=True, how="left"
        ).fillna(0)

        if element_type == "node":
            self.gdf_nodes = elements
        else:
            self.gdf_edges = elements
