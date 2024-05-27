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
import json


with open("../data/osmnx_attributes.json") as f:
    all_attributes = json.load(f)


class OSMnxGraph:
    def __init__(self, gdf_events: gpd.GeoDataFrame, 
                 gdf_nodes: gpd.GeoDataFrame, 
                 gdf_edges: gpd.GeoDataFrame, 
                 start_acc_distance: float = 100,
                 y_column_name: str = 'accidents_count'
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
        if not any(col in self.gdf_nodes.columns for col in ['geometry', 'x', 'y', self.y_column_name, 'ref']):
            return self.gdf_nodes
        attrs = self.gdf_nodes.drop([ 'x', 'y', 'ref'], axis=1)
        attrs.replace('NaN', np.nan, inplace=True)
        attrs['highway'] = attrs['highway'].replace(np.nan, 'unknown')
        cols_to_add = set([item for item in all_attributes['nodes_highway'] if item not in set(attrs['highway'])])
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
        vectorized_feature = vectorizer.fit_transform(attrs['highway'])
        df_vect_feature = pd.DataFrame(vectorized_feature.toarray(), columns=vectorizer.get_feature_names_out())
        df_vect_feature.index = attrs.index
        cleaned_df = pd.merge(attrs, df_vect_feature, left_index=True, right_index=True)
        cleaned_df.fillna(0, inplace=True)
        cleaned_df.drop(['highway', 'unknown'], axis=1, inplace=True)
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
            """Method that converts speed from mph to km/h
            """
            return round(speed * 1.60934)
        
        def _check_max_speed(speed):
            """Method that checks if speed is in km/h and converts it if it is in mph
            """
            if 'mph' in speed:
                speed = speed.replace('mph', '')
                speed = int(speed)
                speed = _convert_mph_to_kmh(speed)
            return speed
        

        def _convert_to_meters(width):
            """Method that converts width from feet to meters
            """
            width = width.replace('ft', '') if 'ft' in width else width
            width = width.replace("\'", '.') if "\'" in width else width
            width = width.replace('"', '') if '"' in width else width
            width = float(width)
            return round(width * 0.3048, 2)
        
        features_groups = ['highway', 'access', 'junction', 'bridge', 'tunnel']
        attrs = self.gdf_edges
        for col in attrs.columns:
            attrs = attrs.explode(col)
        if not any(col in attrs.columns for col in ['highway','osmid', 'access', 'junction', 'bridge', 'tunnel']):
            return attrs
        else:
            attrs = attrs.drop(['ref', 'name'], axis=1)
            attrs.replace('NaN', np.nan, inplace=True)
            attrs['width'] = attrs['width'].astype(str).apply(_convert_to_meters) if convert_to_meters else attrs['width']
            attrs['width'] = pd.to_numeric(attrs['width'], errors='coerce')
            attrs['width'] = attrs['width'].fillna(round(attrs['width'].mean(), 2)).astype(float)
            attrs['length'] = round(attrs['length'],2).astype(float)
            attrs['lanes'] = attrs['lanes'].fillna(attrs['lanes'].astype(float).mean()).astype(int)
            attrs['maxspeed'] = attrs['maxspeed'].astype(str).apply(_check_max_speed)
            attrs['maxspeed'] = attrs["maxspeed"].replace('nan', np.nan)
            attrs['maxspeed'] = attrs["maxspeed"].fillna(round(attrs['maxspeed'].astype(float).mean()))
            attrs['maxspeed'] = attrs['maxspeed'].astype(int)
            attrs['reversed'] = attrs['reversed'].map({True: 1, False: 0}).astype(int)
            attrs['oneway'] = attrs['oneway'].map({True: 1, False: 0}).astype(int)
            attrs = attrs.fillna("unspecified")
            vect = CountVectorizer(tokenizer=lambda x: x.split())
            idx = attrs.index
            cleaned_df = attrs.copy()
            cleaned_df.reset_index(drop=True, inplace=True)
            cleaned_df = cleaned_df.drop(['highway','osmid', 'access', 'junction', 'bridge', 'tunnel'], axis=1)
            for col in features_groups:                  
                vectorized_feature = vect.fit_transform(attrs[col])
                df_feature_count = pd.DataFrame(vectorized_feature.toarray(), columns=vect.get_feature_names_out())
                df_feature_count = df_feature_count.reset_index(drop=True)
                columns_list = df_feature_count.columns.tolist()
                columns_to_add = [item for item in all_attributes[f"edges_{col}"] if item not in columns_list]
                df_feature_count = df_feature_count.rename(columns={col_nm: col + "_" + col_nm for col_nm in columns_list})
                for col_to_add in columns_to_add:
                    df_feature_count[col + "_" + col_to_add] = 0

                cleaned_df.reset_index(drop=True, inplace=True)
                df_feature_count.reset_index(drop=True, inplace=True)
                cleaned_df = pd.concat([cleaned_df, df_feature_count], axis=1)
            
            cleaned_df.set_index(idx, inplace=True) 
            cleaned_df.drop(['est_width'], axis=1, inplace=True) if 'est_width' in cleaned_df.columns else None
            cols_to_drop = [col for col in cleaned_df.columns if 'unspecified' in col]
            for col in cols_to_drop:
                if col in cleaned_df.columns:
                    cleaned_df = cleaned_df.drop(col, axis=1)
            self.gdf_edges = cleaned_df
            return cleaned_df
        
        
    def create_graph(self, 
                     element_type: Union[Literal["node"], Literal["edge"]], 
                     aggregation_method: Union[Literal["sum"], Literal["mean"], Literal["count"]],
                     normalize_y=True):
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
            pyg_graph.y = torch.tensor(self.gdf_nodes[self.y_column_name].values, dtype=torch.long)
            attrs = self.get_node_attrs()
        elif element_type == "edge":
            pyg_graph.y = torch.tensor(self.gdf_edges[self.y_column_name].values, dtype=torch.long)
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
    
    
    
    
    def _find_nearest_node(self, event_point, nodes):
        """Finds nearest node from given event point based on cdist

        Args:
            event_point (Point): event point
            nodes (pd.Series): pandas series with node geometry

        Returns:
            first_dist_osmid (int): osmid for node closest to event point
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
        
        event_point = np.array(event_point.xy).T[0]
        event_series = pd.Series([event_point] * len(nodes), index=nodes.index)
        sorted_nodes = nodes.combine(event_series, _calculate_distance).sort_values()
        first_dist_osmid = sorted_nodes.index[0]
        return first_dist_osmid
    
    def _find_nearest_edge(self, event_point, edges):
        """Finds nearest edge from given event point

        Args:
            event_point (Point): event point
            edges (pd.Series): edges geometry series
        """
        def _calculate_edge_distance(edge, event_point):
            return edge.distance(event_point)
        
        event_series = pd.Series([event_point] * len(edges), index=edges.index)
        sorted_edges = edges.combine(event_series, _calculate_edge_distance).sort_values()
        first_dist_osmid = sorted_edges.index[0]
        return first_dist_osmid
    
    def _aggregate(self, 
                   element_type: Union[Literal["node"], Literal["edge"]], 
                   aggregation_method: Union[Literal["sum"], Literal["mean"], Literal["count"]]):
        """Method to aggregate event to node or edge

        Args:
            element_type (Union[Literal[&quot;node&quot;], Literal[&quot;edge&quot;]]): aggregation type
            aggregation_method (Union[Literal[&quot;sum&quot;], Literal[&quot;mean&quot;], Literal[&quot;count&quot;]]): aggregation method

        """
        if element_type == "node":
            self.gdf_nodes['count'] = 0
            self.gdf_nodes['sum'] = 0
        elif element_type == "edge":
            self.gdf_edges['count'] = 0
            self.gdf_edges['sum'] = 0


        square_edge_length = self.start_acc_distance

        for _, event in self.gdf_events.iterrows():
            # Create square around event point
            event_point = event.geometry
            event_value = event[self.y_column_name] if aggregation_method != "count" and self.y_column_name in event else 1
            square_N, square_S, square_E, square_W = self._get_lat_lon_distance(event_point.y, event_point.x, square_edge_length/2)
            square = box(square_W, square_S,
                        square_E, square_N)
            
            if element_type == "node":
                # Check for nodes within the square
                nodes_or_edges_within_square = self.gdf_nodes[self.gdf_nodes.intersects(square)]
            elif element_type == "edge":
                # Check for edges within the square
                nodes_or_edges_within_square = self.gdf_edges[
                    self.gdf_edges.intersects(square)
                ]

            # If no nodes/edges found, increase square size and repeat
            while len(nodes_or_edges_within_square) == 0:
                square_edge_length += 100
                square_N, square_S, square_E, square_W = self._get_lat_lon_distance(event_point.y, event_point.x, square_edge_length/2)
                square = box(square_W, square_S,
                            square_E, square_N)
                
                # Check for nodes/edges within the square
                if element_type == "node":
                    nodes_or_edges_within_square = self.gdf_nodes[self.gdf_nodes.intersects(square)]
                elif element_type == "edge":
                    nodes_or_edges_within_square = self.gdf_edges[self.gdf_edges.intersects(square)]
            
            # Find nearest node/edge within the square and update y_column_name column in gdf_nodes/edges
            if element_type == "node":
                nearest_osmid = self._find_nearest_node(event_point,  nodes_or_edges_within_square.geometry.apply(lambda x: np.array(x.xy).T[0]))
                self.gdf_nodes.at[nearest_osmid, 'sum'] += event_value
                self.gdf_nodes.at[nearest_osmid, 'count'] += 1
                
            elif element_type == "edge":
                nearest_osmid = self._find_nearest_edge(event_point,  nodes_or_edges_within_square.geometry)
                self.gdf_edges.at[nearest_osmid, 'sum'] += event_value
                self.gdf_edges.at[nearest_osmid, 'count'] += 1

            # Reset square edge length for next iteration
            square_edge_length = self.start_acc_distance
        
        if aggregation_method == "mean":
            if element_type == "node":
                self.gdf_nodes[self.y_column_name] = self.gdf_nodes['sum'] / self.gdf_nodes['count']
            elif element_type == "edge":
                self.gdf_edges[self.y_column_name] = self.gdf_edges['sum'] / self.gdf_edges['count']

        self.gdf_nodes.rename(columns={aggregation_method: self.y_column_name}, inplace=True)
        self.gdf_edges.rename(columns={aggregation_method: self.y_column_name}, inplace=True)
        
        self.gdf_nodes.drop('count', axis=1, inplace=True) if 'count' in self.gdf_nodes.columns else None
        self.gdf_nodes.drop('sum', axis=1, inplace=True) if 'sum' in self.gdf_nodes.columns else None
        self.gdf_edges.drop('count', axis=1, inplace=True) if 'count' in self.gdf_edges.columns else None
        self.gdf_edges.drop('sum', axis=1, inplace=True) if 'sum' in self.gdf_edges.columns else None
            
    
    

        




