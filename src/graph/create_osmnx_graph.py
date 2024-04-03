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
    def __init__(self, gdf_accidents: gpd.GeoDataFrame, 
                 gdf_nodes: gpd.GeoDataFrame, 
                 gdf_edges: gpd.GeoDataFrame, 
                 start_acc_distance: float = 100):
        
        
        self.gdf_accidents = gdf_accidents
        self.gdf_nodes = gdf_nodes
        self.gdf_edges = gdf_edges
        self.start_acc_distance = start_acc_distance
        self.graph_nx = None
        self.graph_data = None


    
    def get_node_attrs(self):
        """
        Node features out of osmnx. Could be replaced with custom features
        """
        attrs = self.gdf_nodes.drop(['geometry', 'x', 'y', 'accidents_count', 'ref'], axis=1)
        attrs['highway'] = attrs['highway'].replace(0, 'unknown')
        vectorizer = CountVectorizer()

        # Fit and transform the text data
        X = vectorizer.fit_transform(attrs['highway'])
        df_count = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        df_count['index'] = attrs.index
        df_count.set_index("index", inplace = True)
        result_df = pd.merge(attrs, df_count, left_index=True, right_index=True)
        result_df.fillna(0, inplace=True)
        result_df.drop(['highway', 'unknown'], axis=1, inplace=True)
        return result_df
    
    
    def get_edge_attrs(self):
        """
        Edge attributes out of osmnx.
        """
        attrs = self.gdf_edges.drop(['geometry', 'accidents_count', 'ref', 'name'], axis=1)

        attrs['highway'] = attrs['highway'].replace(0, 'unknown')
        vectorizer = CountVectorizer()

        # Fit and transform the text data
        X = vectorizer.fit_transform(attrs['highway'])
        df_count = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        df_count['index'] = attrs.index
        df_count.set_index("index", inplace = True)
        result_df = pd.merge(attrs, df_count, left_index=True, right_index=True)
        result_df.fillna(0, inplace=True)
        result_df.drop(['highway', 'unknown'], axis=1, inplace=True)
        
        result_df.fillna('unknown', inplace=True)
        return result_df
        
        
    def create_graph(self, aggregation_type, normalize_y=True):
        self._aggregate_accidents(aggregation_type)
        self.gdf_nodes.fillna(0, inplace=True)
        self.gdf_edges.fillna(0, inplace=True)
        self.graph_nx = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)
        pyg_graph = from_networkx(self.graph_nx)
        # x and y are node attrs to assign edge attributes use 'pyg_graph.edge_attr'
        if aggregation_type == "node":
            pyg_graph.y = torch.tensor(self.gdf_nodes['accidents_count'].values, dtype=torch.long)
            attrs = self.get_node_attrs()
        elif aggregation_type == "edge":
            pyg_graph.y = torch.tensor(self.gdf_edges['accidents_count'].values, dtype=torch.long)
            attrs = self.get_edge_attrs()
        if normalize_y:
                pyg_graph.y = torch.where(pyg_graph.y > 0, torch.ones_like(pyg_graph.y), torch.zeros_like(pyg_graph.y))
        pyg_graph.x = torch.tensor(attrs.values, dtype=torch.float32)
        self.graph_data = pyg_graph
        return pyg_graph
        
    
    def show_statistics(self):
        data = self.graph_data
        max_edges = data.num_nodes * (data.num_nodes- 1) if data.is_directed() else data.num_nodes * (data.num_nodes - 1) // 2
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
        """ 
        Converts meters to degrees for bbox
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
    
    def _calculate_distance(self, coord1, coord2):
        return cdist([coord1], [coord2])[0, 0]
    
    
    def _find_nearest_node(self, accident_point, nodes):
        accident_point = np.array(accident_point.xy).T[0]
        accident_series = pd.Series([accident_point] * len(nodes), index=nodes.index)
        sorted_nodes = nodes.combine(accident_series, self._calculate_distance).sort_values()
        first_dist_osmid = sorted_nodes.index[0]
        return first_dist_osmid
    
    def _find_nearest_edge(self, accident_point, edges):
        def _calculate_edge_distance(edge, accident_point):
            return edge.distance(accident_point)
        
        accident_series = pd.Series([accident_point] * len(edges), index=edges.index)
        sorted_edges = edges.combine(accident_series, _calculate_edge_distance).sort_values()
        first_dist_osmid = sorted_edges.index[0]
        # distances = {}
        # print(type(edges))
        # print(edges.iloc[0])
        # for id, edge in zip(edges.index, edges):
        #     print(type(edge))
        #     print(edge)
        #     distance = edge.distance(accident_point)
        #     distances[id] = distance
        # sorted_dict = dict(sorted(distances.items(), key=lambda item: item[1]))
        # first_element = next(iter(sorted_dict.items()))
        return first_dist_osmid
    
    def _aggregate_accidents(self, aggregation_type: Union[Literal["node"], Literal["edge"]]):
        """
        Aggregate accidents to node or egde.
        """
        if aggregation_type == "node":
            self.gdf_nodes['accidents_count'] = 0
        elif aggregation_type == "edge":
            self.gdf_edges['accidents_count'] = 0
        else:
            raise ValueError("Invalid aggregation_type. Choose either 'node' or 'edge'.")
    
        square_edge_length = self.start_acc_distance

        # Iterate through accidents
        for _, accident in self.gdf_accidents.iterrows():
            # Create square around accident point
            accident_point = accident.geometry
            square_N, square_S, square_E, square_W = self._get_lat_lon_distance(accident_point.y, accident_point.x, square_edge_length/2)
            square = box(square_W, square_S,
                        square_E, square_N)
            
            if aggregation_type == "node":
                # Check for nodes within the square
                nodes_or_edges_within_square = self.gdf_nodes[self.gdf_nodes.intersects(square)]
            elif aggregation_type == "edge":
                # Check for edges within the square
                nodes_or_edges_within_square = self.gdf_edges[self.gdf_edges.intersects(square)]

            
            # If no nodes/edges found, increase square size and repeat
            while len(nodes_or_edges_within_square) == 0:
                square_edge_length += 100
                square_N, square_S, square_E, square_W = self._get_lat_lon_distance(accident_point.y, accident_point.x, square_edge_length/2)
                square = box(square_W, square_S,
                            square_E, square_N)
                
                # Check for nodes/edges within the square
                if aggregation_type == "node":
                    nodes_or_edges_within_square = self.gdf_nodes[self.gdf_nodes.intersects(square)]
                elif aggregation_type == "edge":
                    nodes_or_edges_within_square = self.gdf_edges[self.gdf_edges.intersects(square)]
            
            # Find nearest node within the square and update accidents_count column in gdf_nodes/edges
            if aggregation_type == "node":
                nearest_osmid = self._find_nearest_node(accident_point,  nodes_or_edges_within_square.geometry.apply(lambda x: np.array(x.xy).T[0]))
                self.gdf_nodes.at[nearest_osmid, 'accidents_count'] += 1
            elif aggregation_type == "edge":
                nearest_osmid = self._find_nearest_edge(accident_point,  nodes_or_edges_within_square.geometry)
                self.gdf_edges.at[nearest_osmid, 'accidents_count'] += 1

        # Reset square edge length for next iteration
            square_edge_length = self.start_acc_distance
    
    
    


