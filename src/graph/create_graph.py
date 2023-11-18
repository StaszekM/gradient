import geopandas as gpd
import torch_geometric
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt

class OSMEmbedderGraph:
    def __init__(self, gdf: gpd.GeoDataFrame, labels_column_name:str, weights:str):
        self.weights = weights
        self.gdf = gdf
        self.node_labels = gdf[labels_column_name].to_numpy(dtype=np.int8)
        self.node_features = gdf.drop(['geometry', labels_column_name], axis=1).to_numpy(dtype=np.float32)
        self.graph_nx = None
        
    def createGraph_nx(self):
        graph = nx.Graph()
        
        for i in range(len(self.gdf)):
            graph.add_node(i)

        for i in range(len(self.gdf)):
            for j in range(i + 1, len(self.gdf)):
                node_i = self.gdf.iloc[i]
                node_j = self.gdf.iloc[j]
                if node_i['geometry'].touches(node_j['geometry']):
                    graph.add_edge(i, j)
        
        if self.weights=='shortest_path':
            weighted_graph = graph.copy()
            for i in range(len(self.gdf)):
                for j in range(i + 1, len(self.gdf)):
                    node_i = self.gdf.iloc[i]
                    node_j = self.gdf.iloc[j]
                    if node_i['geometry'].touches(node_j['geometry']):
                        weighted_graph[i][j]['weight'] = 1
                        pass
                    else:
                        try:
                            weight = nx.shortest_path_length(graph, i, j)
                            weighted_graph.add_edge(i, j, weight=1/weight)
                        except nx.NetworkXNoPath:
                            continue
            return weighted_graph
        
        elif self.weights=='centroid':
            weighted_graph = graph.copy()
            for i in range(len(self.gdf)):
                for j in range(i + 1, len(self.gdf)):
                    node_i = self.gdf.iloc[i]
                    node_j = self.gdf.iloc[j]
                    distance = node_i['geometry'].centroid.distance(node_j['geometry'].centroid)
                    weight = 1 / (distance + 1e-9)  # Use a small epsilon to avoid division by zero
                    weighted_graph[i][j]['weight'] = weight
            return weighted_graph  
        
        else:
            return graph
    
    def createGraph_Data(self):
        graph_nx = self.createGraph_nx()
        graph = torch_geometric.utils.from_networkx(graph_nx)
        x = torch.tensor(self.node_features, dtype=torch.float)
        y = torch.tensor(self.node_labels , dtype=torch.long)
        graph.x = x
        graph.y = y
        return graph
    
    def graph_visualization(self):
        if self.graph_nx==None:
            self.graph_nx==self.createGraph_nx()
        weights=list(nx.get_edge_attributes(self.nx_graph, 'weight').values())
        nx.draw(self.nx_graph, node_size=30, node_color='plum', width=weights)
        plt.show()




