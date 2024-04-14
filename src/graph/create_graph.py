import geopandas as gpd
import torch_geometric
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt


class OSMEmbedderGraph:
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        labels_column_name: str,
        weights_type: str = "neighbours",
    ):
        self.weights = weights_type
        self.gdf = gdf
        self.node_labels = gdf[labels_column_name].to_numpy(dtype=np.int8)
        self.node_features = gdf.drop(
            ["geometry", labels_column_name], axis=1
        ).to_numpy(dtype=np.float32)
        self.graph_nx = None
        self.graph_data = None

    def createGraph_nx(self):
        assert (
            self.weights in ["neighbours", "shortest_path", "centroid"]
        ), f"Unknown weights_type: '{self.weights}'. Select one from ['neighbours', 'shortest_path', 'centroid']"

        graph = nx.Graph()

        for i in range(len(self.gdf)):
            graph.add_node(i)

        for i in range(len(self.gdf)):
            for j in range(i + 1, len(self.gdf)):
                node_i = self.gdf.iloc[i]
                node_j = self.gdf.iloc[j]
                if node_i["geometry"].touches(node_j["geometry"]):
                    graph.add_edge(i, j)

        if self.weights == "shortest_path":
            weighted_graph = graph.copy()
            for i in range(len(self.gdf)):
                for j in range(i + 1, len(self.gdf)):
                    node_i = self.gdf.iloc[i]
                    node_j = self.gdf.iloc[j]
                    if node_i["geometry"].touches(node_j["geometry"]):
                        weighted_graph[i][j]["weight"] = 1
                        pass
                    else:
                        try:
                            weight = nx.shortest_path_length(graph, i, j)
                            weighted_graph.add_edge(i, j, weight=1 / weight)
                        except nx.NetworkXNoPath:
                            continue
            self.graph_nx = weighted_graph
            return weighted_graph

        elif self.weights == "centroid":
            weighted_graph = graph.copy()
            for i in range(len(self.gdf)):
                for j in range(i + 1, len(self.gdf)):
                    node_i = self.gdf.iloc[i]
                    node_j = self.gdf.iloc[j]
                    distance = node_i["geometry"].centroid.distance(
                        node_j["geometry"].centroid
                    )
                    weight = 1 / (
                        distance + 1e-9
                    )  # Use a small epsilon to avoid division by zero
                    try:
                        weighted_graph[i][j]["weight"] = weight
                    except:
                        weighted_graph.add_edge(i, j, weight=weight)
            self.graph_nx = weighted_graph
            return weighted_graph

        else:
            self.graph_nx = graph
            return graph

    def createGraph_Data(self):
        if self.graph_nx == None:
            self.createGraph_nx()
        graph = torch_geometric.utils.from_networkx(self.graph_nx)
        x = torch.tensor(self.node_features, dtype=torch.float)
        y = torch.tensor(self.node_labels, dtype=torch.long)
        graph.x = x
        graph.y = y
        self.graph_data = graph
        return graph

    def graph_visualization(self):
        if self.graph_nx == None:
            self.createGraph_nx()
            
        fig, ax = plt.subplots()

        if self.weights == "centroid" or self.weights == "shortest_path":
            widths = list(nx.get_edge_attributes(self.graph_nx, "weight").values())
            max_width = max(widths)
            widths = [width / max_width for width in widths]
            nx.draw(self.graph_nx, node_size=30, node_color="plum", width=widths)
        else:
            nx.draw(self.graph_nx, node_size=30, node_color="plum", width=1)
            
        ax.set_title("Graph Visualization")
        return fig

    def show_statistics(self):
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

