from typing import Callable, Dict, Iterable, List, Set, Tuple, cast
import geopandas as gpd
from srai.neighbourhoods import H3Neighbourhood
from enum import Enum
import h3
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry


class SourceType(Enum):
    OSMNX_NODES = "osmnx_nodes"
    OSMNX_EDGES = "osmnx_edges"


class GraphLayerController:

    def __init__(
        self,
        hexes_gdf: gpd.GeoDataFrame,
        osmnx_nodes_gdf: gpd.GeoDataFrame,
        osmnx_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        self._assert_constructor_args(hexes_gdf, osmnx_nodes_gdf, osmnx_edges_gdf)

        self.hexes_gdf = hexes_gdf
        self.osmnx_nodes_gdf = osmnx_nodes_gdf
        self.osmnx_edges_gdf = osmnx_edges_gdf

        self._h3_neighbourhood = H3Neighbourhood(self.hexes_gdf)
        self._hexes_centroids_gdf = gpd.GeoDataFrame(
            hexes_gdf.centroid, columns=["centroid_geometry"]
        )

        self._virtual_edges_dfs_cache: Dict[SourceType, gpd.GeoDataFrame] = dict()
        self._edges_between_hexes_cache: Dict[int, pd.DataFrame] = dict()
        self._hex_resolution = self._get_unique_resolutions(hexes_gdf).item()
        self._virtual_linestrings_cache: Dict[SourceType, gpd.GeoDataFrame] = dict()

    def get_virtual_edges_to_hexes(self, source: SourceType) -> gpd.GeoDataFrame:
        if source in self._virtual_edges_dfs_cache:
            return self._virtual_edges_dfs_cache[source]

        if source == SourceType.OSMNX_NODES:
            source_gdf = self.osmnx_nodes_gdf
            predicate = "contains"
        elif source == SourceType.OSMNX_EDGES:
            source_gdf = self.osmnx_edges_gdf
            predicate = "intersects"
        else:
            raise ValueError(
                f"Unknown source: {source}, expected SourceType.OSMNX_NODES or SourceType.OSMNX_EDGES"
            )

        just_geo_gdf: gpd.GeoDataFrame = cast(
            gpd.GeoDataFrame, self.hexes_gdf[["geometry"]]
        )

        just_geo_source_gdf = source_gdf[["geometry"]]

        result = (
            just_geo_gdf.sjoin(just_geo_source_gdf, predicate=predicate)
            .iloc[:, [-1]]
            .rename(columns=dict(index_right="source_id"))
            .reset_index()
        )

        self._virtual_edges_dfs_cache[source] = result

        return result

    def get_edges_between_hexes(self, k_distance: int = 1) -> pd.DataFrame:
        if k_distance in self._edges_between_hexes_cache:
            return self._edges_between_hexes_cache[k_distance]

        v_finder: Callable[[str], Set[str]] = (
            lambda hex: self._h3_neighbourhood.get_neighbours_at_distance(
                hex, k_distance
            )
        )

        result = (
            pd.DataFrame(
                {
                    "u": self.hexes_gdf.index,
                    "v": self.hexes_gdf.index.map(v_finder),
                },
            )
            .explode(column="v")
            .reset_index(drop=True)
        )

        hex_id_int = result["u"].apply(int, base=16).astype(pd.Int64Dtype())
        neighbour_int = result["v"].apply(int, base=16).astype(pd.Int64Dtype())

        sorter: Callable[[Iterable[int]], Tuple[int, ...]] = lambda x: tuple(sorted(x))

        series = pd.Series(cast(List[int], zip(hex_id_int, neighbour_int))).apply(
            sorter
        )
        result["edge_id"] = series

        result = result.drop_duplicates(subset="edge_id").drop(columns="edge_id")

        self._edges_between_hexes_cache[k_distance] = result

        return result

    # adds linestrings between sources and hexes centroids based on virtual edges (for not it assumes sources are points)
    def get_virtual_linestrings(self, source: SourceType) -> gpd.GeoDataFrame:
        if source in self._virtual_linestrings_cache:
            return self._virtual_linestrings_cache[source]

        linestring_creator: Callable[[Dict[str, BaseGeometry]], LineString]

        if source == SourceType.OSMNX_NODES:
            source_gdf = self.osmnx_nodes_gdf
            linestring_creator = lambda x: cast(
                LineString, LineString([x["geometry"], x["centroid_geometry"]])
            )
        elif source == SourceType.OSMNX_EDGES:
            source_gdf = self.osmnx_edges_gdf
            linestring_creator = lambda x: cast(
                LineString,
                LineString(
                    [
                        x["geometry"].line_interpolate_point(0.5, normalized=True),
                        x["centroid_geometry"],
                    ]
                ),
            )
        else:
            raise ValueError(
                f"Unknown source: {source}, expected SourceType.OSMNX_NODES or SourceType.OSMNX_EDGES"
            )

        virtual_edges_df = self.get_virtual_edges_to_hexes(source=source)

        merged = virtual_edges_df.merge(
            self._hexes_centroids_gdf, left_on="region_id", right_index=True
        ).merge(source_gdf[["geometry"]], left_on="source_id", right_index=True)

        virtual_edge_geometry = merged.apply(linestring_creator, axis=1)
        merged["virtual_edge"] = gpd.GeoSeries(virtual_edge_geometry)

        result = gpd.GeoDataFrame(
            merged[["region_id", "source_id", "virtual_edge"]],
            geometry="virtual_edge",
            crs=self.hexes_gdf.crs,
        )  # type: ignore

        self._virtual_linestrings_cache[source] = result

        return result

    def get_virtual_linestrings_between_centroids(self, k_distance: int = 1):
        hexes_edges_gdf = self.get_edges_between_hexes(k_distance=k_distance)
        merged = (
            hexes_edges_gdf.merge(
                self._hexes_centroids_gdf, left_on="u", right_index=True
            )
            .merge(self._hexes_centroids_gdf, left_on="v", right_index=True)
            .rename(
                columns=dict(
                    centroid_geometry_x="u_geometry", centroid_geometry_y="v_geometry"
                )
            )
        )

        edge_creator: Callable[[Dict[str, BaseGeometry]], LineString] = lambda x: cast(
            LineString, LineString([x["u_geometry"], x["v_geometry"]])
        )

        merged["edge"] = merged.apply(edge_creator, axis=1)

        return gpd.GeoDataFrame(
            merged[["u", "v", "edge"]],
            geometry="edge",
            crs=self.hexes_gdf.crs,
        )  # type: ignore

    def _assert_constructor_args(
        self,
        hexes_gdf: gpd.GeoDataFrame,
        osmnx_nodes_gdf: gpd.GeoDataFrame,
        osmnx_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        assert (
            hexes_gdf.index.name == "region_id"
        ), f"GraphLayerController: Expected 'region_id' as index name in hexes_gdf, got {hexes_gdf.index.name}"

        assert (
            osmnx_nodes_gdf.index.name == "osmid"
        ), f"GraphLayerController: Expected 'osmid' as index name in osmnx_nodes_gdf, got {osmnx_nodes_gdf.index.name}"

        assert (
            "x" in osmnx_nodes_gdf.columns and "y" in osmnx_nodes_gdf.columns
        ), f"GraphLayerController: Expected 'x' and 'y' columns in osmnx_nodes_gdf, got {osmnx_nodes_gdf.columns}"

        assert (
            osmnx_edges_gdf.index.name == "edge_id"
        ), f"GraphLayerController: Expected 'edge_id' as index name in osmnx_edges_gdf, got {osmnx_edges_gdf.index.name}"

        assert (
            "u" in osmnx_edges_gdf.columns and "v" in osmnx_edges_gdf.columns
        ), f"GraphLayerController: Expected 'u' and 'v' columns in osmnx_edges_gdf, got {osmnx_edges_gdf.columns}"

        unique_resolutions = self._get_unique_resolutions(hexes_gdf)

        assert (
            len(unique_resolutions) == 1
        ), f"GraphLayerController: Expected the same resolution for all hexes, got {unique_resolutions}"

    def _get_unique_resolutions(
        self,
        hexes_gdf: gpd.GeoDataFrame,
    ):
        return hexes_gdf.index.map(h3.get_resolution).unique()
