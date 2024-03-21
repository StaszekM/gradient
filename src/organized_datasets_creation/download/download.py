from typing import Dict, List, Tuple, Union
import geopandas as gpd
import pandas as pd
from srai.regionalizers import geocode_to_region_gdf
from ..utils.nominatim import (
    convert_nominatim_name_to_filename,
    resolve_nominatim_city_name,
)
import os
from srai.loaders import OSMPbfLoader, OSMWayLoader, OSMNetworkType

from srai.regionalizers import H3Regionalizer


def load_accidents_for_city(
    nominatim_city_name: str, year: int, accidents_path: str
) -> gpd.GeoDataFrame:
    """Loads accidents.csv, filters it by city and year, and returns a GeoDataFrame with accidents for this city and year."""
    accidents_df = pd.read_csv(accidents_path)
    accidents_df = (
        accidents_df[
            (accidents_df["gmi_nazwa"] == nominatim_city_name)
            & (accidents_df["year"] == year)
        ]
        .reset_index()
        .rename(columns={"index": "feature_id"})
        .set_index("feature_id")
    )
    accidents_gdf = gpd.GeoDataFrame(
        accidents_df,
        geometry=gpd.points_from_xy(accidents_df.wsp_gps_x, accidents_df.wsp_gps_y),
        crs="EPSG:4326",
    )  # type: ignore

    return accidents_gdf


osm_keys: Dict[str, Union[List[str], str, bool]] = {
    "aeroway": True,
    "amenity": True,
    "building": True,
    "healthcare": True,
    "historic": True,
    "landuse": True,
    "leisure": True,
    "military": True,
    "natural": True,
    "office": True,
    "shop": True,
    "sport": True,
    "tourism": True,
    "waterway": True,
    "water": True,
}


def load_osm_data(nominatim_city_name: str, osm_cache_folder: str) -> gpd.GeoDataFrame:
    city_filename = convert_nominatim_name_to_filename(
        resolve_nominatim_city_name(nominatim_city_name)
    )
    osm_path = os.path.join(osm_cache_folder, f"{city_filename}.geojson")
    has_cached_osm = os.path.exists(osm_path)
    if not has_cached_osm:
        loader = OSMPbfLoader(download_directory=os.path.join(osm_cache_folder, "srai"))
        features_gdf = loader.load(geocode_to_region_gdf(nominatim_city_name), osm_keys)
        features_gdf.to_file(osm_path, driver="GeoJSON")
        return features_gdf

    return gpd.read_file(osm_path).set_index("feature_id")


def load_osm_way_data(
    nominatim_city_name: str, osm_way_cache_folder: str
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    city_filename = convert_nominatim_name_to_filename(
        resolve_nominatim_city_name(nominatim_city_name)
    )
    osm_way_nodes_path = os.path.join(
        osm_way_cache_folder, f"{city_filename}.nodes.geojson"
    )
    osm_way_edges_path = os.path.join(
        osm_way_cache_folder, f"{city_filename}.edges.geojson"
    )
    has_cached_osm_way = os.path.exists(osm_way_nodes_path) & os.path.exists(
        osm_way_edges_path
    )
    if not has_cached_osm_way:
        way_loader = OSMWayLoader(
            network_type=OSMNetworkType.DRIVE,
        )
        nodes_gdf, edges_gdf = way_loader.load(
            geocode_to_region_gdf(nominatim_city_name)
        )
        nodes_gdf.to_file(osm_way_nodes_path, driver="GeoJSON")
        edges_gdf.to_file(osm_way_edges_path, driver="GeoJSON")
        return nodes_gdf, edges_gdf

    return gpd.read_file(osm_way_nodes_path).set_index("osmid"), gpd.read_file(
        osm_way_edges_path
    ).set_index("feature_id")


def create_hexes_gdf(
    nominatim_city_name: str, resolution: int, hexes_cache_folder: str
) -> gpd.GeoDataFrame:
    city_filename = convert_nominatim_name_to_filename(
        resolve_nominatim_city_name(nominatim_city_name)
    )
    hexes_path = os.path.join(
        hexes_cache_folder, f"{city_filename}_{resolution}.geojson"
    )
    has_cached_gdf = os.path.exists(hexes_path)
    if has_cached_gdf:
        return gpd.read_file(hexes_path).set_index("region_id")

    city_name = resolve_nominatim_city_name(nominatim_city_name)
    region_gdf = geocode_to_region_gdf(city_name)
    regionalizer = H3Regionalizer(resolution)
    hexes = regionalizer.transform(region_gdf)

    hexes.to_file(hexes_path, driver="GeoJSON")

    return hexes
