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
    """Load accidents for a city and year from an `accidents.csv` file.

    The accidents.csv file is usually in the `data/wypadki-pl` folder.

    Parameters
    ----------
    nominatim_city_name : str
        Name of the city that should comply to the Nominatim format and be present in the accidents.csv file.
        For proper name resolution, use the `resolve_nominatim_city_name` function.
        More info on Nominatim [here](https://nominatim.openstreetmap.org/ui/search.html).
    year : int
        Year that the accidents should be from, it should be present in the accidents.csv file.
    accidents_path : str
        The path to the accidents.csv file.

    Returns
    -------
    gpd.GeoDataFrame
        The resulting GeoDataFrame with the accidents for the city and year.

        Its rows represent accidents and its columns describe features of the accidents.
        The most important columns are:
        - feature_id: Row index
        - year: The year of the accident
        - month: The month of the accident
        - day: The day of the accident
        - czas_zdarzenia: The time of the accident (HH:MM)
        - geometry: The point geometry of the accident
    """
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
    """Uses SRAI [OSMPbfLoader](https://kraina-ai.github.io/srai/0.7.0/api/loaders/OSMPbfLoader/) to load OSM data for a city.

    It will attempt to look for a cached version of the data in the `osm_cache_folder` and load it if it exists.
    If it doesn't exist, it will download the data and cache it in this folder.

    The keys passed to the OSMPbfLoader are:
    ```python
    {
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
    ```

    Parameters
    ----------
    nominatim_city_name : str
        Name of the city that should comply to the Nominatim format.
        For proper name resolution, use the `resolve_nominatim_city_name` function.
        More info on Nominatim [here](https://nominatim.openstreetmap.org/ui/search.html).
    osm_cache_folder : str
        The folder where the OSM data will be cached.

    Returns
    -------
    gpd.GeoDataFrame
        The resulting GeoDataFrame with the OSM data for the city.

        It is indexed by the `feature_id` column and contains the columns named the same
        as the keys passed to the OSMPbfLoader (described above).
        The values in these columns are the types of the features present in the OSM data (string or None if not present).
    """
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
    """Uses SRAI [OSMWayLoader](https://kraina-ai.github.io/srai/0.7.0/api/loaders/OSMWayLoader/) to load OSM way data for a city.

    It will attempt to look for a cached version of the data in the `osm_way_cache_folder` and load it if it exists.
    If it doesn't exist, it will download the data and cache it in this folder.

    The `network_type` passed to the OSMWayLoader is `OSMNetworkType.DRIVE`.

    Parameters
    ----------
    nominatim_city_name : str
        Name of the city that should comply to the Nominatim format.
        For proper name resolution, use the `resolve_nominatim_city_name` function.
        More info on Nominatim [here](https://nominatim.openstreetmap.org/ui/search.html).
    osm_way_cache_folder : str
        The folder where the OSM data will be cached.

    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        Returns a tuple with two GeoDataFrames: intersections (indexed by `osmid`) and roads (indexed by `feature_id`).
    """
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


def download_hexes_for_nominatim(nominatim_city_name: str, resolution: int):
    city_name = resolve_nominatim_city_name(nominatim_city_name)
    region_gdf = geocode_to_region_gdf(city_name)
    regionalizer = H3Regionalizer(resolution)
    hexes = regionalizer.transform(region_gdf)

    return hexes


def create_hexes_gdf(
    nominatim_city_name: str, resolution: int, hexes_cache_folder: str
) -> gpd.GeoDataFrame:
    """Creates a hexagonal grid for a city using the SRAI [H3Regionalizer](https://kraina-ai.github.io/srai/0.7.0/api/regionalizers/H3Regionalizer/).

    It attempts to find a cached version of the hexes in the `hexes_cache_folder` and load it if it exists.
    If the cached version doesn't exist, it will download the data from Nominatim API, create the hexes and cache them in this folder.

    Cache is separate for each city and resolution
    Parameters
    ----------
    nominatim_city_name : str
        Name of the city that should comply to the Nominatim format.
        For proper name resolution, use the `resolve_nominatim_city_name` function.
        More info on Nominatim [here](https://nominatim.openstreetmap.org/ui/search.html).
    resolution : int
        H3 resolution (0-15). Tested on 6 through 11.
    hexes_cache_folder : str
        Cache folder for the hexes. The function will traverse the folder to find the cached hexes, according to the city and resolution.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataframe indexed with `region_id` and single column `geometry` (Polygons)
    """
    city_filename = convert_nominatim_name_to_filename(
        resolve_nominatim_city_name(nominatim_city_name)
    )
    hexes_path = os.path.join(
        hexes_cache_folder, f"{city_filename}_{resolution}.geojson"
    )
    has_cached_gdf = os.path.exists(hexes_path)
    if has_cached_gdf:
        return gpd.read_file(hexes_path).set_index("region_id")

    hexes = download_hexes_for_nominatim(nominatim_city_name, resolution)

    hexes.to_file(hexes_path, driver="GeoJSON")

    return hexes
