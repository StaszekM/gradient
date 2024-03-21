from folium import Map
import geopandas as gpd
import pandas as pd
from typing import cast
import numpy as np
from sklearn.cluster import KMeans
from srai.plotting import plot_numeric_data


def display_count_embedder_sanity_check(
    hexes: gpd.GeoDataFrame, embedding: pd.DataFrame, osm_data: gpd.GeoDataFrame
) -> Map:
    """Displays the sanity check for the count embedder dataset.

    The sanity check is a map with the hexes colored by the number of ATMs in the region.
    The function may work unexpectedly if the `amenity_atm` column is not present in the `embedding` DataFrame
    or if there are no ATMs in any of the regions.
    """
    region_ids = embedding[embedding.amenity_atm > 3].index
    if len(region_ids) == 0:
        region_ids = embedding[embedding.amenity_atm > 2].index
    if len(region_ids) == 0:
        region_ids = embedding[embedding.amenity_atm > 1].index
    if len(region_ids) == 0:
        region_ids = embedding[embedding.amenity_atm > 0].index
    m = cast(gpd.GeoDataFrame, hexes[np.isin(hexes.index, region_ids)]).explore()

    return cast(
        gpd.GeoDataFrame,
        osm_data.loc[osm_data["amenity"] == "atm", ["geometry", "amenity"]],  # type: ignore
    ).explore(m=m, color="red")


def display_hex2vec_sanity_check(
    hexes: gpd.GeoDataFrame, embedding: pd.DataFrame, osm_data: gpd.GeoDataFrame
) -> Map:
    """Displays the sanity check for the hex2vec dataset.

    All hexes' embeddings are clusterized using KMeans with 5 clusters and random state of 42.
    The sanity check is a map with the hexes colored by the cluster they belong to.

    The `osm_data` parameter is not used in this function but is kept for consistency with the other functions in this module.
    """

    clusterizer = KMeans(n_clusters=5, random_state=42)
    clusterizer.fit(embedding)

    embedding["cluster"] = clusterizer.labels_

    return plot_numeric_data(hexes, "cluster", embedding)


def display_highway2vec_sanity_check(
    hexes: gpd.GeoDataFrame, embedding: pd.DataFrame, osm_data: gpd.GeoDataFrame
) -> Map:
    """Displays the sanity check for the highway2vec dataset.

    All hexes' embeddings are clusterized using KMeans with 5 clusters and random state of 42.
    The sanity check is a map with the hexes colored by the cluster they belong to.

    The `osm_data` parameter is not used in this function but is kept for consistency with the other functions in this module.
    """

    clusterizer = KMeans(n_clusters=5, random_state=42)
    clusterizer.fit(embedding)

    embedding["cluster"] = clusterizer.labels_

    return plot_numeric_data(hexes, "cluster", embedding)
