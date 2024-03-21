"""Module with functions that join various necessary DataFrames and GeoDataFrames together.
"""

import geopandas as gpd
import pandas as pd
from srai.joiners import IntersectionJoiner


def get_accidents_per_hex_column(  # type: ignore
    accidents_gdf: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame
) -> pd.Series:  # type: ignore
    """Joins together the accidents and hexes GeoDataFrames and counts the number of accidents per hex.

    Parameters
    ----------
    accidents_gdf : gpd.GeoDataFrame
        Accidents GeoDataFrame with the point geometry of the accidents.
    hexes : gpd.GeoDataFrame
        Hexes GeoDataFrame with the hexagon geometry.

    Returns
    -------
    pd.Series
        Number of accidents per hex, indexed by `region_id`.
    """
    joiner = IntersectionJoiner()
    accidents_joint_df = (
        joiner.transform(hexes, accidents_gdf)
        .reset_index()
        .set_index("region_id")
        .groupby("region_id")
        .count()
    )

    result = pd.Series(accidents_joint_df["feature_id"])
    result.name = "accidents_count"

    return result


def create_dataset_gdf(
    regions: gpd.GeoDataFrame,
    embedding: pd.DataFrame,
    accidents_column: pd.Series,  # type: ignore
) -> gpd.GeoDataFrame:
    """Joins together the regions (index) embeddings (X) with the accidents count column (y).

    The resulting GeoDataFrame is a complete dataset with embeddings as features and accidents count as the target.

    The method automatically fills regions which have no accidents with 0.

    Parameters
    ----------
    regions : gpd.GeoDataFrame
        H3 regions GeoDataFrame indexed by the region id.
    embedding : pd.DataFrame
        Embeddings DataFrame indexed the same way as the `regions`.
    accidents_column : pd.Series
        Series with the number of accidents per region, indexed the same way as the `regions`.

    Returns
    -------
    gpd.GeoDataFrame
        The dataset GeoDataFrame with the following structure:
        - index: `region_id`
        - columns
            - the same columns as in the embedding method
            - `accidents_count` column with the number of accidents per region
            - `geometry` column with the hexagon geometry
    """
    return gpd.GeoDataFrame(embedding.join(accidents_column).fillna(0).join(regions))
