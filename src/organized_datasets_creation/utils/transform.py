import geopandas as gpd
import pandas as pd
from srai.joiners import IntersectionJoiner


def get_accidents_per_hex_column(  # type: ignore
    accidents_gdf: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame
) -> pd.Series:  # type: ignore
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
    return gpd.GeoDataFrame(embedding.join(accidents_column).fillna(0).join(regions))
