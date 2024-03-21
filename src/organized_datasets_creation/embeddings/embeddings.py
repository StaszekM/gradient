import os
from typing import Tuple
import geopandas as gpd
import pandas as pd
from srai.joiners import IntersectionJoiner
from srai.embedders import CountEmbedder
from srai.neighbourhoods import H3Neighbourhood


from ..download.download import (
    create_hexes_gdf,
    load_osm_data,
    load_osm_way_data,
)
from ..utils.nominatim import (
    convert_nominatim_name_to_filename,
    resolve_nominatim_city_name,
)
import torch
from pytorch_lightning import seed_everything

from srai.embedders import CountEmbedder, Hex2VecEmbedder, Highway2VecEmbedder


import warnings
from pytorch_lightning import seed_everything


def create_count_embedder_dataset(
    nominatim_city_name: str,
    resolution: int,
    hexes_cache_folder: str,
    osm_cache_folder: str,
    osm_way_cache_folder: str,
    hex2vec_cache_folder: str,
    highway2vec_cache_folder: str,
    SEED: int = 42,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Creates full dataset with features provided by SRAI [CountEmbedder](https://kraina-ai.github.io/srai/0.7.0/api/embedders/CountEmbedder/).

    The process of creating the dataset involves:
    - loading hexes GeoDataFrame for given city and resolution (from cache if possible)
    - loading OSM data for given city (from cache if possible)
    - joining hexes with OSM data using SRAI [IntersectionJoiner](https://kraina-ai.github.io/srai/0.7.0/api/joiners/IntersectionJoiner/)
    - creating features using CountEmbedder (with `count_subcategories=True`,
        i.e. it will count each subcategory individually, e.g. 'amenity_parking' instead of grouping, e.g 'amenity')

    Some parameters are unused, but are included for consistency with other functions in the module.

    Parameters
    ----------
    nominatim_city_name : str
        Name of the city that should comply to the Nominatim format.
        For proper name resolution, use the `resolve_nominatim_city_name` function.
        More info on Nominatim [here](https://nominatim.openstreetmap.org/ui/search.html).
    resolution : int
        H3 resolution (0-15). Tested on 6 through 11.
    hexes_cache_folder : str
        Location of folder where the function will look for existing hexes GeoDataFrames for this city and resolution.
    osm_cache_folder : str
        Location of folder where the function will look for existing OSM data for this city.
    osm_way_cache_folder : str
        Unused
    hex2vec_cache_folder : str
        Unused
    highway2vec_cache_folder : str
        Unused
    SEED : int, optional
        Unused

    Returns
    -------
    Tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]
        The tuple contains:
        - hexes GeoDataFrame with geometries indexed by `region_id`
        - embeddings DataFrame, indexed by `region_id`
        - original OSM data GeoDataFrame
    """
    hexes = create_hexes_gdf(nominatim_city_name, resolution, hexes_cache_folder)
    osm_data = load_osm_data(nominatim_city_name, osm_cache_folder)

    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(hexes, osm_data, return_geom=True)
    wide_embedder = CountEmbedder(count_subcategories=True)

    embedding = wide_embedder.transform(hexes, osm_data, joint_gdf)

    return hexes, embedding, osm_data


def create_hex2vec_dataset(
    nominatim_city_name: str,
    resolution: int,
    hexes_cache_folder: str,
    osm_cache_folder: str,
    osm_way_cache_folder: str,
    hex2vec_cache_folder: str,
    highway2vec_cache_folder: str,
    SEED: int = 42,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Creates full dataset with features provided by SRAI [Hex2VecEmbedder](https://kraina-ai.github.io/srai/0.7.0/api/embedders/Hex2VecEmbedder/).

    The process of creating the dataset involves:
    - loading hexes GeoDataFrame for given city and resolution (from cache if possible)
    - loading OSM data for given city (from cache if possible)
    - loading embeddings (from cache if possible)

    If embeddings are not cached, the function will create them using Hex2VecEmbedder and cache them.

    Parameters for the embedding process:
    - encoder_sizes: `[15, 10]`
    - max_epochs: `10`
    - batch_size: `100`

    Some parameters are unused, but are included for consistency with other functions in the module.

    Parameters
    ----------
    nominatim_city_name : str
        Name of the city that should comply to the Nominatim format.
        For proper name resolution, use the `resolve_nominatim_city_name` function.
        More info on Nominatim [here](https://nominatim.openstreetmap.org/ui/search.html).
    resolution : int
        H3 resolution (0-15). Tested on 6 through 11.
    hexes_cache_folder : str
        Location of folder where the function will look for existing hexes GeoDataFrames for this city and resolution.
    osm_cache_folder : str
        Location of folder where the function will look for existing OSM data for this city.
    osm_way_cache_folder : str
        Unused
    hex2vec_cache_folder : str
        Location of folder where the function will look for existing embeddings for this city and resolution.
    highway2vec_cache_folder : str
        Unused
    SEED : int, optional
        Seed passed to `torch_lightning.seed_everything()` before fitting embeddings, by default 42

    Returns
    -------
    Tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]
        The tuple contains:
        - hexes GeoDataFrame with geometries indexed by `region_id`
        - embeddings DataFrame, indexed by `region_id`
          It contains embeddings columns with names `emb0`, `emb1` up to `emb9`, inclusive
        - original OSM data GeoDataFrame
    """
    hexes = create_hexes_gdf(nominatim_city_name, resolution, hexes_cache_folder)
    osm_data = load_osm_data(nominatim_city_name, osm_cache_folder)

    city_filename = convert_nominatim_name_to_filename(
        resolve_nominatim_city_name(nominatim_city_name)
    )
    hex2vec_path = os.path.join(
        hex2vec_cache_folder, f"{resolution}_{city_filename}.csv"
    )
    has_cached_hex2vec = os.path.exists(hex2vec_path)

    if has_cached_hex2vec:
        return hexes, pd.read_csv(hex2vec_path, index_col=0), osm_data

    seed_everything(SEED)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    encoder_sizes = [15, 10]
    max_epochs = 10
    batch_size = 100

    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(hexes, osm_data, return_geom=False)

    neighbourhood = H3Neighbourhood(hexes)
    embedder = Hex2VecEmbedder(encoder_sizes=encoder_sizes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = embedder.fit_transform(
            hexes,
            osm_data,
            joint_gdf,
            neighbourhood,
            trainer_kwargs={
                "max_epochs": max_epochs,
                "accelerator": accelerator,
                "logger": False,
                "enable_checkpointing": False,
            },
            batch_size=batch_size,
        )

    embedding.rename(
        columns={key: f"emb{key}" for key in embedding.columns}, inplace=True
    )
    embedding.to_csv(hex2vec_path)

    return hexes, embedding, osm_data


def create_highway2vec_dataset(
    nominatim_city_name: str,
    resolution: int,
    hexes_cache_folder: str,
    osm_cache_folder: str,
    osm_way_cache_folder: str,
    hex2vec_cache_folder: str,
    highway2vec_cache_folder: str,
    SEED: int = 42,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """Creates full dataset with features provided by SRAI [Highway2VecEmbedder](https://kraina-ai.github.io/srai/0.7.0/api/embedders/Highway2VecEmbedder/).

    The process of creating the dataset involves:
    - loading hexes GeoDataFrame for given city and resolution (from cache if possible)
    - loading OSM way data for given city (from cache if possible)
    - loading embeddings (from cache if possible)

    If embeddings are not cached, the function will create them using Highway2VecEmbedder and cache them.

    Parameters for the embedding process are the same as in the original SRAI implementation of Hex2VecEmbedder.
    Refer to the SRAI documentation to learn more about the parameters.

    Parameters
    ----------
    nominatim_city_name : str
        Name of the city that should comply to the Nominatim format.
        For proper name resolution, use the `resolve_nominatim_city_name` function.
        More info on Nominatim [here](https://nominatim.openstreetmap.org/ui/search.html).
    resolution : int
        H3 resolution (0-15). Tested on 6 through 11.
    hexes_cache_folder : str
        Location of folder where the function will look for existing hexes GeoDataFrames for this city and resolution.
    osm_cache_folder : str
        Unused
    osm_way_cache_folder : str
        Location of folder where the function will look for existing OSM way data for this city.
    hex2vec_cache_folder : str
        Unused
    highway2vec_cache_folder : str
        Location of folder where the function will look for existing embeddings for this city and resolution.
    SEED : int, optional
        Seed passed to `torch_lightning.seed_everything()` before fitting embeddings, by default 42

    Returns
    -------
    Tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]
        The tuple contains:
        - hexes GeoDataFrame with geometries indexed by `region_id`
        - embeddings DataFrame, indexed by `region_id`
          It contains embeddings columns with names `emb0`, `emb1` up to `emb29`, inclusive
        - original OSMNX way data GeoDataFrame
    """
    hexes = create_hexes_gdf(nominatim_city_name, resolution, hexes_cache_folder)
    _, osm_way_edges_data = load_osm_way_data(nominatim_city_name, osm_way_cache_folder)

    city_filename = convert_nominatim_name_to_filename(
        resolve_nominatim_city_name(nominatim_city_name)
    )
    highway2vec_path = os.path.join(
        highway2vec_cache_folder, f"{resolution}_{city_filename}.csv"
    )
    has_cached_highway2vec = os.path.exists(highway2vec_path)

    if has_cached_highway2vec:
        return hexes, pd.read_csv(highway2vec_path, index_col=0), osm_way_edges_data

    seed_everything(SEED)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(hexes, osm_way_edges_data, return_geom=False)

    embedder = Highway2VecEmbedder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = embedder.fit_transform(
            hexes,
            osm_way_edges_data,
            joint_gdf,
            trainer_kwargs={
                "accelerator": accelerator,
                "logger": False,
                "enable_checkpointing": False,
            },
        )

    embedding.rename(
        columns={key: f"emb{key}" for key in embedding.columns}, inplace=True
    )
    embedding.to_csv(highway2vec_path)

    return hexes, embedding, osm_way_edges_data
