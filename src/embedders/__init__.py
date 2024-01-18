"""
This module contains embedders, used to convert spatial data to their vector representations.

Embedders are designed to unify different types of spatial data embedding methods, such as hex2vec
or gtfs2vec into a single interface. This allows to easily switch between different embedding
methods without changing the rest of the code.
"""

from .osm_data_embedder import OSMDataEmbedder

__all__= ["OSMDataEmbedder"]