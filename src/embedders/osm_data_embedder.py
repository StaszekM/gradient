"""Module for performing OSM Data Embedding"""


from typing import Iterable, List, Union
import pandas as pd
import geopandas as gpd
from srai.loaders import OSMOnlineLoader
from shapely.geometry.base import BaseGeometry
from srai.embedders import Embedder
from srai.joiners import IntersectionJoiner
from srai.regionalizers import Regionalizer


class OSMDataEmbedder:
    """
    Open Street Map Data Embedder.

    Data Embedder allows to perform embeddings on loaded area with specified query. 
    Uses OnlineLoader to prepare the dataframe and IntersectionJoiner for tag aggregation.
    """
    def __init__(self, area: Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame], 
                 embedder: Embedder, 
                 regionalizer: Regionalizer, 
                 query: dict) -> None:
        """ 
        Init OSMDataEmbedder.

        Args:
            area (Union[BaseGeometry, Iterable[BaseGeometry], gpd.GeoSeries, gpd.GeoDataFrame]):
                Area for which to download objects. 

            embedder (Embedder): Choosen Embedder tfor further feature transformation.

            regionalizer (Regionalizer): Regionalizer that will split data into regions.
            
            query (dict): A dictionary
                specifying which tags to download.
                The keys should be OSM tags (e.g. `building`, `amenity`).
                The values should either be `True` for retrieving all objects with the tag,
                string for retrieving a single tag-value pair
                or list of strings for retrieving all values specified in the list.
                `tags={'leisure': 'park}` would return parks from the area.
                `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
                would return parks, all amenity types, bakeries and bicycle shops.
        
        """
        self.area = area
        self.embedder = embedder
        self.regionalizer = regionalizer
        self.query = query
    
    def __load_regions(self) -> gpd.GeoDataFrame:
        """ 
        Load regions using OSMOnlineLoader.

        In case of overloading the data is loaded for key-value pairs from query dictionary.

        Returns:
            gpd.GeoDataFrame: Data for given area with features selected in query.
        """
        loader = OSMOnlineLoader()
        try:
            data_gdf: gpd.GeoDataFrame = loader.load(self.area, self.query)
        except: 
            print("Error occured while loading data. Loading tags separately started...")
            gdf_list: List[gpd.GeoDataFrame] = []
            for key, value in self.query.items():
                single_query = {key: value}
                data_gdf: gpd.GeoDataFrame = loader.load(self.area, single_query)
                gdf_list.append(data_gdf)
            data_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(pd.concat(gdf_list))
        return data_gdf
        
    
    def make_embeddings(self) -> pd.DataFrame:
        """ 
        Performs Embedding using choosen Embedder and IntersectionJoiner.

        Returns:
            pd.Dataframe: Data with embeddings.
        """
        regions_gdf: gpd.GeoDataFrame = self.regionalizer.transform(self.area)
        data_gdf: gpd.GeoDataFrame = self.__load_regions()
        joiner = IntersectionJoiner()
        joint_df: gpd.GeoDataFrame = joiner.transform(regions_gdf,
                                                       data_gdf)
        embeddings: pd.DataFrame = self.embedder.transform(regions_gdf,
                                                       data_gdf,
                                                         joint_df)
        return embeddings



    