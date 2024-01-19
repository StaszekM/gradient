"""Module for downloading OSM Historical data"""
import geopandas as gpd
from typing import List, Optional
import os
import pandas as pd
from srai.loaders.osm_loaders.filters import OsmTagsFilter
from dateutil.rrule import rrule, MONTHLY
from datetime import datetime
from srai.loaders import OSMOnlineLoader
import osmnx as ox
import string
import time


class OSMHistoricalLoader:
    """
    Open Street Map historical data loader.

    Allows the retrieval of historical data using the module to retrieve and record spatial data 
    from a specified time period.
    """
    def __init__(self, 
                 area:  gpd.GeoDataFrame, 
                 start_date_str: str, end_date_str: str, 
                 query: OsmTagsFilter, 
                 load_freq: int = MONTHLY
                 ) -> None:
        """
        Init OSMHistoricalLoader

        Args:
            area (gpd.GeoDataFrame): Area for which to download objects.

            start_date_str (str): Start date for the period during which the data will be collected. 
                                  Required format: %Y-%m-%dT%H:%M:%SZ
            
            end_date_str (str): End date for the period during which the data will be collected. 
                                Required format: %Y-%m-%dT%H:%M:%SZ
            
            query (dict): A dictionary
                specifying which tags to download.
                The keys should be OSM tags (e.g. `building`, `amenity`).
                The values should either be `True` for retrieving all objects with the tag,
                string for retrieving a single tag-value pair
                or list of strings for retrieving all values specified in the list.
                `tags={'leisure': 'park}` would return parks from the area.
                `tags={'leisure': 'park, 'amenity': True, 'shop': ['bakery', 'bicycle']}`
                would return parks, all amenity types, bakeries and bicycle shops.
            
            load_freq (int): dateutil.rrule int constant, one of: 'YEARLY', 'MONTHLY', 
                            'WEEKLY', 'DAILY', 'HOURLY', 'MINUTELY', 'SECONDLY' (see dateutil.rrule docs)

        """

        self.area = area
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.query = query
        self.load_freq = load_freq

    
    def __remove_punctuation(self, input_string: str):
        """
        Helper method to remove punctuation from given string (area index which is the name of the place).
        Used for file save name creation.

        Args:
            input_string (str): String for which punctuation will be removed.

        Returns:
            str: Cleaned string.

        """
        translation_table = str.maketrans("", "", string.punctuation)
        cleaned_string = input_string.translate(translation_table)
        
        return cleaned_string
    

    def __load_single_df(self, date_str: str, max_retries_allowed: int) -> Optional[gpd.GeoDataFrame]:
        """
        Loads single GeoDataFrame using OSMOnlineLoader for given date and query.

        Args: 
            date_str (str): Date in format %Y-%m-%dT%H:%M:%SZ

        Returns:
            gpd.GeoDataFrame: Data for given area and date with features selected in query.

        """
        cs = f'[out:json][timeout:300][date:"{date_str}"]'
        ox.settings.overpass_settings = cs # type: ignore

        retry_delay = 60  # Set the delay in seconds for each retry

        for retry_count in range(max_retries_allowed):
            try:
                loader: OSMOnlineLoader = OSMOnlineLoader()
                data_gdf: gpd.GeoDataFrame = loader.load(self.area, self.query)
                return data_gdf
            except Exception as e:
                if retry_count == max_retries_allowed - 1:
                    raise e 
                print(f"ConnectionError: Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    

    def save_data_for_period(self, save_path: Optional[str] = None, max_retries_allowed: int = 3):
        """
        Load data for given time period and save and save as a GeoDataFrame covering the entire given year. 
        If the time interval covers several years, several objects will be created. Each saved GeoDataFrame
        also contains "date" column that refers to the date from which the data comes.

        Args:
            save_path (Optional[str]): Optional parameter for save path. DO NOT include file extension. 
        """
        
        assert datetime.strptime(self.start_date_str, "%Y-%m-%dT%H:%M:%SZ"), f"Invalid date format: {self.start_date_str}"
        assert datetime.strptime(self.end_date_str, "%Y-%m-%dT%H:%M:%SZ"), f"Invalid date format: {self.end_date_str}"

        if not save_path:
            if not os.path.exists("data/geospatial_features"):
                 os.makedirs("data/geospatial_features")

        area_name = self.__remove_punctuation(self.area.index[0]).replace(" ", "_")
        start_date = datetime.strptime(self.start_date_str, "%Y-%m-%dT%H:%M:%SZ")
        end_date = datetime.strptime(self.end_date_str, "%Y-%m-%dT%H:%M:%SZ")
        # Generate days between start_date and end_date every specified time interval
        whole_period: List[datetime] = [dt for dt in rrule(self.load_freq, 
                                                                          dtstart=start_date, 
                                                                          until=end_date)]
        current_year = str(start_date.year)
        gdfs_list = []
        for date in whole_period:
            next_year = str(date.year)
            if current_year != next_year:
                final_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(pd.concat(gdfs_list, 
                                                                         ignore_index=False))
                if save_path:
                    final_gdf.to_pickle(f"{save_path}_{next_year}.pkl")
                else:
                    final_gdf.to_pickle(f"data/geospatial_features/gdf_{area_name}_{current_year}.pkl")
                gdfs_list: List[gpd.GeoDataFrame] = []
                current_year =  next_year
            else:
                gdf_for_day = self.__load_single_df(date.strftime("%Y-%m-%dT%H:%M:%SZ"), max_retries_allowed)
                if gdf_for_day is not None:
                    gdf_for_day['date'] = date
                    gdfs_list.append(gdf_for_day)

        final_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(pd.concat(gdfs_list, 
                                                                 ignore_index=False))
        if save_path:
            final_gdf.to_pickle(f"{save_path}_{current_year}.pkl")
        else: 
            final_gdf.to_pickle(f"data/geospatial_features/gdf_{area_name}_{current_year}.pkl")
        