from sklearn.metrics import f1_score, roc_auc_score
import streamlit as st
import pandas as pd
import sys
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from streamlit_folium import st_folium
from src.organized_datasets_creation.utils.nominatim import (
    convert_nominatim_name_to_filename,
    resolve_nominatim_city_name,
)
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
import joblib
from typing import cast
import colorsys
import folium


st.set_page_config(layout="wide", page_title="Main page")
st.title("🏠Airbnb prices prediction🏠")
