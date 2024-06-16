from typing import cast
import folium
from sklearn.metrics import f1_score, roc_auc_score
import streamlit as st
import pandas as pd
import colorsys
import sys
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.metrics import f1_score
from streamlit_folium import st_folium
from src.organized_datasets_creation.utils.nominatim import (
    convert_nominatim_name_to_filename,
    resolve_nominatim_city_name,
)
from joblib import load
import branca.colormap as cm
from joblib import dump, load

st.set_page_config(layout="wide", page_title="Accidents")
st.title("🚗Accidents prediction🚗")

ACCIDENTS_LOCATION = "./data/results_showcase/accidents/accidents.parquet"
ACCIDENTS_COUNT_LOCATION = "./data/results_showcase/accidents/accidents_count.csv"
GRAPH_DATA_DICT_PATH = "./data/results_showcase/accidents/data.pkl"
MODEL_RESPONSES_PATH = "./data/results_showcase/accidents/model_responses.pkl"
ORGANIZED_HEXES_LOCATION = "./data/organized-hexes"


@st.cache_data
def load_graph_data_and_model_responses():
    data = load(GRAPH_DATA_DICT_PATH)
    responses = load(MODEL_RESPONSES_PATH)
    return data, responses


data, responses = load_graph_data_and_model_responses()

city_value = st.selectbox("Select a city", data.keys())

if city_value is None:
    sys.exit()

response = responses[city_value]


@st.cache_data
def load_map(city_value):

    city_folder_name = convert_nominatim_name_to_filename(
        resolve_nominatim_city_name(city_value)
    )

    hexes_years_folder = os.path.join(ORGANIZED_HEXES_LOCATION, city_folder_name)

    subfolders = [
        int(f)
        for f in os.listdir(hexes_years_folder)
        if os.path.isdir(os.path.join(hexes_years_folder, f))
    ]
    highest_year = subfolders[np.argmax(subfolders)]

    hexes: gpd.GeoDataFrame = gpd.read_parquet(
        os.path.join(
            ORGANIZED_HEXES_LOCATION,
            f"{city_folder_name}/{highest_year}/h9/count-embedder/dataset.parquet",
        )
    )

    hexes = cast(
        gpd.GeoDataFrame,
        hexes.rename(columns={"region_id": "h3_id"}).rename_axis("region_id", axis=0),
    )
    return hexes


hexes = load_map(city_value)

hexes = hexes.assign(pred=response.argmax(dim=-1).detach().cpu().numpy())
hexes = hexes.assign(pred_proba=response[:, 1].cpu().numpy())
hexes = hexes.assign(ground_truth=data[city_value]["ground_truth"])

accidents_count = pd.read_csv(ACCIDENTS_COUNT_LOCATION)
hexes = hexes.reset_index().merge(accidents_count, on="h3_id", how="inner")


def create_error_column(row):
    if row["pred"] == row["ground_truth"] and row["pred"] == 1:
        return "TP"
    elif row["pred"] == row["ground_truth"] and row["pred"] == 0:
        return "TN"
    elif row["pred"] != row["ground_truth"] and row["ground_truth"] == 1:
        return "FN"
    elif row["pred"] != row["ground_truth"] and row["ground_truth"] == 0:
        return "FP"


hexes["error"] = hexes.apply(create_error_column, axis=1)  # type: ignore

max_accidents = hexes["accidents_count"].max()
mean_accidents = hexes["accidents_count"].mean()

min_color_saturation = 0.1


def cmap_fn(feature):
    error_type = feature["properties"]["error"]
    acc_count = feature["properties"]["accidents_count"]

    if error_type == "FN":
        color = colorsys.hsv_to_rgb(
            0,
            min_color_saturation
            + (1 - min_color_saturation) * (acc_count / max_accidents),
            255,
        )
        return f"rgb{color}"
    elif error_type == "FP":
        color = colorsys.hsv_to_rgb(30 / 360, 0.1, 255)
        return f"rgb{color}"
    elif error_type == "TP" or error_type == "TN":
        color = colorsys.hsv_to_rgb(
            1 / 3,
            min_color_saturation
            + (1 - min_color_saturation) * (acc_count / max_accidents),
            255,
        )
        return f"rgb{color}"
    return "white"


gdf_accidents = gpd.read_parquet(ACCIDENTS_LOCATION)


TP_accidents = hexes.loc[hexes["error"] == "TP", "accidents_count"].sum()
FN_accidents = hexes.loc[hexes["error"] == "FN", "accidents_count"].sum()

correctly_predicted_accidents_ratio = (TP_accidents) / (TP_accidents + FN_accidents)
st.header("Results:")

st.metric(
    "F1 score",
    data[city_value]["F1 score"],
)
st.metric(
    "AUC",
    data[city_value]["AUC"],
)
st.metric(
    "Accuracy",
    data[city_value]["Accuracy"],
)
st.metric(
    "Percent of correctly predicted accidents",
    data[city_value]["Percent of correctly predicted accidents"],
)


color_positive_min = colorsys.hsv_to_rgb(
    1 / 3,
    min_color_saturation,
    255,
)
color_positive_max = colorsys.hsv_to_rgb(
    1 / 3,
    1,
    255,
)
color_negative_min = colorsys.hsv_to_rgb(
    0,
    min_color_saturation,
    255,
)
color_negative_max = colorsys.hsv_to_rgb(
    0,
    1,
    255,
)

cmap_positive = cm.LinearColormap(
    [color_positive_min, color_positive_max], vmin=0, vmax=max_accidents
)
cmap_positive.caption = "Acknowledged accidents"
cmap_negative = cm.LinearColormap(
    [color_negative_min, color_negative_max], vmin=0, vmax=max_accidents
)
cmap_negative.caption = "Omitted accidents"

with st.spinner("Loading map..."):
    st.header("Results map:")
    st.write(
        "You can control the map layers using the layer control on the top right corner of the map."
    )
    map = folium.Map(
        tiles="CartoDB positron",
    )
    bounds = hexes.total_bounds.tolist()
    map.fit_bounds([bounds[:2][::-1], bounds[2:][::-1]])

    map = cast(
        gpd.GeoDataFrame,
        gdf_accidents.loc[
            gdf_accidents["mie_nazwa"] == city_value.replace(", Poland", ""), :
        ],
    ).explore(m=map, name="accidents", tooltip=None)

    map = cast(
        gpd.GeoDataFrame,
        hexes[
            [
                "ground_truth",
                "pred",
                "pred_proba",
                "error",
                "accidents_count",
                "geometry",
            ]
        ],
    ).explore(
        m=map,
        column="error",
        legend=True,
        style_kwds=dict(
            fillOpacity=0.6,
            opacity=0.1,
            style_function=lambda feature: dict(
                fillColor=cmap_fn(feature),
                color="black",
            ),
        ),
        categorical=True,
        name="hexes",
    )
    folium.LayerControl().add_to(map)
    map.add_child(cmap_positive)
    map.add_child(cmap_negative)
    st_folium(map, returned_objects=[], use_container_width=True, return_on_hover=False)
