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
st.title("🐸Żabka shops prediction🐸")

ZABKA_SHOPS_LOCATION = "./data/results_showcase/zabka_shops/zabka_locations.csv"
DATA_DICT_PATH = "./data/results_showcase/zabka_shops/tabular_data_zabka.pkl" 
MODEL_PATH = "./data/results_showcase/zabka_shops/model_zabka.pkl"
ORGANIZED_HEXES_LOCATION = "./data/organized-hexes"

@st.cache_data
def load_graph_data_and_model():
    data = joblib.load(DATA_DICT_PATH)
    model = joblib.load(MODEL_PATH)
    return data, model


data, model = load_graph_data_and_model()


city_value = st.selectbox("Select a city", data.keys())

# data = data.drop("hex_id", axis=1)
print(data)

if city_value is None:
    sys.exit()



# Logistic regression fit
folds = [
    ("Wrocław", "Kraków"),
    ("Kraków", "Poznań"),
    ("Poznań", "Szczecin"),
    ("Szczecin", "Warszawa"),
    ("Warszawa", "Wrocław"),
]
scaler = StandardScaler()
X = pd.concat(
[
    m["X"]
    for key, m in data.items()
    if key != city_value
]
).to_numpy()
y = (pd.concat(
    [
        m["y"]
        for key, m in data.items()
        if key != city_value
    ]
).to_numpy().ravel())

X = scaler.fit_transform(X)
test_X = data[city_value]["X"].to_numpy()
test_X = scaler.transform(test_X)
test_y = data[city_value]["y"].to_numpy().ravel()
y_pred = model.predict(test_X)
y_proba = model.predict_proba(test_X)[:, 1]



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

hexes = hexes.assign(pred=y_pred)
hexes = hexes.assign(pred_proba=y_proba)
hexes = hexes.assign(ground_truth=data[city_value]['y'])

zabkas_count = pd.read_csv(ZABKA_SHOPS_LOCATION)
# hexes = hexes.reset_index().merge(zabkas_count, on="h3_id", how="inner")




def cmap_fn(feature):
    error_type = feature["properties"]["error"]
    acc_count = feature["properties"]["accidents_count"]
    min_color_saturation = 0.1
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




st.header("Results:")
st.metric("F1 score", 
          f"{f1_score(test_y, y_pred, pos_label=1, average='binary'):.4f}",
          )
st.metric("AUC", 
          f"{roc_auc_score(test_y, y_proba, average='micro'):.4f}",
          )
st.metric("Accuracy", 
          f"{(y_pred == test_y).mean():.4f}",
          )



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

    # map = cast(
    #     gpd.GeoDataFrame,
    #     gdf_accidents.loc[
    #         gdf_accidents["mie_nazwa"] == city_value.replace(", Poland", ""), :
    #     ],
    # ).explore(m=map, name="accidents", tooltip=None)

    map = cast(
        gpd.GeoDataFrame,
        hexes[
            [
                "ground_truth",
                "pred",
                "pred_proba",
                # "error",
                # "accidents_count",
                "geometry",
            ]
        ],
    ).explore(
        m=map,
        # column="error",
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
    st_folium(map, returned_objects=[], use_container_width=True, return_on_hover=False)


