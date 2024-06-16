import colorsys
import os
import folium
from sklearn.metrics import f1_score, roc_auc_score
import streamlit as st
import torch
from joblib import load
from typing import cast
import geopandas as gpd
import numpy as np
from streamlit_folium import st_folium
import branca.colormap as cm

st.set_page_config(layout="wide", page_title="Airbnb")

MODEL_RESPONSES_PATH = "data/results_showcase/airbnb"
GRAPH_DATA_PATH = "data/results_showcase/airbnb/graph_data.pkl"
ORGANIZED_HEXES_LOCATION = "data/organized-hexes"

names_map = {
    "new_york": "New York",
    "seattle": "Seattle",
}

city_value = st.selectbox("Select a city", names_map.values())
city_value = list(names_map.keys())[list(names_map.values()).index(city_value)]


@st.cache_data
def load_graph_data_and_model_responses(city_value):
    data = load(GRAPH_DATA_PATH)
    y_hat = torch.load(
        os.path.join(MODEL_RESPONSES_PATH, f"model_responses_{city_value}.pkl")
    )

    return data, y_hat


data, y_hat = load_graph_data_and_model_responses(city_value)


f1 = f1_score(
    data[city_value]["hex"].y.detach().cpu().numpy(),
    y_hat.argmax(dim=-1).detach().cpu().numpy(),
    pos_label=1,
    average="weighted",
)

auc = roc_auc_score(
    data[city_value]["hex"].y.detach().cpu().numpy(),
    y_hat.detach().cpu().numpy(),
    average="micro",
    multi_class="ovr",
)
accuracy = (y_hat.argmax(dim=-1) == data[city_value]["hex"].y).sum().item() / len(
    data[city_value]["hex"].y
)

st.metric("F1 score", f"{f1:.4f}")
st.metric("AUC", f"{auc:.4f}")
st.metric("Accuracy", f"{accuracy:.4f}")


@st.cache_data
def load_map(city_value):

    hexes_years_folder = os.path.join(ORGANIZED_HEXES_LOCATION, city_value)

    subfolders = [
        int(f)
        for f in os.listdir(hexes_years_folder)
        if os.path.isdir(os.path.join(hexes_years_folder, f))
    ]
    highest_year = subfolders[np.argmax(subfolders)]

    hexes: gpd.GeoDataFrame = gpd.read_parquet(
        os.path.join(
            ORGANIZED_HEXES_LOCATION,
            f"{city_value}/{highest_year}/h9/count-embedder/dataset.parquet",
        )
    )

    hexes = cast(
        gpd.GeoDataFrame,
        hexes.rename(columns={"region_id": "h3_id"}).rename_axis("region_id", axis=0),
    )
    return hexes


hexes = load_map(city_value)
hexes = hexes.assign(pred=y_hat.argmax(dim=-1).detach().cpu().numpy())
hexes = hexes.assign(pred_proba=y_hat.amax(dim=-1).detach().cpu().numpy())
hexes = hexes.assign(ground_truth=data[city_value]["hex"].y.detach().cpu().numpy())


def cmap_fn(feature):
    ground_truth = feature["properties"]["ground_truth"]
    pred = feature["properties"]["pred"]

    if ground_truth == pred:
        color = colorsys.hsv_to_rgb(
            1 / 3,
            (ground_truth + 1) / 5,
            255,
        )
        return f"rgb{color}"
    abs_diff = abs(ground_truth - pred)
    color = colorsys.hsv_to_rgb(
        0,
        ((abs_diff + 1) / 5),
        255,
    )
    return f"rgb{color}"


color_positive_min = colorsys.hsv_to_rgb(
    1 / 3,
    0.2,
    255,
)
color_positive_max = colorsys.hsv_to_rgb(
    1 / 3,
    1,
    255,
)
color_negative_min = colorsys.hsv_to_rgb(
    0,
    0.2,
    255,
)
color_negative_max = colorsys.hsv_to_rgb(
    0,
    1,
    255,
)

cmap_positive = cm.LinearColormap(
    [color_positive_min, color_positive_max],
    vmin=0,
    vmax=5,
).to_step(5)
cmap_positive.caption = "Acknowledged Airbnb price class"
cmap_negative = cm.LinearColormap(
    [color_negative_min, color_negative_max], vmin=0, vmax=5
).to_step(5)
cmap_negative.caption = (
    "Absolute difference between predicted and ground truth price class"
)

map = folium.Map(
    tiles="CartoDB positron",
)
bounds = hexes.total_bounds.tolist()
map.fit_bounds([bounds[:2][::-1], bounds[2:][::-1]])

map = cast(
    gpd.GeoDataFrame,
    hexes[
        [
            "ground_truth",
            "pred",
            "pred_proba",
            "geometry",
        ]
    ],
).explore(
    m=map,
    legend=True,
    style_kwds=dict(
        fillOpacity=0.6,
        opacity=0.1,
        style_function=lambda feature: dict(
            fillColor=cmap_fn(feature),
            color="black",
        ),
    ),
    name="hexes",
)
folium.LayerControl().add_to(map)
map.add_child(cmap_positive)
map.add_child(cmap_negative)

st.header("Results map:")
st.write(
    "You can control the map layers using the layer control on the top right corner of the map."
)
st_folium(map, returned_objects=[], use_container_width=True, return_on_hover=False)

# left = list(data.keys())
# right = [x.replace("model.", "") for x in list(torch.load(MODEL_PATH).keys())]

# import numpy as np

# st.write("Left not in right")
# st.write(np.setdiff1d(left, right))
# st.write("Right not in left")
# st.write(np.setdiff1d(right, left))

# st.write(torch.unique(data["new_york"]["hex"]["y"]))
