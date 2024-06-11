from sklearn.metrics import f1_score, roc_auc_score
import streamlit as st
import pandas as pd
import colorsys
import sys
from src.graph_layering.data_processing import Normalizer
from src.lightning.hetero_gnn_module import HeteroGNNModule
import os
import pandas as pd
import torch
import geopandas as gpd
import numpy as np
from sklearn.metrics import f1_score
from streamlit_folium import st_folium
from src.organized_datasets_creation.utils.nominatim import (
    convert_nominatim_name_to_filename,
    resolve_nominatim_city_name,
)

st.set_page_config(layout="wide", page_title="Main page")

ACCIDENTS_LOCATION = "./data/results_showcase/accidents/accidents.parquet"
ACCIDENTS_COUNT_LOCATION = "./data/results_showcase/accidents/accidents_count.csv"
GRAPH_DATA_DICT_PATH = "./data/results_showcase/accidents/data.pkl"
MODEL_PATH = "./data/results_showcase/accidents/model.ckpt"
ORGANIZED_HEXES_LOCATION = "./data/organized-hexes"


def load_graph_data_and_model():
    data = pd.read_pickle(GRAPH_DATA_DICT_PATH)

    model = HeteroGNNModule.load_from_checkpoint(
        MODEL_PATH, hetero_data=list(data.values())[3], map_location=torch.device("cpu")
    )
    return data, model


data, model = load_graph_data_and_model()

city_value = st.selectbox("Select a city", data.keys())

if city_value is None:
    sys.exit()


folds = [
    ("Wrocław, Poland", "Kraków, Poland"),
    ("Kraków, Poland", "Poznań, Poland"),
    ("Poznań, Poland", "Szczecin, Poland"),
    ("Szczecin, Poland", "Warszawa, Poland"),
    ("Warszawa, Poland", "Wrocław, Poland"),
]

normalizer = Normalizer()
val_city, test_city = next((x for x in folds if x[1] == city_value), (None, None))
train_data = [data[x] for x in data.keys() if x not in [val_city, test_city]]
test_data = data[test_city]

normalizer.fit(train_data)
normalizer.transform_inplace([test_data])


def forward(city_value):
    with torch.no_grad():
        city_data = data[city_value]
        response = model(
            city_data.x_dict, city_data.edge_index_dict, city_data.edge_attr_dict
        )
        response = torch.softmax(response["hex"], dim=-1)
        return response


response = forward(city_value)
st.header("Results:")
st.table(
    {
        "F1 score": f1_score(
            data[city_value]["hex"].y.cpu().numpy(),
            response.argmax(dim=-1).detach().cpu().numpy(),
            pos_label=1,
            average="binary",
        ),
        "AUC": roc_auc_score(
            test_data["hex"].y.cpu().numpy(),
            response[:, 1].cpu().numpy(),
            average="micro",
        ),
        "Accuracy": (response.argmax(dim=-1) == test_data["hex"].y).sum().item()
        / len(test_data["hex"].y),
    },
)


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

    hexes = hexes.rename(columns={"region_id": "h3_id"}).rename_axis(
        "region_id", axis=0
    )
    return hexes


hexes = load_map(city_value)

hexes = hexes.assign(pred=response.argmax(dim=-1).detach().cpu().numpy())
hexes = hexes.assign(pred_proba=response[:, 1].cpu().numpy())
hexes = hexes.assign(ground_truth=data[city_value]["hex"].y.cpu().numpy())

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


hexes["error"] = hexes.apply(create_error_column, axis=1)
max_accidents = hexes["accidents_count"].max()
mean_accidents = hexes["accidents_count"].mean()


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


gdf_accidents = gpd.read_parquet(ACCIDENTS_LOCATION)


map = gdf_accidents.loc[
    gdf_accidents["mie_nazwa"] == city_value.replace(", Poland", ""), :
].explore(
    tiles="CartoDB positron",
)
map = hexes[
    ["ground_truth", "pred", "pred_proba", "error", "accidents_count", "geometry"]
].explore(
    m=map,
    column="error",
    legend=True,
    style_kwds=dict(
        opacity=0.6,
        fillOpacity=0.6,
        style_function=lambda feature: dict(
            fillColor=cmap_fn(feature),
            color=cmap_fn(feature),
        ),
    ),
    categorical=True,
)

TP_accidents = hexes.loc[hexes["error"] == "TP", "accidents_count"].sum()
FN_accidents = hexes.loc[hexes["error"] == "FN", "accidents_count"].sum()

correctly_predicted_accidents_ratio = (TP_accidents) / (TP_accidents + FN_accidents)

st.metric(
    label="Percent of correctly predicted accidents",
    value=f"{correctly_predicted_accidents_ratio*100:.2f}%",
)


with st.spinner("Loading map..."):
    st.header("Results map:")
    st_folium(map, returned_objects=[], use_container_width=True, return_on_hover=False)
