from sklearn.metrics import f1_score, roc_auc_score
import streamlit as st
import pandas as pd
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
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler


st.set_page_config(layout="wide", page_title="Main page")

ZABKA_SHOPS_LOCATION = "./data/downstream_tasks/zabka_shops/accidents.csv"
GRAPH_DATA_DICT_PATH = "./data/results_showcase/accidents/data.pkl"
MODEL_PATH = "./data/results_showcase/zabka_shops/model.ckpt"
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
y = (
pd.concat(
    [
        m["y"]
        for key, m in data.items()
        if key != city_value
    ]
)
.to_numpy()
.ravel()
)

X = scaler.fit_transform(X)
test_X = data[city_value]["X"].to_numpy()
test_X = scaler.transform(test_X)
test_y = data[city_value]["y"].to_numpy().ravel()
y_pred = model.predict(test_X)
y_proba = model.predict_proba(test_X)[:, 1]

auc = roc_auc_score(test_y, y_proba, average="micro")
accuracy = (y_pred == test_y).mean()
f1 = f1_score(
    test_y,
    y_pred,
    pos_label=1,
    average="binary",
)
st.header("Results:")
st.table(
    {
        "F1 score": f1_score(
            test_y,
            y_pred,
            pos_label=1,
            average="binary",
        ),
        "AUC": roc_auc_score(
            test_y,
            response[:, 1].cpu().numpy(),
            average="micro",
        ),
        "Accuracy": (response.argmax(dim=-1) == test_data["hex"].y).sum().item()
        / len(test_data["hex"].y),
    },
)