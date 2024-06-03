from typing import cast
from sklearn.linear_model import LogisticRegression
import wandb
from joblib import load
import os
import pandas as pd
import numpy as np


def get_feature_importance():
    api = wandb.Api()
    model_artifact = api.artifact(
        "gradient_pwr/accidents-downstream-task-v2/model_b3fs9h38_fold_1:v0"
    )  # the best model from the hex-only sweep on accidents task (or any other task)
    path_str = model_artifact.download()

    with open(path_str + "/model_b3fs9h38_fold_1.pkl", "rb") as f:
        model: LogisticRegression = cast(LogisticRegression, load(f))

    data_artifact = api.artifact(
        "gradient_pwr/accidents-downstream-task-v2/tabular_data:v2"
    )  # data artifact from the same sweep

    data_path = data_artifact.download()
    with open(data_path + "/tabular_data_25f95thh.pkl", "rb") as f:
        data = load(f)
    train_X = pd.concat([x["X"] for _, x in data.items()])  # just for getting col names

    return pd.DataFrame(
        {
            "col_name": train_X.columns,
            "importance": np.mean(np.abs(model.coef_), axis=0),
        }
    ).sort_values(by="importance", ascending=False)
