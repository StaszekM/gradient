import pandas as pd
from typing import Union, Literal
import json
import os



def get_top_features(task: Union[Literal["accidents"], Literal["airbnb"], Literal['zabka_shops']], n_top_vals: int, percent: bool = True):
    
    """
    Gets top features form feature importance analysis.
    Args:
        - task (Union[Literal["accidents"], Literal["airbnb"], Literal['zabka_shops']]) - supported downstream task name
        - n_top_vals (int) - number of records with highest feature importance
        - percent (bool) - if n_top_values should be taken as percent of all feature importance records (True) or as a number of records (False)

    """
    top_values = {}
    if task == "accidents":
        task_folder = "accidents_prediction"
    else:
        task_folder = task
    all_features = pd.read_csv(f"../../data/downstream_tasks/{task_folder}/{task}_hexes_feature_importance.csv")
    all_features = all_features.sort_values(by='importance', ascending=False)
    if percent:
        top_k = all_features.head(int(len(all_features) * (n_top_vals/100)))['col_name'].tolist()
    else:
        top_k = all_features.head(n_top_vals)['col_name'].tolist()
    
    top_values['top_values'] = top_k
    save_folder = "../../data/downstream_tasks/feature_importance"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(f"{save_folder}/{task}_top_{n_top_vals}_percent_{percent}.json", 'w') as f:
        json.dump(top_values, f)
        



tasks = ['airbnb', 'accidents']
values = [20, 50]
percent = [True, False]

for task in tasks:
    for val in values:
        for p in percent:
            get_top_features(task, val, p)