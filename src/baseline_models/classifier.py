from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib

class CustomClassifier:
    def __init__(self, data: pd.DataFrame, target_column: str, model=None):
        self.data = data
        self.target_column = target_column
        self.clf_model = model if model else LogisticRegression()
        self.X = None
        self.y = None

    def _prepare_data(self):
        self.X = self.data.loc[:, self.data.columns != self.target_column]
        self.y = self.data.loc[:, self.data.columns == self.target_column]

    def split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self._prepare_data()
        return train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.clf_model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.clf_model.predict(X_test)

    def results(self, y_test: np.ndarray, y_pred: np.ndarray):
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=self.clf_model.classes_, ax=ax)
        fig.suptitle("Confusion Matrix for the Baseline Classifier")
        plt.show()
        print(classification_report(y_test, y_pred, output_dict=False))

    def visualize_tsne(self, data: pd.DataFrame = None, n_iter: int = 500) -> None:
        tsne = TSNE(n_iter=n_iter)
        ts = tsne.fit_transform(self.X if data is None else data)

        flattened_y = self.data[self.target_column].astype('category').cat.codes

        cmap = matplotlib.colormaps.get_cmap('viridis')
        colors_array = np.arange(0, np.unique(flattened_y).shape[0])
        colors_array = colors_array / colors_array.max()
        colors_array = cmap(colors_array)

        fig, ax = plt.subplots()
        sc = ax.scatter(ts[:, 0], ts[:, 1], cmap="viridis", c=flattened_y)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        fig.colorbar(sc, ax=ax, label=self.target_column)

        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=10,
                                  label=f'Class {x}') for x, col in enumerate(colors_array)]

        plt.legend(handles=legend_elements)
        fig.suptitle('t-SNE 2D plot of data points')
        fig.set_size_inches(10, 6)
        plt.show()
