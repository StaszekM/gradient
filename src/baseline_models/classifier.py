import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator

class CustomClassifier:
    """
    A custom classifier that provides a simple interface for training and evaluating machine learning models.

    """
    def __init__(self, model: BaseEstimator = LogisticRegression()):
        """
        Init CustomClassifier.

        Args:
            model (sklearn.base.BaseEstimator): Scikit-learn compatible classification model (default: LogisticRegression)

        """
        self.clf_model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the classifier on the given training data.

        Args:
            X_train: Input features for training.
            y_train: Target labels for training.
        """
        self.clf_model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given test data.

        Args:
            X_test: Input features for testing.

        Returns:
            Predicted labels for the input test data.
        """
        return self.clf_model.predict(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        Evaluates the classifier's performance using various metrics and displays results.

        Parameters:
            X_test: Input features for testing.
            y_test: True labels for the test data.
            y_pred: Predicted labels for the test data.
        Returns:
            metrics: Dictionary containing performance metrics such as F1 Score, Accuracy, Precision, Recall, and AUC Score (if applicable).
            conf_matrix_disp: Confusion Matrix visualization.
        """
        metrics = {}

        f1 = f1_score(y_test, y_pred, average='weighted')
        metrics['F1 Score'] = f1

        accuracy = accuracy_score(y_test, y_pred)
        metrics['Accuracy'] = accuracy

        precision = precision_score(y_test, y_pred, average='weighted')
        metrics['Precision'] = precision

        recall = recall_score(y_test, y_pred, average='weighted')
        metrics['Recall'] = recall

        if hasattr(self.clf_model, "predict_proba"):
            y_prob = self.clf_model.predict_proba(X_test)
            auc_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average='weighted')
            metrics['AUC Score'] = auc_score

        cm = confusion_matrix(y_test, y_pred, labels=self.clf_model.classes_)
        conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=self.clf_model.classes_)
        
        return metrics, conf_matrix_disp
