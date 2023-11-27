import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


class CustomClassifier:
    """
    A custom classifier that provides a simple interface for training and evaluating machine learning models.

    """
    def __init__(self, model=None):
        """
        Init CustomClassifier.

        Args:
            model: Scikit-learn compatible classification model (default: LogisticRegression)

        """
        self.clf_model = model if model else LogisticRegression()

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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Evaluates the classifier's performance using various metrics and displays results.

        Parameters:
            X_test: Input features for testing.
            y_test: True labels for the test data.
            y_pred: Predicted labels for the test data.
        Returns:
            Dictionary containing evaluation metrics (F1 score, accuracy, precision, recall, AUC score).
        """
        metrics = {}

        f1 = f1_score(y_test, y_pred)
        metrics['F1 Score'] = f1

        accuracy = accuracy_score(y_test, y_pred)
        metrics['Accuracy'] = accuracy

        precision = precision_score(y_test, y_pred, average='weighted')
        metrics['Precision'] = precision

        recall = recall_score(y_test, y_pred, average='weighted')
        metrics['Recall'] = recall

        if hasattr(self.clf_model, "predict_proba"):
            y_prob = self.clf_model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
            metrics['AUC Score'] = auc_score

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=self.clf_model.classes_, ax=ax)
        fig.suptitle("Confusion Matrix for the Baseline Classifier")
        plt.show()

        return metrics
