import numpy as np


class LogReg:
    def __init__(self, threshold: float = 0.5):
        self.w: np.ndarray = None
        self.b: float = None
        self.threshold: float = threshold

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1000):
        """Fits the model to the given data

        Args:
            X (np.ndarray): Data to fit the model to
            y (np.ndarray): Labels for the data
            lr (float, optional): Learning rate. Defaults to 0.1.
            epochs (int, optional): Number of epochs to train for. Defaults to 1000.
        """

        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(epochs):
            for i in range(len(X)):
                x = X[i]
                y_hat = self._sigmoid(x @ self.w + self.b)
                self.w -= lr * (y_hat - y[i]) * x
                self.b -= lr * (y_hat - y[i])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the labels for the given data

        Args:
            X (np.ndarray): Data to predict the labels for

        Returns:
            np.ndarray: Predicted labels
        """
        return np.array(
            [
                1 if self._sigmoid(x @ self.w + self.b) >= self.threshold else 0
                for x in X
            ]
        )

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function

        Args:
            x (float): Input for sigmoid function

        Returns:
            float: sigmoid of the input
        """
        return 1 / (1 + np.exp(-x))

    def score(
        self, X: np.ndarray, y: np.ndarray, _print: bool = False
    ) -> tuple[float, float, float, float]:
        """Scores the model by calculating and printing the accuracy, precisions and recalls on the given data

        Args:
            X (np.ndarray): Data to score the model on
            y (np.ndarray): Labels for the data
            _print (bool, optional): Whether to print the scores. Defaults to False.

        Returns:
            tuple[float, float, float, float]: True positives, true negatives, false positives and false negatives
        """

        predicted = self.predict(X)
        true_positives = np.sum(
            [1 for i in range(len(predicted)) if predicted[i] == 1 and y[i] == 1]
        )
        true_negatives = np.sum(
            [1 for i in range(len(predicted)) if predicted[i] == 0 and y[i] == 0]
        )
        false_positives = np.sum(
            [1 for i in range(len(predicted)) if predicted[i] == 1 and y[i] == 0]
        )
        false_negatives = np.sum(
            [1 for i in range(len(predicted)) if predicted[i] == 0 and y[i] == 1]
        )

        if _print:
            print("--------Results--------")
            print(f"Accuracy: {(true_positives + true_negatives)/len(predicted)}")
            print("----Class 1----")
            print(f"Precision: {true_positives/(true_positives + false_positives)}")
            print(f"Recall: {true_positives/(true_positives + false_negatives)}")
            print("----Class 0----")
            print(f"Precision: {true_negatives/(true_negatives + false_negatives)}")
            print(f"Recall: {true_negatives/(true_negatives + false_positives)}")

        return true_positives, true_negatives, false_positives, false_negatives
