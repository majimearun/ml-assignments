import numpy as np


class LogReg:
    def __init__(self, threshold: float = 0.5):
        self.w: np.ndarray = None
        self.b: float = None
        self.threshold: float = threshold

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return -1 * np.mean(
            [
                y[i] * np.log(self._sigmoid(X[i] @ self.w + self.b) + 1e-15)
                + (1 - y[i]) * np.log(1 - self._sigmoid(X[i] @ self.w + self.b) + 1e-15)
                for i in range(len(X))
            ]
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        lr: float = 0.1,
        epochs: int = 1000,
        descent: str = "batch",
        batch_size: int = 32,
    ) -> tuple[list[float], list[float]]:
        """Fits the model to the given data

        Args:
            X_train (np.ndarray): Data to fit the model to
            y_train (np.ndarray): Labels for the data
            X_test (np.ndarray): Data to test the model on
            y_test (np.ndarray): Labels for the test data
            lr (float, optional): Learning rate. Defaults to 0.1.
            epochs (int, optional): Number of epochs to train for. Defaults to 1000.

        Returns:
            tuple[list[float], list[float]]: Training and test losses
        """

        self.w = np.zeros(X_train.shape[1])
        self.b = 0
        train_losses = []
        test_losses = []
        for _ in range(epochs):
            if descent == "batch":
                dw = np.zeros(X_train.shape[1])
                db = 0
                for i in range(len(X_train)):
                    x = X_train[i]
                    y_hat = self._sigmoid(x @ self.w + self.b)
                    dw += (y_hat - y_train[i]) * x
                    db += y_hat - y_train[i]
                self.w -= lr * dw / len(X_train)
                self.b -= lr * db / len(X_train)
            elif descent == "stochastic":
                for i in range(len(X_train)):
                    x = X_train[i]
                    y_hat = self._sigmoid(x @ self.w + self.b)
                    self.w -= lr * (y_hat - y_train[i]) * x
                    self.b -= lr * (y_hat - y_train[i])

            elif descent == "mini-batch":
                for i in range(0, len(X_train), batch_size):
                    dw = np.zeros(X_train.shape[1])
                    db = 0
                    for j in range(i, i + batch_size):
                        if j >= len(X_train):
                            break
                        x = X_train[j]
                        y_hat = self._sigmoid(x @ self.w + self.b)
                        dw += (y_hat - y_train[j]) * x
                        db += y_hat - y_train[j]
                    self.w -= lr * dw / batch_size
                    self.b -= lr * db / batch_size

            train_losses.append(self.loss(X_train, y_train))
            test_losses.append(self.loss(X_test, y_test))

        return train_losses, test_losses

    def predict(self, X: np.ndarray, proba: bool = False) -> np.ndarray:
        """Predicts the labels for the given data

        Args:
            X (np.ndarray): Data to predict the labels for

        Returns:
            np.ndarray: Predicted labels
        """
        if not proba:
            return np.array(
                [
                    1 if self._sigmoid(x @ self.w + self.b) >= self.threshold else 0
                    for x in X
                ]
            )
        else:
            return np.array([self._sigmoid(x @ self.w + self.b) for x in X])

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
