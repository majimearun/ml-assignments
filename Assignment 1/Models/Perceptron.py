import numpy as np


class Perceptron:
    def __init__(self):
        self.w: np.ndarray = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        inf_loop: bool = True,
        epochs: int = 1000,
        delta=0.001,
        n_threshold=100,
    ):
        """Fits the model to the given data

        Args:
            X (np.ndarray): Data to fit the model to
            y (np.ndarray): Labels for the data
            epochs (int, optional): Number of epochs to train for. Defaults to 1000.

        """

        self.w = np.zeros(X.shape[1])
        n_accurate = 0
        prev_n_accurate = 0
        not_change_count = 0
        if inf_loop:
            while True:
                for i in range(len(X)):
                    x = X[i]
                    if y[i] * (x @ self.w) <= 0:
                        self.w += y[i] * x
                    else:
                        n_accurate += 1
                if n_accurate == len(X):
                    print("Breaking as learnt perfect decision boundary")
                    break
                if np.abs(n_accurate - prev_n_accurate) / len(X) < delta:
                    not_change_count += 1
                if not_change_count == n_threshold:
                    print("Breaking as not improving")
                    break
                prev_n_accurate = n_accurate
                n_accurate = 0

        else:
            for _ in range(epochs):
                for i in range(len(X)):
                    x = X[i]
                    if y[i] * (x @ self.w) <= 0:
                        self.w += y[i] * x
                    else:
                        n_accurate += 1
                if n_accurate == len(X):
                    break
                n_accurate = 0

    def _classify(self, x: np.ndarray):
        if x @ self.w > 0:
            return 1
        else:
            return -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the labels for the given data

        Args:
            X (np.ndarray): Data to predict the labels for

        Returns:
            np.ndarray: Predicted labels
        """
        return [self._classify(x) for x in X]

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
            [1 for i in range(len(predicted)) if predicted[i] == -1 and y[i] == -1]
        )
        false_positives = np.sum(
            [1 for i in range(len(predicted)) if predicted[i] == 1 and y[i] == -1]
        )
        false_negatives = np.sum(
            [1 for i in range(len(predicted)) if predicted[i] == -1 and y[i] == 1]
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
