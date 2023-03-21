import numpy as np


class GaussianDistribution:
    def __init__(self, mean: float, std: float):
        """Gaussian distribution class

        Args:
            mean (float): Mean of the distribution
            std (float): Standard deviation of the distribution
        """
        self.mean = mean
        self.std = std

    def find_probability(self, x):
        """Find the probability of a point being in a distribution

        Args:
            x (float): Point to find the probability of

        Returns:
            float: Probability of the point being in the distribution
        """
        return np.exp(-((x - self.mean) ** 2) / (2 * (self.std**2))) / (
            np.sqrt(2 * np.pi) * self.std
        )


class FischersLinearDiscriminant:
    def __init__(self):
        """Fischers Linear Discriminant Classifier"""
        self.w: np.ndarray = None
        self._class1_prior: float = 0
        self._class2_prior: float = 0
        self._class1_gaussian: GaussianDistribution = None
        self._class2_gaussian: GaussianDistribution = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Finds the line onto which the data can be projected to maximise the separation between the two classes and minimises the variance within each class. Also finds and sets the priors and gaussian distributions for each class.

        Args:
            X (np.ndarray): Data to fit the model to
            y (np.ndarray): Labels for the data
        """
        X_class1 = [X[i] for i in range(len(X)) if y[i] == 1]
        X_class2 = [X[i] for i in range(len(X)) if y[i] == 0]
        class1_mean = np.mean(X_class1, axis=0)
        class2_mean = np.mean(X_class2, axis=0)

        s1_squared = np.sum(
            [
                (x - class1_mean).reshape(-1, 1) @ (x - class1_mean).reshape(1, -1)
                for x in X_class1
            ],
            axis=0,
        )
        s2_squared = np.sum(
            [
                (x - class1_mean).reshape(-1, 1) @ (x - class1_mean).reshape(1, -1)
                for x in X_class2
            ],
            axis=0,
        )

        sw = s1_squared + s2_squared
        self.w = np.linalg.inv(sw) @ (class1_mean - class2_mean).reshape(-1, 1)

        projected_class1 = np.array([self._project(x) for x in X_class1])
        projected_class2 = np.array([self._project(x) for x in X_class2])

        self._class1_gaussian = GaussianDistribution(
            np.mean(projected_class1), np.std(projected_class1)
        )
        self._class2_gaussian = GaussianDistribution(
            np.mean(projected_class2), np.std(projected_class2)
        )

        self._class1_prior = len(X_class1) / len(X)
        self._class2_prior = len(X_class2) / len(X)

    def _project(self, x) -> float:
        """Projects a point onto the line (discriminant) found by the fit method

        Args:
            x (np.ndarray): Point to project onto the line

        Returns:
            float: Projection of the point onto the line
        """
        return x @ self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the classes of a set of points

        Args:
            X (np.ndarray): Points to predict the classes of

        Returns:
            np.ndarray: Predicted classes of the points (1D array)
        """
        return np.array([self._classify(x) for x in X])

    def _classify(self, x: np.ndarray) -> int:
        """Classifies a point as either class 1 or class 0

        Args:
            x (np.ndarray): Point to classify

        Returns:
            int: Class of the point
        """
        decision = (
            np.log(self._class1_prior)
            - np.log(self._class2_prior)
            + np.log(self._class1_gaussian.find_probability(x @ self.w))
            - np.log(self._class2_gaussian.find_probability(x @ self.w))
        )
        if decision > 0:
            return 1
        else:
            return 0

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
