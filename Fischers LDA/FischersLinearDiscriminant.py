import numpy as np


class GaussianDistribution:
    
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        
    def find_probability(self, x):
        return np.exp(-((x - self.mean) ** 2) / (2 * (self.std ** 2)))/(np.sqrt (2 * np.pi) * self.std)

class FischersLinearDiscriminant:
    def __init__(self):
        self._w: np.ndarray = None
        self._class1_prior: float = 0
        self._class2_prior: float = 0
        self._class1_gaussian: GaussianDistribution = None
        self._class2_gaussian: GaussianDistribution = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_class1 = [X[i] for i in range(len(X)) if y[i] == 1]
        X_class2 = [X[i] for i in range(len(X)) if y[i] == -1]

        class1_mean = np.mean(X_class1, axis=0)
        class2_mean = np.mean(X_class2, axis=0)
        
        s1_squared = np.mean([(x-class1_mean).reshape(-1, 1) @ (x - class1_mean).reshape(1, -1) for x in X_class1], axis=0)
        s2_squared = np.mean([(x-class1_mean).reshape(-1, 1) @ (x - class1_mean).reshape(1, -1) for x in X_class2], axis=0)
        
        sw = s1_squared + s2_squared
        self._w = np.linalg.inv(sw) @ (class1_mean - class2_mean).reshape(-1, 1)

        
        projected_class1 = np.array([self._project(x) for x in X_class1])
        projected_class2 = np.array([self._project(x) for x in X_class2])
        
        self._class1_gaussian = GaussianDistribution(np.mean(projected_class1), np.std(projected_class1))
        self._class2_gaussian = GaussianDistribution(np.mean(projected_class2), np.std(projected_class2))
        
        self._class1_prior = len(X_class1)/len(X)
        self._class2_prior = len(X_class2)/len(X)
        
        
    
    def _project(self, x):
        return x @ self._w
        
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._classify(x) for x in X])
        
    
    def _classify(self, x: np.ndarray) -> np.ndarray:
        decision = np.log(self._class1_prior) - np.log(self._class2_prior) + np.log(self._class1_gaussian.find_probability(x @ self._w)) - np.log(self._class2_gaussian.find_probability(x @ self._w))
        if decision > 0:
            return 1
        else:
            return -1
        
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predicted = self.predict(X)
        return np.sum(y == predicted)/len(y)
