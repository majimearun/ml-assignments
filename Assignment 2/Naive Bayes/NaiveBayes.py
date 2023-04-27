import pandas as pd
import numpy as np
import typing


class Distribution:
    def __init__(self, label: str, prior: float):
        self.label = label
        self.prior = prior
        self.conditionals = dict()

    def add_conditional(self, attribute: str, value: typing.Any, probability: float):
        self.conditionals[(attribute, value)] = probability

    def get_conditional(self, attribute: str, value: typing.Any):
        if self.conditionals.get((attribute, value)) is None:
            return None
        return self.conditionals[(attribute, value)]


class NaiveBayes:
    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target
        self.distributions = dict()
        self.attributes = list(data.columns)
        self.attributes.remove(target)
        self.labels = list(data[target].unique())
        self._train()

    def _train(self, alpha: float = 1e-10):
        # Calculate priors
        for label in self.labels:
            prior = len(self.data[self.data[self.target] == label]) / len(self.data)
            self.distributions[label] = Distribution(label, prior)

        # Calculate conditionals
        for label in self.labels:
            for attribute in self.attributes:
                for value in self.data[attribute].unique():
                    probability = (
                        len(
                            self.data[
                                (self.data[attribute] == value)
                                & (self.data[self.target] == label)
                            ]
                        )
                        + alpha
                    ) / (
                        len(self.data[self.data[self.target] == label])
                        + alpha * len(self.attributes)
                    )
                    self.distributions[label].add_conditional(
                        attribute, value, probability
                    )

    def predict(self, data: pd.DataFrame):
        predictions = list()
        for _, row in data.iterrows():
            max_posterior = 0
            max_label = None
            for label in self.labels:
                posterior = self.distributions[label].prior
                for attribute in self.attributes:
                    if (
                        self.distributions[label].get_conditional(
                            attribute, row[attribute]
                        )
                        is None
                    ):
                        posterior *= 1 / (
                            len(self.data[self.data[self.target] == label])
                            + len(self.attributes)
                        )
                    else:
                        posterior *= self.distributions[label].get_conditional(
                            attribute, row[attribute]
                        )
                if posterior > max_posterior:
                    max_posterior = posterior
                    max_label = label
            predictions.append(max_label)
        return predictions

    def accuracy(self, data: pd.DataFrame):
        predictions = self.predict(data)
        labels = self.data[self.target].unique()
        tp,tn,fp,fn = 0,0,0,0
        for index, row in data.iterrows():
            if row[self.target] == predictions[index] and row[self.target] == labels[0]:
                tp += 1
            if row[self.target] != predictions[index] and row[self.target] == labels[0]:
                fp += 1
            if row[self.target] == predictions[index] and row[self.target] == labels[1]:
                tn += 1
            if row[self.target] != predictions[index] and row[self.target] == labels[1]:
                fn += 1
        return tp,tn,fp,fn
