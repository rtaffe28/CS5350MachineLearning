import collections
import numpy as np
import pandas as pd
from EnsembleLearning.DecisionStump import DecisionStump
from DecisionTree import DecisionTree

class AdaBoost:
    def __init__(self, T=500):
        self.T = T
        self.stumps = []
        self.alphas = []
        self.errors = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.full(n_samples, 1 / n_samples)
        for t in range(self.T):
            stump = DecisionStump()
            #print(f"Iteration {t}")
            stump.fit(X, y, weights)
            predictions = stump.predict(X)

            error = 0.5 - 0.5*(np.dot(weights, y*predictions))
            #print(f"Error: {error}")

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            #print(f"Alpha: {alpha}")

            weights *= np.exp(-alpha * y*predictions)
            weights /= np.sum(weights)

            self.stumps.append(stump)
            self.alphas.append(alpha)
            self.errors.append(error)

    def predict(self, X):
        n_samples = X.shape[0]
        final_predictions = np.zeros(n_samples)

        for stump, alpha in zip(self.stumps, self.alphas):
            final_predictions += alpha * stump.predict(X)

        return np.sign(final_predictions)
