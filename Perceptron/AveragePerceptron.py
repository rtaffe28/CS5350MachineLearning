import numpy as np
import pandas as pd
np.random.seed(42)

class AveragedPerceptron:
    def __init__(self, T, learning_rate=1):
        self.T = T
        self.learning_rate = learning_rate
        self.weights = None
        self.average_weights = None
    
    def predict(self, x):
        x_with_bias = np.append(x, 1)
        return np.sign(np.dot(self.average_weights, x_with_bias))
    
    def train(self, X, y):
        n_samples, n_features = X.shape
        
        X_with_bias = np.c_[X, np.ones(n_samples)]
        
        self.weights = np.zeros(n_features + 1)
        self.average_weights = np.zeros(n_features + 1)

        for epoch in range(self.T):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]
            
            for i, x_i in enumerate(X_shuffled):
                if y_shuffled[i] * np.dot(self.weights, x_i) <= 0:
                    self.weights += self.learning_rate * y_shuffled[i] * x_i
                self.average_weights += self.weights
