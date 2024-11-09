import numpy as np
import pandas as pd
np.random.seed(42)

class Perceptron:
    def __init__(self, T, learning_rate=1):
        self.T = T 
        self.learning_rate = learning_rate
        self.weights = None
    
    def predict(self, x):
        x_with_bias = np.append(x, 1)
        return np.sign(np.dot(x_with_bias, self.weights))
    
    def train(self, X, y):
        n_samples, n_features = X.shape
        X_with_bias = np.c_[X, np.ones(n_samples)]
        self.weights = np.zeros(n_features + 1)
        
        for epoch in range(self.T):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]
            
            for idx, x_i in enumerate(X_shuffled):
                if y_shuffled[idx] * np.dot(x_i, self.weights) <= 0:
                    self.weights += self.learning_rate * y_shuffled[idx] * x_i
            
