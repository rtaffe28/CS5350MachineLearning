import numpy as np
import pandas as pd
np.random.seed(42)

class SVM_primal:
    def __init__(self, T, learning_rate=1, C=1, a=None):
        self.T = T 
        self.learning_rate = learning_rate
        self.C = C 
        self.a = a
        self.weights = None
        self.hinge_losses = []
    
    def hinge_loss(self, w, X, y):
        n = len(y)
        loss = 0.5 * np.dot(w, w) + self.C * np.sum(np.maximum(0, 1 - y * (X.dot(w))))
        return loss

    def train(self, X, y):
        n_samples, n_features = X.shape
        X_with_bias = np.c_[X, np.ones(n_samples)]
        self.weights = np.zeros(n_features + 1)
        t = 1

        for epoch in range(self.T):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]
            
            for i, (x_i, y_i) in enumerate(zip(X_shuffled, y_shuffled)):
                if self.a is None:
                    gamma_t = self.learning_rate / (1 + t)
                else:
                    gamma_t = self.learning_rate / (1 + (t*(self.learning_rate/self.a)))
                
                if y_i * np.dot(self.weights, x_i) <= 1:
                    self.weights = self.weights - (gamma_t * self.weights) + (gamma_t * self.C * n_samples * y_i * x_i)
                else:
                    self.weights = (1 - gamma_t) * self.weights
                t+=1
            
            epoch_loss = self.hinge_loss(self.weights, X_with_bias, y)
            self.hinge_losses.append(epoch_loss)
    
    def predict(self, x):
        x_with_bias = np.append(x, 1)
        return np.sign(x_with_bias.dot(self.weights))
