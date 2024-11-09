
import numpy as np
import pandas as pd
np.random.seed(42)

class VotedPerceptron:
    def __init__(self, T, learning_rate=1):
        self.T = T
        self.learning_rate = learning_rate
        self.weight_vectors = []
    
    def predict(self, x):
        x_with_bias = np.append(x, 1)
        vote_sum = 0
        for w, c in self.weight_vectors:
            vote_sum += c * np.sign(np.dot(w, x_with_bias))
        return np.sign(vote_sum)
    
    def train(self, X, y):
        n_samples, n_features = X.shape
        w = np.zeros(n_features+1)
        X_with_bias = np.c_[X, np.ones(n_samples)]
        count = 0

        for epoch in range(self.T):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]
            
            for i, x_i in enumerate(X_shuffled):
                if y_shuffled[i] * np.dot(w,x_i) <= 0:
                    if count > 0:
                        self.weight_vectors.append((w.copy(), count))
                    
                    w = w + self.learning_rate * y_shuffled[i] * x_i
                    count = 1
                else:
                    count += 1
        
        if count > 0:
            self.weight_vectors.append((w, count))