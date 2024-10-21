import numpy as np
import pandas as pd
from DecisionTree.DecisionTree import DecisionTreeID3
from scipy import stats

class Bagging:
    def __init__(self, T=500):
        self.T = T
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        while len(self.trees) < self.T:
            tree = DecisionTreeID3()
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            
            tree.fit(X_sample, y_sample)


            self.trees.append(tree)
            
            
    def predict(self, X):
        n_samples = X.shape[0]
        tree_predictions = np.zeros((len(self.trees), n_samples))
        
        for i, tree in enumerate(self.trees):
            tree_predictions[i] = tree.predict(X)
        
        final_predictions = stats.mode(tree_predictions, axis=0)

        return final_predictions.mode
    
