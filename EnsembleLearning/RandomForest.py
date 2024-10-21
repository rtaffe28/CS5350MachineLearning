import numpy as np
import pandas as pd
from DecisionTree.DecisionTree import DecisionTreeID3
from util import util
from collections import defaultdict
from scipy import stats

class RandForestTree:
    def __init__(self, subset_size=2):
        self.subset_size = subset_size
        self.tree = None
        self.most_common_label = None 
        self.median = None

    def fit(self, A, Label):
        self.most_common_label = Label.mode()[0]
        self.tree = self.build_tree(A, Label)

    def predict(self, X):
        return X.apply(self.predict_row, axis=1)

    def predict_row(self, row):
        node = self.tree
        while isinstance(node, dict):
            feature = list(node.keys())[0]
            value = row[feature]
            if isinstance(value, (int, float)):
                median = node[feature]['median']
                if value <= median:
                    node = node[feature][f'<={median}']
                else:
                    node = node[feature][f'>{median}']
            else:
                if value in node[feature]:
                    node = node[feature][value]
                else:
                    return self.most_common_label
        return node

    def build_tree(self, A, Label, depth=0):
        if len(np.unique(Label)) == 1:
            return Label.iloc[0]

        if A.empty:
            return self.most_common_label
        
        n = min(self.subset_size, A.shape[1])
        random_features = A.sample(n=n, axis=1, random_state=np.random.randint(0, 10000))
        feature = self.best_split(random_features, Label)
        if feature is None:
            return self.most_common_label

        node = defaultdict(defaultdict)
        
        if A[feature].dtype == np.int64:
            median = A[feature].median()
            node[feature]['median'] = median
            node[feature][f'<={median}'] = self.build_tree(A[A[feature] <= median].drop(columns=[feature]), Label[A[feature] <= median], depth + 1)
            node[feature][f'>{median}'] = self.build_tree(A[A[feature] > median].drop(columns=[feature]), Label[A[feature] > median], depth + 1)
        else:
            for value in A[feature].unique():
                subset_v = A[A[feature] == value]
                subset_Label = Label[A[feature] == value]

                if subset_v.empty:
                    node[feature][value] = Label.mode()[0]
                else:
                    remaining_attributes = subset_v.drop(columns=[feature])
                    node[feature][value] = self.build_tree(remaining_attributes, subset_Label, depth + 1)

        return node

    def best_split(self, A, Label):
        ans = [-1, None]

        for feature in A.columns:
            gain = util.information_gain(A[feature], Label)
            if gain > ans[0]:
                ans[0] = gain
                ans[1] = feature

        return ans[1]



class RandomForest:
    def __init__(self, T=500, subset_size=2):
        self.subset_size = subset_size
        self.T = T
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        for t in range(self.T):
            tree = RandForestTree(subset_size=self.subset_size)
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
    
