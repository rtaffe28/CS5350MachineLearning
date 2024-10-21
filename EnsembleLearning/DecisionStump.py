import numpy as np
import pandas as pd
from util import util
from collections import defaultdict

class DecisionStump:
    def __init__(self):
        self.max_depth = 2
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

        if A.empty or (self.max_depth is not None and depth >= self.max_depth):
            return self.most_common_label
        
        feature = self.best_split(A, Label)
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

    def best_split(self, A, Label, weights):
        best_feature = None
        max_gain = -float('inf')

        for feature in A.columns:
            gain = util.information_gain_weighted(A[feature], Label, weights)
            print(f"Feature: {feature}, Information Gain: {gain}")
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        print(best_feature)
        return best_feature
