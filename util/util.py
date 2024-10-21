import numpy as np
import pandas as pd
from collections import defaultdict

def entropy(Label):
    probs = Label.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-10))

#returns information gain given a attribute and a label
def information_gain(attribute, Label):
    parent_entropy = entropy(Label)
    
    weighted_entropy = 0
    for value in attribute.unique():
        subset = Label[attribute == value]
        weighted_entropy += (len(subset) / len(Label)) * entropy(subset)
    
    return parent_entropy - weighted_entropy

#returns information gain given a attribute and a label
def majority_error(attribute, Label):
    parent_me = 1 - ((Label == Label.mode()[0]).sum()/len(Label))

    weighted_me = 0
    for value in attribute.unique():
        subset = Label[attribute == value]
        error_rate = 1 - ((subset == subset.mode()[0]).sum()/len(subset))
        weighted_me += error_rate* (len(subset)/len(attribute))
    return parent_me - weighted_me

#returns information gain given a attribute and a label
def gini_index(attribute, Label):
    parent_gi = 1 - sum([v**2 for v in Label.value_counts(normalize=True)])

    weighted_gi = 0
    for value in attribute.unique():
        subset = Label[attribute == value]
        error_rate = 1 - sum([v**2 for v in subset.value_counts(normalize=True)])
        weighted_gi += error_rate* (len(subset)/len(attribute))
    return parent_gi - weighted_gi

def entropy_weighted(labels, weights):
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0 
    weights = weights.tolist()
    label_weights = defaultdict(float)

    for i in range(len(labels)):
        label = labels.iloc[i]
        label_weights[label] += weights[i]
    
    probs = np.array(list(label_weights.values())) / total_weight
    return -np.sum(probs * np.log2(probs + 1e-10))


def information_gain_weighted(attribute, Label, weights):
    parent_entropy = entropy_weighted(Label, weights)
    weighted_entropy = 0
    total_weight = np.sum(weights)
    
    for value in attribute.unique():
        subset_mask = (attribute == value)
        subset_weights = weights[subset_mask]
        subset_labels = Label[subset_mask]
        weight_fraction = np.sum(subset_weights) / total_weight

        weighted_entropy += weight_fraction * entropy_weighted(subset_labels, subset_weights)

    return parent_entropy - weighted_entropy
