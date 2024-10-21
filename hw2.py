import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from EnsembleLearning.AdaBoost import AdaBoost
from EnsembleLearning.Bagging import Bagging
from EnsembleLearning.RandomForest import RandomForest
from LinearRegression import GradientDescent
from joblib import Parallel, delayed


train = pd.read_csv('bank/train.csv')
test = pd.read_csv('bank/test.csv')
X_train = train.drop(columns=['y'])
y_train = train['y']
y_train.replace('yes', 1, inplace=True)
y_train.replace('no', -1, inplace=True)

X_test = test.drop(columns=['y'])
y_test = test['y']
y_test.replace('yes', 1, inplace=True)
y_test.replace('no', -1, inplace=True)
accuracy_history = []

for i in range(1,501):
    tree = Bagging(T=i)
    tree.fit(X_train, y_train)
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)

    train_correct = (train_pred == y_train).sum()
    train_total = len(y_train)
    train_accuracy = train_correct / train_total
    print(f"Train Accuracy with a T={i}: {train_accuracy:.4f}")

    test_correct = (test_pred == y_test).sum()
    test_total = len(y_test)
    test_accuracy = test_correct / test_total
    accuracy_history.append((train_accuracy, test_accuracy))


    print(f"Test Accuracy with T={i}: {test_accuracy:.4f}")
plt.plot(range(len(accuracy_history)), accuracy_history)
plt.ylim(0, 1)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy function over iterations')
plt.show()



n_repeats = 10
n_trees = 100
n_samples = 1000

bias_single_tree = []
variance_single_tree = []
error_single_tree = []

bias_bagged_trees = []
variance_bagged_trees = []
error_bagged_trees = []

def run_iteration(i, X_train, y_train, X_test, y_test, n_samples, n_trees):
    print(i)
    sample_indices = np.random.choice(X_train.index, size=n_samples, replace=False)
    X_sample = X_train.loc[sample_indices]
    y_sample = y_train.loc[sample_indices]

    bagged_model = Bagging(T=n_trees)
    bagged_model.fit(X_sample, y_sample)

    single_tree_pred = bagged_model.trees[0].predict(X_test)

    bagged_predictions = np.array([tree.predict(X_test) for tree in bagged_model.trees])
    average_bagged_pred = np.mean(bagged_predictions, axis=0)

    bias_single = np.mean((single_tree_pred - y_test) ** 2)
    variance_single = np.var(single_tree_pred)
    general_error_single = bias_single + variance_single

    bias_bagged = np.mean((average_bagged_pred - y_test) ** 2)
    variance_bagged = np.var(average_bagged_pred)
    general_error_bagged = bias_bagged + variance_bagged

    return bias_single, variance_single, general_error_single, bias_bagged, variance_bagged, general_error_bagged

results = Parallel(n_jobs=-1)(delayed(run_iteration)(i, X_train, y_train, X_test, y_test, n_samples, n_trees) for i in range(n_repeats))

for res in results:
    bias_single, variance_single, general_error_single, bias_bagged, variance_bagged, general_error_bagged = res
    
    bias_single_tree.append(bias_single)
    variance_single_tree.append(variance_single)
    error_single_tree.append(general_error_single)
    
    bias_bagged_trees.append(bias_bagged)
    variance_bagged_trees.append(variance_bagged)
    error_bagged_trees.append(general_error_bagged)

avg_bias_single = np.mean(bias_single_tree)
avg_variance_single = np.mean(variance_single_tree)
avg_error_single = np.mean(error_single_tree)

avg_bias_bagged = np.mean(bias_bagged_trees)
avg_variance_bagged = np.mean(variance_bagged_trees)
avg_error_bagged = np.mean(error_bagged_trees)

print(f"Single Tree - Bias: {avg_bias_single:.4f}, Variance: {avg_variance_single:.4f}, Error: {avg_error_single:.4f}")
print(f"Bagged Trees - Bias: {avg_bias_bagged:.4f}, Variance: {avg_variance_bagged:.4f}, Error: {avg_error_bagged:.4f}")

labels = ['Bias', 'Variance', 'Error']
single_tree_metrics = [avg_bias_single, avg_variance_single, avg_error_single]
bagged_tree_metrics = [avg_bias_bagged, avg_variance_bagged, avg_error_bagged]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, single_tree_metrics, width, label='Single Tree')
rects2 = ax.bar(x + width/2, bagged_tree_metrics, width, label='Bagged Trees')

ax.set_ylabel('Metrics')
ax.set_title('Bias, Variance, and Error: Single Tree vs. Bagged Trees')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()




def train_evaluate_tree(i):
    print(i)
    tree = RandomForest(T=i, subset_size=4)
    tree.fit(X_train, y_train)
    
    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)

    train_correct = (train_pred == y_train).sum()
    train_total = len(y_train)
    train_accuracy = train_correct / train_total

    test_correct = (test_pred == y_test).sum()
    test_total = len(y_test)
    test_accuracy = test_correct / test_total
    
    return train_accuracy, test_accuracy

n_jobs = -1 
results = Parallel(n_jobs=n_jobs)(delayed(train_evaluate_tree)(i) for i in range(1, 10))

accuracy_history = list(results)

train_accuracies, test_accuracies = zip(*accuracy_history)

plt.plot(range(1, 10), train_accuracies, label='Train Accuracy')
plt.plot(range(1, 10), test_accuracies, label='Test Accuracy')
plt.ylim(0, 1)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy function over iterations')
plt.legend()
plt.show()

n_repeats = 10
n_trees = 100
n_samples = 1000

bias_single_tree = []
variance_single_tree = []
error_single_tree = []

bias_bagged_trees = []
variance_bagged_trees = []
error_bagged_trees = []

def run_iteration(_, X_train, y_train, X_test, y_test, n_samples, n_trees):
    print(_)
    sample_indices = np.random.choice(X_train.index, size=n_samples, replace=False)
    X_sample = X_train.loc[sample_indices]
    y_sample = y_train.loc[sample_indices]

    bagged_model = RandomForest(T=n_trees, subset_size=4)
    bagged_model.fit(X_sample, y_sample)

    single_tree_pred = bagged_model.trees[0].predict(X_test)

    bagged_predictions = np.array([tree.predict(X_test) for tree in bagged_model.trees])
    average_bagged_pred = np.mean(bagged_predictions, axis=0)

    bias_single = np.mean((single_tree_pred - y_test) ** 2)
    variance_single = np.var(single_tree_pred)
    general_error_single = bias_single + variance_single

    bias_bagged = np.mean((average_bagged_pred - y_test) ** 2)
    variance_bagged = np.var(average_bagged_pred)
    general_error_bagged = bias_bagged + variance_bagged

    return bias_single, variance_single, general_error_single, bias_bagged, variance_bagged, general_error_bagged

results = Parallel(n_jobs=-1)(delayed(run_iteration)(_, X_train, y_train, X_test, y_test, n_samples, n_trees) for _ in range(n_repeats))

for res in results:
    bias_single, variance_single, general_error_single, bias_bagged, variance_bagged, general_error_bagged = res
    
    bias_single_tree.append(bias_single)
    variance_single_tree.append(variance_single)
    error_single_tree.append(general_error_single)
    
    bias_bagged_trees.append(bias_bagged)
    variance_bagged_trees.append(variance_bagged)
    error_bagged_trees.append(general_error_bagged)

avg_bias_single = np.mean(bias_single_tree)
avg_variance_single = np.mean(variance_single_tree)
avg_error_single = np.mean(error_single_tree)

avg_bias_bagged = np.mean(bias_bagged_trees)
avg_variance_bagged = np.mean(variance_bagged_trees)
avg_error_bagged = np.mean(error_bagged_trees)

print(f"Single Tree - Bias: {avg_bias_single:.4f}, Variance: {avg_variance_single:.4f}, Error: {avg_error_single:.4f}")
print(f"Bagged Trees - Bias: {avg_bias_bagged:.4f}, Variance: {avg_variance_bagged:.4f}, Error: {avg_error_bagged:.4f}")

labels = ['Bias', 'Variance', 'Error']
single_tree_metrics = [avg_bias_single, avg_variance_single, avg_error_single]
bagged_tree_metrics = [avg_bias_bagged, avg_variance_bagged, avg_error_bagged]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, single_tree_metrics, width, label='Single Tree')
rects2 = ax.bar(x + width/2, bagged_tree_metrics, width, label='Bagged Trees')

ax.set_ylabel('Metrics')
ax.set_title('Bias, Variance, and Error: Single Tree vs. Bagged Trees')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()


GradientDescent.batch_gradient_descent()
print()
GradientDescent.stochastic_gradient_descent()
print()
GradientDescent.analytic_equation()