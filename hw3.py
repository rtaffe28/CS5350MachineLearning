import numpy as np
import pandas as pd
from Perceptron.Perceptron import Perceptron  
from Perceptron.VotedPerceptron import VotedPerceptron  
from Perceptron.AveragePerceptron import AveragedPerceptron

train_data = pd.read_csv("bank-note/train.csv", header=None)
test_data = pd.read_csv("bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
y_train = np.where(y_train == 0, -1, 1)

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values
y_test = np.where(y_test == 0, -1, 1)


print('Part A)')
errors = []
for T in range(1,11):
    perceptron = Perceptron(T=T, learning_rate=1)
    perceptron.train(X_train, y_train)

    print(f"Epoch {T} Learned weight vector:", perceptron.weights)

    predictions = np.array([perceptron.predict(x) for x in X_test])

    accuracy = np.mean(predictions == y_test)
    test_error = 1 - accuracy

    errors.append(test_error)

print(f"average error: {sum(errors)/10}")
print(f'final error {errors[-1]}')


print()
print('Part B)')
errors = []
for T in range(1,11):
    perceptron = VotedPerceptron(T=T, learning_rate=1)
    perceptron.train(X_train, y_train)

    print(f"Epoch {T} Learned weight vector:", perceptron.weight_vectors[-1])

    predictions = np.array([perceptron.predict(x) for x in X_test])

    accuracy = np.mean(predictions == y_test)
    test_error = 1 - accuracy

    errors.append(test_error)

print(f"average error: {sum(errors)/10}")
print(f'final error {errors[-1]}')


print()
print('Part C)')
errors = []
for T in range(1,11):
    perceptron = AveragedPerceptron(T=T, learning_rate=1)
    perceptron.train(X_train, y_train)

    print(f"Epoch {T} Learned weight vector:", perceptron.weights)

    predictions = np.array([perceptron.predict(x) for x in X_test])

    accuracy = np.mean(predictions == y_test)
    test_error = 1 - accuracy

    errors.append(test_error)

print(f"average error: {sum(errors)/10}")
print(f'final error {errors[-1]}')