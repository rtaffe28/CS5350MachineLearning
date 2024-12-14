import numpy as np
import pandas as pd
from NeuralNetworks.neuralnet import NeuralNetwork
from NeuralNetworks.torchnn import TorchNeuralNetwork

train_data = pd.read_csv("bank-note/train.csv", header=None)
test_data = pd.read_csv("bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

for w in [5,10,25,50,100]:
    model = NeuralNetwork(4, w)

    model.train(X_train, y_train)

    train_predictions = [model.predict(x) for x in X_train]
    test_predictions =  [model.predict(x) for x in X_test]


    accuracy = np.mean(train_predictions == y_train)
    train_error = 1 - accuracy
    accuracy = np.mean(test_predictions == y_test)
    test_error = 1 - accuracy

    print(f"train error for width {w}: {train_error}")
    print(f'test error for width {w}:  {test_error}')

depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activation_functions = ['tanh', 'relu']

for a in activation_functions:
    for d in depths:
        for w in widths:
            model = TorchNeuralNetwork(4, d, w, a)
            model.train_model(X_train, y_train)

            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            train_accuracy = np.mean(train_predictions.cpu().numpy() == y_train)
            train_error = 1 - train_accuracy
            test_accuracy = np.mean(test_predictions.cpu().numpy() == y_test)
            test_error = 1 - test_accuracy

            print(f"Train error for width {w}, depth {d}, activation_fn {a}: {train_error}")
            print(f"Test error for width {w}, depth {d}, activation_fn {a}: {test_error}")

