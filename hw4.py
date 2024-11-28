import numpy as np
import pandas as pd
from SVM.SVM_primal import SVM_primal
from SVM.SVM_dual import SVM_dual
import matplotlib.pyplot as plt


train_data = pd.read_csv("bank-note/train.csv", header=None)
test_data = pd.read_csv("bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
y_train = np.where(y_train == 0, -1, 1)

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values
y_test = np.where(y_test == 0, -1, 1)


print('Problem 2)')
Cs = [100/873, 500/873, 700/873]
print('Part a)')
for c in Cs:
    errors = []

    SVM_model = SVM_primal(T=100, learning_rate=1, C=c, a=10)
    SVM_model.train(X_train, y_train)

    train_predictions = np.array([SVM_model.predict(x) for x in X_train])

    accuracy = np.mean(train_predictions == y_train)
    train_error = 1 - accuracy

    test_predictions = np.array([SVM_model.predict(x) for x in X_test])

    accuracy = np.mean(test_predictions == y_test)
    test_error = 1 - accuracy

    # plt.plot(SVM_model.hinge_losses, label='Hinge Loss', color='blue', linewidth=2)
    # plt.xlabel('Iterations', fontsize=14)
    # plt.ylabel('Hinge Loss', fontsize=14)
    # plt.title('Hinge Loss Over Iterations', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid(True)
    # plt.show()
    print(f"train error for C of {c}: {train_error}")
    print(f'test error for C of {c}:  {test_error}')


print('Part b)')
for c in Cs:
    errors = []

    SVM_model = SVM_primal(T=100, learning_rate=1, C=c)
    SVM_model.train(X_train, y_train)

    train_predictions = np.array([SVM_model.predict(x) for x in X_train])

    accuracy = np.mean(train_predictions == y_train)
    train_error = 1 - accuracy

    test_predictions = np.array([SVM_model.predict(x) for x in X_test])

    accuracy = np.mean(test_predictions == y_test)
    test_error = 1 - accuracy

    # plt.plot(SVM_model.hinge_losses, label='Hinge Loss', color='blue', linewidth=2)
    # plt.xlabel('Iterations', fontsize=14)
    # plt.ylabel('Hinge Loss', fontsize=14)
    # plt.title('Hinge Loss Over Iterations', fontsize=16)
    # plt.legend(fontsize=12)
    # plt.grid(True)
    # plt.show()
    print(f"train error for C of {c}: {train_error}")
    print(f'test error for C of {c}:  {test_error}')

print('Problem 3)')
print('Part a)')
for c in Cs:
    errors = []

    SVM_model = SVM_dual(C=c, kernel='l')
    SVM_model.train(X_train, y_train)
    
    train_predictions = np.array([SVM_model.predict(x) for x in X_train])

    accuracy = np.mean(train_predictions == y_train)
    train_error = 1 - accuracy

    test_predictions = np.array([SVM_model.predict(x) for x in X_test])

    accuracy = np.mean(test_predictions == y_test)
    test_error = 1 - accuracy

    print(f"train error for C of {c}: {train_error}")
    print(f'test error for C of {c}:  {test_error}')

print('Part b)')
gammas = [0.1, 0.5, 1, 5, 100]
for c in Cs:
    for g in gammas:
        errors = []

        SVM_model = SVM_dual(C=c, gamma=g)
        SVM_model.train(X_train, y_train)
        
        train_predictions = np.array([SVM_model.predict(x) for x in X_train])

        accuracy = np.mean(train_predictions == y_train)
        train_error = 1 - accuracy

        test_predictions = np.array([SVM_model.predict(x) for x in X_test])

        accuracy = np.mean(test_predictions == y_test)
        test_error = 1 - accuracy

        print(f"train error for C of {c} and a gamma of {g}: {train_error}")
        print(f'test error for C of {c} and a gamma of {g}:  {test_error}')