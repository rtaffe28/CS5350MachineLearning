import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('concrete/train.csv')
test_data = pd.read_csv('concrete/test.csv')

X_train = train_data.iloc[:, :-1].values 
y_train = train_data.iloc[:, -1].values 
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def cost_function(X, y, w):
        predictions = X.dot(w)
        cost = (1 / 2) * np.sum((y-predictions) ** 2)
        return cost

def batch_gradient_descent():
    w = np.zeros(X_train.shape[1]) 
    r = 0.5 
    tolerance = 1e-6
    cost_history = []

    i = 0
    while True:
        predictions = X_train.dot(w)
        
        errors = y_train - predictions
        
        gradient_w = - (1 / len(y_train)) * X_train.T.dot(errors)

        new_w = w - (r * gradient_w)
        
        cost = cost_function(X_train, y_train, w)
        cost_history.append(cost)
        
        if np.linalg.norm(w - new_w) < tolerance:
            print(f"Converged after {i} iterations.")
            break
        w = new_w
        i += 1

    print(f"Final weight vector: {w}")
    print(f"Learning rate: {r}")

    # plt.plot(range(len(cost_history[:20])), cost_history[:20])
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.title('Cost function over iterations')
    # plt.show()

    test_cost = cost_function(X_test, y_test, w)
    print(f"Test set cost: {test_cost}")

def stochastic_gradient_descent():
    w = np.zeros(X_train.shape[1])
    max_iterations = 20000
    r = 0.03
    cost_history = []

    for t in range(max_iterations):
        for i in range(len(X_train)):
            xi = X_train[i]
            yi = y_train[i]

            learning_rate = r / (1 + t *0.001)

            for j in range(len(w)):
                w[j] += learning_rate * (yi - np.dot(w, xi)) * xi[j]
            
            cost = cost_function(X_train, y_train, w)
            cost_history.append( cost)

    print(f"Final weight vector: {w}")
    print(f"Learning rate: {r}")

    test_cost = cost_function(X_test, y_test, w)
    print(f"Test set cost: {test_cost}")

    # plt.plot(range(len(cost_history[:1000])), cost_history[:1000])
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.title('Cost function over iterations')
    plt.show()

    

def analytic_equation():
    train_data = pd.read_csv('concrete/train.csv')

    X_train = train_data.iloc[:, :-1].values 
    y_train = train_data.iloc[:, -1].values

    w_opt = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

    print(f"Optimal weight vector (Analytic Equation): {w_opt}")