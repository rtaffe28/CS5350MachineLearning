import numpy as np
from scipy.optimize import minimize

class SVM_dual:
    def __init__(self, C=1.0, kernel='g', gamma = 0.1):
        self.C = C
        self.alphas = None
        self.weights = None
        self.bias = None
        self.kernel=kernel
        self.gamma = gamma
        self.X_train = None
        self.y_train = None

    def linear_kernel(self, X):
        return np.dot(X, X.T)

    def gaussian_kernel(self, X):
        sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[:, None].T - 2 * np.dot(X, X.T)
        return np.exp(-sq_dists/self.gamma)


    def recover_weights(self, X, y, alpha):
        return np.sum(alpha[:, None] * y[:, None] * X, axis=0)

    def compute_bias(self, X, y, alpha):
        support_vectors = np.where((alpha > 1e-5) & (alpha < self.C))[0]
        b = np.mean([
            y[k] - np.sum(alpha * y * np.dot(X[k], X.T))
            for k in support_vectors
        ])
        return b

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        n_samples, n_features = X.shape
        if self.kernel == 'l':
            K = self.linear_kernel(X)
        elif self.kernel == 'g':
            K = self.gaussian_kernel(X)
        k = np.diag(y) @ K @ np.diag(y)

        def dual_objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(k,  alpha)) - np.sum(alpha)


        constraints = [
            {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)},
            {'type': 'ineq', 'fun': lambda alpha: alpha},
            {'type': 'ineq', 'fun': lambda alpha: self.C - alpha},
        ]

        result = minimize(
            dual_objective,
            x0=np.zeros(n_samples),
            constraints=constraints,
            method='SLSQP'
        )

        self.alphas = result.x
        self.support_vecs = np.where(self.alphas > 1e-5)[0]
        self.weights = self.recover_weights(X, y, self.alphas)
        self.bias = self.compute_bias(X, y, self.alphas)
        
    def predict(self, x):
        if self.kernel == 'g':
            decision_value = 0
            for j in self.support_vecs:
                if self.kernel == 'g':
                    kernel_val = np.exp(-np.linalg.norm(self.X_train[j] - x)**2 / self.gamma)
                elif self.kernel == 'l':
                    kernel_val = np.dot(self.X_train[j], x)

                decision_value += self.alphas[j] * self.y_train[j] * kernel_val

            return np.sign(decision_value)  
        if self.kernel == 'l':
            return np.sign(np.dot(x, self.weights) + self.bias)

