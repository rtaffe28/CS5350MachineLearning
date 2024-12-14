import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate=0.1, epochs=200, d=10):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size
        self.hidden_size_2 = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.d = d

        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        
        self.W2 = np.random.randn(hidden_size, hidden_size) 
        self.b2 = np.random.randn(hidden_size) 
        
        self.W3 = np.random.randn(hidden_size, 1)

        # self.W1 = np.zeros((input_size, hidden_size))
        # self.b1 = np.zeros(hidden_size)
        
        # self.W2 = np.zeros((hidden_size, hidden_size))
        # self.b2 = np.zeros(hidden_size)
        
        # self.W3 = np.zeros((hidden_size, 1))
        # self.b3 = np.zeros(1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3)
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3

    def backward(self, X, y):
        m = X.shape[0]
        delta3 = self.a3 - y
        dW3 = np.dot(self.a2.T, delta3) / m
        db3 = np.sum(delta3, axis=0) / m
        
        delta2 = np.dot(delta3, self.W3.T) * self.sigmoid_derivative(self.a2)
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3

    def train(self, X, y):
        t = 1
        for epoch in range(self.epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i, (x_i, y_i) in enumerate(zip(X_shuffled, y_shuffled)):
                self.learning_rate = self.learning_rate / (1 + (t*(self.learning_rate/self.d)))
                x_i = x_i.reshape(1, -1)
                y_i = y_i.reshape(1, -1)
                self.forward(x_i)
                self.backward(x_i, y_i)
            t += 1
                
    def predict(self, X):
        output = self.forward(X)
        return 1 if (output > 0.5) else 0
