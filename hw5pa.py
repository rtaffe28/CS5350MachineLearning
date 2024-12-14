import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate=0.1, epochs=100):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size
        self.hidden_size_2 = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.W1 = np.array([[-1, 1], [-2, 2], [-3, 3]], dtype=np.float64)

        self.W2 = np.array([[-2, 2], [-3, 3]], dtype=np.float64)
        self.b2 = np.array([-1, 1], dtype=np.float64)

        self.W3 = np.array([[2], [-1.5]], dtype=np.float64)
        self.b3 = np.array([-1], dtype=np.float64)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.z3

    def backward(self, X, y):
        delta3 = self.z3 - y
        dW3 = np.dot(self.a2.T, delta3)
        db3 = np.sum(delta3, axis=0) 
        
        delta2 = np.dot(delta3, self.W3.T) * self.sigmoid_derivative(self.a2)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0)
        
        delta1 = np.dot(delta2, self.W2) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta1)
        
        self.W1 -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

        print(f"Gradients for W3:\n{dW3}\n")
        print(f"Gradients for b3:\n{db3}\n")
        print(f"Gradients for W2:\n{dW2}\n")
        print(f"Gradients for b2:\n{db2}\n")
        print(f"Gradients for W1:\n{dW1}\n")
        
        
    def train(self, x, y):
        print("forward pass:", self.forward(x))
        print()
        self.backward(x, y)

nn = NeuralNetwork(input_size=3, hidden_size=2, learning_rate=0.1, epochs=1)

x = np.array([[1, 1, 1]])
y = np.array([[1]])

nn.train(x, y)
