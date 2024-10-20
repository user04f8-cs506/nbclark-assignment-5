import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size=64, output_size=None, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size else input_size  # By default, same output size as input
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias_1 = np.zeros((1, self.hidden_size))
        self.weights_2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias_2 = np.zeros((1, self.output_size))
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.weights_1) + self.bias_1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights_2) + self.bias_2
        return self.Z2  # Transformed features
    
    def backward(self, X, grad_output):
        # Gradients for layer 2
        grad_Z2 = grad_output
        grad_weights_2 = np.dot(self.A1.T, grad_Z2)
        grad_bias_2 = np.sum(grad_Z2, axis=0, keepdims=True)
        
        # Gradients for layer 1
        grad_A1 = np.dot(grad_Z2, self.weights_2.T)
        grad_Z1 = grad_A1 * self.relu_derivative(self.Z1)
        grad_weights_1 = np.dot(X.T, grad_Z1)
        grad_bias_1 = np.sum(grad_Z1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights_1 -= self.learning_rate * grad_weights_1
        self.bias_1 -= self.learning_rate * grad_bias_1
        self.weights_2 -= self.learning_rate * grad_weights_2
        self.bias_2 -= self.learning_rate * grad_bias_2
        
    def train(self, X, grad_output, epochs=100):
        for epoch in range(epochs):
            transformed_features = self.forward(X)
            self.backward(X, grad_output)
