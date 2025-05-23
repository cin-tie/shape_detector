import numpy as np
import random
from typing import List, Callable

class Layer:
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.random.randn(output_size, 1) * 0.1
        self.input = None
        self.output = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        raise NotImplementedError

class Dense(Layer):
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output
    
    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        input_error = np.dot(self.weights.T, output_error)
        weights_error = np.dot(output_error, self.input.T)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error
        return input_error
    
class Activation(Layer):
    def __init__(self, activation: Callable, activation_prime: Callable):
        super().__init__(0, 0)
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        return self.activation_prime(self.input) * output_error
    
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        result = input_data
        for layer in self.layers:
            result = layer.forward(result)
        return result
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float):
        for epoch in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                output = x
                for layer in self.layers:
                    output = layer.forwad(output)
                
                error += np.mean((y - output) ** 2)

                grad = output - y
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, error={error}")