import numpy as np

class ReLU:
    def forward(self, x):
        self.last_input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.last_input <= 0] = 0
        return grad_input

class Softmax:
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_output):
        # Для кросс-энтропии обычно объединяется в одну функцию
        return grad_output
