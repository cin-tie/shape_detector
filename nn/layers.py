import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        limit = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weights = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
        self.biases = np.zeros(out_channels)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size, _, h, w = x.shape
        padded_x = np.pad(x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')

        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        out = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        vert_start = i * self.stride
                        vert_end = vert_start + self.kernel_size
                        horiz_start = j * self.stride
                        horiz_end = horiz_start + self.kernel_size

                        region = padded_x[b, :, vert_start:vert_end, horiz_start:horiz_end]
                        out[b, c_out, i, j] = np.sum(region * self.weights[c_out]) + self.biases[c_out]
        return out

    # backward не реализован (учебный пример)

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, channels, out_h, out_w))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        vert_start = i * self.stride
                        vert_end = vert_start + self.kernel_size
                        horiz_start = j * self.stride
                        horiz_end = horiz_start + self.kernel_size

                        region = x[b, c, vert_start:vert_end, horiz_start:horiz_end]
                        out[b, c, i, j] = np.max(region)
        return out

class Flatten:
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.last_input = None

    def forward(self, x):
        self.last_input = x  # Сохраняем вход для обратного прохода
        return x @ self.weights + self.bias

    def backward(self, grad_output, learning_rate):
        # grad_output shape: (batch_size, output_size)
        grad_weights = self.last_input.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.weights.T

        # Обновляем параметры
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input
