import numpy as np

class Conv2D:
    def __init__(self, filters: int, kernel_size: int, input_shape: tuple):
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape

        self.kernels = np.random.randn(filters, kernel_size, kernel_size) * 0.1
        self.biases = np.random.randn(filters) * 0.1

        self.output_shape = (
            input_shape[0] - kernel_size + 1,
            input_shape[0] - kernel_size + 1,
            filters
        )

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        output = np.zeros(self.output_shape)

        for f in range(self.filters):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    output[i, j, f] = np.sum(
                        input_data[i:i+self.kernel_size, j:j+self.kernel_size] * self.kernel_size[f]
                    ) + self.biases[f]
        
        return output
    
    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        input_error = np.zeros(self.input_shape)
        kernels_error = np.zeros_like(self.kernels)
        biases_error = np.zeros_like(self.biases)

        for f in range(self.filters):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    kernels_error[f] += (
                        self.input[i:i+self.kernel_size, j:j+self.kernel_size] * output_error[i, j, f]
                    )
                    input_error[i:i+self.kernel_size, j:j+self.kernel_size] += (
                        self.kernels[f] * output_error[i, j, f]
                    )
                    biases_error[f] += output_error[i, j, f]
        
        self.kernels -= learning_rate * kernels_error
        self.biases -= learning_rate * biases_error

        return input_error
    
class MaxPooling2D:
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
        self.input = None
        self.mask = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        h, w, c = input_data.shape
        self.mask= np.zeros_like(input_data)

        output_h = h // self.pool_size
        output_w = w // self.pool_size
        output = np.zeros((output_h, output_w, c))

        for i in range(output_h):
            for j in range(output_w):
                for k in range(c):
                    h_start = i * self.pool_size
                    w_start = j * self.pool_size
                    h_end = h_start + self.pool_size
                    w_end = w_start + self.pool_size

                    region = input_data[h_start:h_end, w_start, w_end]
                    max_val = np.max(region)
                    output[i, j, k] = max_val

                    mask = (region == max_val).astype(int)
                    self.mask[h_start:h_end, w_start:w_end, k] = mask
        
        return output
    
    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        h, w, c = self.input.shape
        input_error = np.zeros_like(self.input)

        output_h = h // self.pool_size
        output_w = w // self.pool_size

        for i in range(output_h):
            for j in range(output_w):
                for k in range(c):
                    h_start = i * self.pool_size
                    w_start = j * self.pool_size
                    h_end = h_start + self.pool_size
                    w_end = w_start + self.pool_size

                    input_error[h_start:h_end, w_start:w_end, k] += (
                        output_error[i, j, k] * self.mask[h_start:h_end, w_start:w_end, k]
                    )

        return input_error
    
class Flatten:
    def __init__(self):
        self.input_shape = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input_shape = input_data.shape
        return input_data.flatten().reshape(-1, 1)
    
    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        return output_error.reshape(self.input_shape)