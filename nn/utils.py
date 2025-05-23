import numpy as np

def initialize_weights(shape, method='xavier'):
    if method == 'xavier':
        limit = np.sqrt(6 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    else:
        return np.random.randn(*shape) * 0.01
