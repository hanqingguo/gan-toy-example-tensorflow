import numpy as np


def gaussian_data(std, mean, batch_size, dim):
    return np.random.randn(batch_size, dim) * std + mean

