import numpy as np


def estimator_stats(x_hat, x_true):
    mu = np.mean(x_hat, axis=0)
    std = np.std(x_hat, axis=0, ddof=1)
    bias = mu - x_true
    mse = std**2 + bias**2
    return mu, std, bias, mse


def mean_with_standard_error(x, geometric_mean=False, axis=-1):
    N_samples = x.shape[axis]
    if geometric_mean:
        lx = np.log(x)
        m = np.exp(lx.mean(axis))
        return m, m * lx.std(ddof=1, axis=axis) / np.sqrt(N_samples)
    return x.mean(axis), x.std(ddof=1, axis=axis) / np.sqrt(N_samples)
