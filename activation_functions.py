import numpy as np


def relu(x):
    return 0.5 * (x + np.abs(x))


def relu_prime(x):
    return x > 0.


def relu_jac(x):
    return np.diag(relu_prime(x))


def tanh_prime(x):
    return 1 - np.tanh(x)**2


def tanh_jac(x):
    return np.diag(tanh_prime(x))


def linear(x):
    return x


def linear_prime(x):
    return np.ones_like(x)


def linear_jac(x):
    return np.diag(linear_prime(x))