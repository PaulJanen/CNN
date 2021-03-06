from matplotlib.pyplot import axis
import numpy as np


class ActivationFunction:
    def f(self, x):
        raise NotImplementedError

    def df(self, x, cached_y=None):
        raise NotImplementedError

class Identity(ActivationFunction):
    def f(self, x):
        return x

    def df(self, x, cached_y=None):
        return np.full(x.shape, 1)


class Sigmoid(ActivationFunction):
    # Numericaly stable sigmoid implementation - small values = bad. Ex: x = -200000 will result in overflow.
    def f(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def df(self, x, cached_y=None):
        y = cached_y if cached_y is not None else self.f(x)
        return y * (1 - y)

class ReLU(ActivationFunction):
    def f(self, x):
        return np.maximum(0, x)

    def df(self, x, cached_y=None):
        return np.where(x <= 0, 0, 1)

class SoftMax(ActivationFunction):
    def f(self, x):
        y = np.exp(x - np.max(x, axis=1, keepdims=True))
        return y / np.sum(y, axis=1, keepdims=True)

    def df(self, x, cached_y=None):
        raise NotImplementedError


class SoftMaxConv(ActivationFunction):
    # x dims: (batch_size, self.n_h, self.n_w, self.n_c)
    def f(self, x):
        y = np.exp(x - np.max(x, axis = 3, keepdims=True))
        return y / np.sum(y, axis= 3, keepdims=True)
        #y = np.exp(x - np.max(x, axis=1, keepdims=True))
        #return y / np.sum(y, axis=1, keepdims=True)

    # y has to be one hot encoded in convolutional fashion and not as a vector
    def df(self, x, cached_y=None):
        raise NotImplementedError

identity = Identity()
sigmoid = Sigmoid()
relu = ReLU()
softmax = SoftMax()
softmaxConv = SoftMaxConv()