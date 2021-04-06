import numpy as np


class GD():
    def __init__(self, lr=0.01):
        self.lr = lr

    def initialize(self, dims):
        pass

    def __call__(self, weight_gradient_list):
        return weight_gradient_list


class Adam():
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = None

    def initialize(self, dims):
        self.m = []
        self.v = []
        self.t = 0

        for dim in dims:
            self.m.append(np.zeros(dim))
            self.v.append(np.zeros(dim))

    def __call__(self, weight_gradient_list):
        self.t += 1
        weight_gradient_modified = []

        for grad, m_, v_ in zip(weight_gradient_list, self.m, self.v):
            m_[:] = self.beta1 * m_ + (1 - self.beta1) * grad
            v_[:] = self.beta2 * v_ + (1 - self.beta2) * grad**2

            m_hat = m_ / (1 - self.beta1**self.t)
            v_hat = v_ / (1 - self.beta2**self.t)
            grad_modified = m_hat / (np.sqrt(v_hat) + self.eps)
            weight_gradient_modified.append(grad_modified)

        return weight_gradient_modified
