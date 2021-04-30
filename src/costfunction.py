import numpy as np


class MSE:
    def __call__(self, y_pred, y):
        return 0.5 * np.mean((y_pred - y)**2)

    def derivative(self, y_pred, y):
        return y_pred - y


class CrossEntropy:
    def __call__(self, y_pred, y):
        return -np.sum(y * np.log(y_pred))

    def derivative(self, y_pred, y):
        return (y_pred - y) / (y_pred * (1 - y_pred))


class NoCost:
    def __call__(self, y_pred, y):
        return y_pred

    def derivative(self, y_pred, y):
        n_samples, n_targets = y_pred.shape
        return np.ones((n_samples, n_targets))
