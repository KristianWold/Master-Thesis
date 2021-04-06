import numpy as np
import qiskit as qk
from copy import deepcopy
from tqdm.notebook import tqdm
from neuralnetwork import *
from utils import *


class FIM():
    def __init__(self, model):
        self.model = model
        self.fim = None

    def fit(self, x):
        n_samples = x.shape[0]

        self.model.backward(x, samplewise=True, include_loss=False)
        gradient = self.model.weight_gradient_list

        gradient_flattened = []
        for grad in gradient:
            gradient_flattened.append(grad.reshape(n_samples, -1))

        gradient_flattened = np.concatenate(gradient_flattened, axis=1)

        self.fim = 1 / n_samples * gradient_flattened.T @ gradient_flattened

    def eigen(self, sort=False):
        self.eigen = np.linalg.eig(self.fim)[0]
        if sort:
            self.eigen[::-1].sort()
        return np.abs(self.eigen)

    def fisher_rao(self):
        weight = self.model.weight

        weight_flattened = []
        for w in weight:
            weight_flattened.append(w.reshape(-1, 1))

        weight = np.concatenate(weight_flattened, axis=0)

        fr = weight.T @ self.fim @ weight

        return fr[0][0]


def trajectory_length(x):
    diff = (x[1:] - x[:-1])
    diff = np.append(diff, (x[0] - x[-1]).reshape(1, -1), axis=0)
    accum = np.sum(diff**2, axis=1)
    accum = np.sum(np.sqrt(accum))
    return accum


def trajectory_curvature(x):
    diff = (x[1:] - x[:-1])
    dot = np.matmul()
    accum = np.sum(diff**2, axis=1)
    accum = np.sum(np.sqrt(accum))
    return accum
