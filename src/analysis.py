import numpy as np
import qiskit as qk
from copy import deepcopy
from tqdm.notebook import tqdm
from neuralnetwork import *
from utils import *
from sklearn.decomposition import PCA


class FIM():
    """Empirical Fisher Information Matrix

    Parameters
    ----------
    model :
        Parameterized model
    """

    def __init__(self, model):
        self.model = model
        self.fim = None

    def fit(self, x):
        """Calculate the FIM for the given model over the input data x

        Parameters
        ----------
        x : ndarray
            input data
        """
        n_samples = x.shape[0]

        self.model.cost = NoCost()
        self.model.backward(x, samplewise=True)
        gradient = self.model.weight_gradient_list

        gradient_flattened = []
        for grad in gradient:
            gradient_flattened.append(grad.reshape(n_samples, -1))

        gradient_flattened = np.concatenate(gradient_flattened, axis=1)

        self.fim = 1 / n_samples * gradient_flattened.T @ gradient_flattened

    def eigen(self, sort=True):
        """Calculate the eigenvalue spectrum of the FIM.

        Parameters
        ----------
        sort : boolean
            Will sort the resulting eigenvalues in decending order if True.

        Notes
        -----
        Must be called after self.fit(x).
        """
        self.fim = np.array(self.fim, dtype=np.float64)
        self.eigen = np.linalg.eigh(self.fim)[0]

        if sort:
            self.eigen[::-1].sort()
        return np.maximum(self.eigen, 1e-25)

    def fisher_rao(self):
        """Calculate the Fisher-Rao metric.

        Notes
        -----
        Must be called after self.fit(x).
        """
        weight = self.model.weight

        weight_flattened = []
        for w in weight:
            weight_flattened.append(w.reshape(-1, 1))

        weight = np.concatenate(weight_flattened, axis=0)

        fr = weight.T @ self.fim @ weight

        return fr[0][0]


class TrajectoryLength:
    def __init__(self, model):
        self.model = model

    def fit(self, x):
        pca = PCA(n_components=2)

        self.model(x)
        self.trajectory_length = []
        self.trajectory_projection = []
        for trajectory in self.model.a:
            diff = (trajectory[1:] - trajectory[:-1])
            accum = np.sum(diff**2, axis=1)
            accum = np.sum(np.sqrt(accum))
            self.trajectory_length.append(accum)
            self.trajectory_projection.append(pca.fit_transform(trajectory))

        return self.trajectory_length, self.trajectory_projection


def gradient_analysis(network_list):
    n_models = len(network_list)
    gradients = np.zeros(
        (n_models, len(network_list[0].weight_gradient_list)))

    input_partial_avg = np.zeros(len(network_list[0].weight_gradient_list))
    weight_partial_avg = np.zeros(len(network_list[0].weight_gradient_list))

    for i, network in enumerate(network_list):
        for j in range(len(network.weight_gradient_list)):
            grad = network.weight_gradient_list[j]
            input_partial = network.layers[j].input_partial
            weight_partial = network.layers[j].weight_partial

            gradients[i, j] += np.mean(np.abs(grad))
            input_partial_avg[j] += np.mean(np.abs(input_partial))
            weight_partial_avg[j] += np.mean(np.abs(weight_partial))

    gradient_avg = np.mean(gradients, axis=0)
    gradient_std = np.std(gradients, axis=0)
    input_partial_avg /= n_models
    weight_partial_avg /= n_models

    return gradient_avg, input_partial_avg, weight_partial_avg
