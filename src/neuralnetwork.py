import numpy as np
import qiskit as qk
import pickle
from tqdm.notebook import tqdm
from copy import deepcopy

from optimizers import *
from layers import *
from utils import *
from samplers import *
from costfunction import *


class NeuralNetwork():
    def __init__(self, layers=None, cost=MSE(), optimizer=Adam(lr=0.1)):
        self.layers = layers
        self.cost = cost
        self.optimizer = optimizer

        self.layers[0].last_layer = True
        self.dim = []

        if not self.layers == None:
            for layer in self.layers:
                self.dim.append(layer.weight.shape)

        if not self.optimizer == None:
            self.optimizer.initialize(self.dim)

        self.a = []
        self.weight_gradient_list = []

    def __call__(self, x, verbose=False):
        if verbose:
            decerator = tqdm
        else:
            decerator = identity

        self.a = []
        self.a.append(x)
        for layer in decerator(self.layers):
            x = layer(x)
            self.a.append(x)

    def predict(self, x, verbose=False):
        self(x, verbose=verbose)
        return self.a[-1]

    def backward(self, x, y=None, samplewise=False):
        n_samples = x.shape[0]
        self.weight_gradient_list = []

        self(x)
        y_pred = self.a[-1]

        delta = self.cost.derivative(y_pred, y)

        for i, layer in reversed(list(enumerate(self.layers))):
            weight_gradient, delta = layer.grad(
                self.a[i], delta, samplewise=samplewise)
            self.weight_gradient_list.append(weight_gradient)

        self.weight_gradient_list.reverse()

    def step(self):
        weight_gradient_modified = self.optimizer(self.weight_gradient_list)

        for layer, grad in zip(self.layers, weight_gradient_modified):
            layer.weight += -self.optimizer.lr * grad

    def train(self, x, y, epochs=100, verbose=False):
        if verbose:
            dec = tqdm
        else:
            dec = identity

        self.loss = []
        for i in dec(range(epochs)):

            self.backward(x, y)
            self.step()

            y_pred = self.a[-1]
            self.loss.append(np.mean((y_pred - y)**2))

            if verbose:
                print(f"epoch: {i}, loss: {self.loss[-1]}")

        y_pred = self.predict(x)
        self.loss.append(np.mean((y_pred - y)**2))

    def deriv(self, x):
        self.layers[0].last_layer = False
        self.weight_gradient_list = []

        self(x)
        delta = np.ones_like(x)

        for i, layer in reversed(list(enumerate(self.layers))):
            weight_gradient, delta = layer.grad(self.a[i], delta)

        self.layers[0].last_layer = True

        return delta

    @property
    def weight(self):
        weight_list = []
        for layer in self.layers:
            weight_list.append(layer.weight)

        return weight_list

    def randomize_weight(self):
        for layer in self.layers:
            layer.randomize_weight()

    def set_shots(self, shots):
        for layer in self.layers:
            layer.shots = shots

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))

    def load(self, filename):
        self = pickle.load(open(filename, "rb"))


def sequential_qnn(n_qubits=None,
                   dim=None,
                   encoder=Encoder(),
                   ansatz=Ansatz(reps=1),
                   sampler=LastBit(),
                   scale=None,
                   cost=MSE(),
                   optimizer=Adam(lr=0.1),
                   backend=None,
                   shots=None):
    L = len(dim)
    if scale == None:
        scale = (L - 2) * [[-np.pi, np.pi]]
        scale = scale + [1]

    layers = []
    for i in range(L - 1):
        in_dim = dim[i]
        out_dim = dim[i + 1]

        _encoder = deepcopy(encoder)
        _ansatz = deepcopy(ansatz)
        layer = QLayer(n_qubits=n_qubits[i],
                       n_features=in_dim,
                       n_targets=out_dim,
                       encoder=_encoder,
                       ansatz=_ansatz,
                       sampler=sampler,
                       scale=scale[i],
                       backend=backend,
                       shots=shots)
        layers.append(layer)

    network = NeuralNetwork(layers, cost=cost, optimizer=optimizer)

    return network


def sequential_dnn(dim=None,
                   bias=True,
                   scale=None,
                   cost=MSE(),
                   optimizer=Adam(lr=0.1)):
    L = len(dim)

    if scale == None:
        scale = (L - 1) * [1]

    layers = []
    for i in range(L - 1):
        in_dim = dim[i]
        out_dim = dim[i + 1]
        layer = Dense(n_features=in_dim, n_targets=out_dim, scale=scale[i],
                      activation=Sigmoid(), bias=bias)
        layers.append(layer)

    network = NeuralNetwork(layers, cost=cost, optimizer=optimizer)

    return network
