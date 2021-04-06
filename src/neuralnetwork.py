import numpy as np
import qiskit as qk
import pickle
from tqdm.notebook import tqdm

from optimizers import *
from layers import *
from utils import *
from parametrizations import *
from samplers import *


class NeuralNetwork():
    def __init__(self, layers=None, optimizer=None):
        self.layers = layers
        self.layers[0].last_layer = True
        self.dim = []
        self.optimizer = optimizer

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

    def backward(self, x, y=None, samplewise=False, include_loss=True):
        n_samples = x.shape[0]
        self.weight_gradient_list = []

        self(x)
        y_pred = self.a[-1]

        if include_loss:
            delta = (y_pred - y)
        else:
            delta = np.ones((n_samples, 1))

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


def sequential_qnn(q_bits=None, dim=None, reps=None, scale=None, backend=None, shots=None, lr=0.01):
    L = len(dim)
    if scale == None:
        scale = (L - 2) * [2 * np.pi]
        scale = scale + [1]

    layers = []
    for i in range(L - 1):
        in_dim = dim[i]
        out_dim = dim[i + 1]
        layer = QLayer(n_qubits=q_bits[i], n_features=in_dim, n_targets=out_dim, encoder=Encoder(
        ), ansatz=Ansatz(), sampler=Parity(), reps=reps, scale=scale[i], backend=backend, shots=shots)
        layers.append(layer)

    optimizer = Adam(lr=lr)
    network = NeuralNetwork(layers, optimizer)

    return network


def sequential_dnn(dim=None, bias=True, scale=None, lr=0.01):
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

    optimizer = Adam(lr=lr)
    network = NeuralNetwork(layers, optimizer)

    return network
