import numpy as np
import qiskit as qk
from copy import deepcopy
from optimizers import Adam, GD
from data_encoders import *
from parametrizations import *


class Dense():
    def __init__(self, n_features=None, n_targets=None, scale=1, activation=None, bias=True):

        self.n_features = n_features
        self.n_targets = n_targets
        self.scale = scale
        self.activation = activation
        self.bias = bias

        self.randomize_weight()

    def __call__(self, inputs):
        x = inputs @ self.weight[:self.n_features]
        if self.bias:
            x += self.weight[-1].reshape(1, -1)

        x = self.activation(x)

        return self.scale * x

    def grad(self, inputs, delta, samplewise=False):
        n_samples = inputs.shape[0]
        output = self(inputs)
        delta = self.scale * \
            self.activation.derivative(output / self.scale) * delta

        if not samplewise:
            weight_gradient = 1 / n_samples * inputs.T @ delta
        else:
            weight_gradient = [np.outer(input_, delta_)
                               for input_, delta_ in zip(inputs, delta)]
            weight_gradient = np.array(weight_gradient)

        if self.bias:
            bias_gradient = np.mean(delta, axis=0, keepdims=True)
            weight_gradient = np.concatenate(
                (weight_gradient, bias_gradient), axis=0)

        delta = delta @ self.weight[:self.n_features].T

        return weight_gradient, delta

    def randomize_weight(self):
        self.weight = np.random.normal(
            0, 1, (self.n_features + self.bias, self.n_targets))

        #std = 1 / np.sqrt(self.n_targets)
        # self.weight = np.random.uniform(
        #    -std, std, (self.n_features + self.bias, self.n_targets))


class QLayer():
    def __init__(self, n_qubits=None, n_features=None, n_targets=None, reps=1, scale=1, encoder=None, ansatz=None, sampler=None, backend=None, shots=1000):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_targets = n_targets
        self.reps = reps
        self.scale = scale
        self.encoder = encoder
        self.ansatz = ansatz
        self.sampler = sampler
        self.backend = backend
        self.shots = shots

        self.last_layer = False
        self.randomize_weight()

    def __call__(self, inputs):
        outputs = []
        circuit_list = []
        n_samples = inputs.shape[0]
        for x in inputs:
            for i in range(self.n_targets):
                data_register = qk.QuantumRegister(
                    self.n_qubits, name="storage")
                clas_register = qk.ClassicalRegister(
                    self.n_qubits, name="clas_reg")
                registers = [data_register, clas_register]
                circuit = qk.QuantumCircuit(*registers)

                self.encoder(circuit, data_register, x)
                for j in range(self.reps):
                    start = j * self.n_qubits
                    end = (j + 1) * self.n_qubits
                    self.ansatz(circuit, data_register,
                                self.weight[start:end, i])

                circuit.measure(data_register, clas_register)
                circuit_list.append(circuit)

        transpiled_list = qk.transpile(circuit_list, backend=self.backend)
        qobject_list = qk.assemble(transpiled_list,
                                   backend=self.backend,
                                   shots=self.shots,
                                   max_parallel_shots=1,
                                   max_parallel_experiments=0
                                   )
        job = self.backend.run(qobject_list)

        for circuit in circuit_list:
            counts = job.result().get_counts(circuit)
            outputs.append(self.sampler(counts))

        outputs = np.array(outputs).reshape(n_samples, -1)

        return self.scale * np.array(outputs)

    def grad(self, inputs, delta, samplewise=False):
        inputs = deepcopy(inputs)
        n_samples = inputs.shape[0]
        weight_partial = np.zeros((n_samples, *self.weight.shape))
        input_partial = np.zeros((n_samples, self.n_features, self.n_targets))

        for i in range(self.reps * self.n_qubits):
            self.weight[i, :] += np.pi / 2
            weight_partial[:, i, :] = 1 / 2 * self(inputs)
            self.weight[i, :] += -np.pi
            weight_partial[:, i, :] += -1 / 2 * self(inputs)
            self.weight[i, :] += np.pi / 2

        weight_gradient = weight_partial * delta.reshape(n_samples, 1, -1)
        if not samplewise:
            weight_gradient = np.mean(weight_gradient, axis=0)

        if not self.last_layer:
            for i in range(self.n_features):
                inputs[:, i] += np.pi / 2
                input_partial[:, i, :] = 1 / 2 * self(inputs)
                inputs[:, i] += -np.pi
                input_partial[:, i, :] += -1 / 2 * self(inputs)
                inputs[:, i] += np.pi / 2

            delta = np.einsum("ij,ikj->ik", delta, input_partial)

        return weight_gradient, delta

    def randomize_weight(self):
        self.weight = np.random.uniform(
            0, 2 * np.pi, (self.reps * self.n_qubits, self.n_targets))


class Sigmoid():

    def __call__(self, x):
        x = 1 / (1 + np.exp(-x))

        return x

    def derivative(self, x):
        x = x * (1 - x)

        return x


class ReLu():

    def __call__(self, x):
        x = np.minimum(0, x)

        return x

    def derivative(self, x):
        x = x * (1 - x)

        return x


class Tanh():

    def __call__(self, x):
        x = np.tanh(x)

        return x

    def derivative(self, x):
        x = 1 - x**2

        return x


class Identity():

    def __call__(self, x):

        return x

    def derivative(self, x):

        return 1
