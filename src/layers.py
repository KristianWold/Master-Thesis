import numpy as np
import qiskit as qk
from copy import deepcopy
from optimizers import Adam, GD
from data_encoders import *
from ansatzes import *


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
            if not samplewise:
                bias_gradient = np.mean(delta, axis=0, keepdims=True)

                weight_gradient = np.concatenate(
                    (weight_gradient, bias_gradient), axis=0)
            else:
                bias_gradient = delta

                weight_gradient = np.concatenate(
                    (weight_gradient, bias_gradient[:, None, :]), axis=1)

        delta = delta @ self.weight[:self.n_features].T

        return weight_gradient, delta

    def randomize_weight(self):
        """Xavier initialization"""

        std = 1 / np.sqrt(self.n_features)
        self.weight = np.random.uniform(
            -std, std, (self.n_features + self.bias, self.n_targets))


class QLayer():
    def __init__(self,
                 n_qubits=None,
                 n_features=None,
                 n_targets=None,
                 scale=1,
                 encoder=None,
                 ansatz=None,
                 sampler=None,
                 backend=None,
                 shots=1000):

        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_targets = n_targets
        self.scale = scale
        self.encoder = encoder
        self.ansatz = ansatz
        self.sampler = sampler
        self.backend = backend
        self.shots = shots

        self.encoder.calculate_n_weights(self.n_qubits)
        self.ansatz.calculate_n_weights(self.n_qubits)
        self.n_weights_per_target = self.encoder.n_weights_per_target + \
            self.ansatz.n_weights_per_target

        self.last_layer = False
        self.randomize_weight()

    def __call__(self, inputs):
        backend = qk.providers.aer.StatevectorSimulator()
        outputs = []
        circuit_list = []
        n_samples = inputs.shape[0]
        for x in inputs:
            for i in range(self.n_targets):
                data_register = qk.QuantumRegister(
                    self.n_qubits)
                clas_register = qk.ClassicalRegister(
                    self.n_qubits)
                registers = [data_register]

                if self.shots != 0:
                    registers.append(clas_register)

                circuit = qk.QuantumCircuit(*registers)

                idx = self.encoder.n_weights_per_target
                self.encoder(circuit, data_register,
                             self.weight[:idx, i], x)

                self.ansatz(circuit, data_register,
                            self.weight[idx:, i])

                if self.shots == 0:
                    circuit = self.sampler.observable(circuit, data_register)
                else:
                    circuit.measure(data_register, clas_register)

                circuit_list.append(circuit)

        #transpiled_list = qk.transpile(circuit_list, backend=self.backend)

        if self.shots == 0:
            for circuit in circuit_list:
                counts = qk.execute(circuit, backend).result().get_counts()
                outputs.append(self.sampler(counts))
        else:
            qobject_list = qk.assemble(circuit_list,
                                       backend=self.backend,
                                       shots=self.shots,
                                       max_parallel_shots=1,
                                       max_parallel_experiments=0
                                       )

            job = self.backend.run(qobject_list)
            for counts in job.result().get_counts():
                outputs.append(self.sampler(counts))

        outputs = np.array(outputs).reshape(n_samples, -1)

        return self._scale_output(np.array(outputs))

    def grad(self, inputs, delta, samplewise=False):
        inputs = deepcopy(inputs)
        n_samples = inputs.shape[0]
        self.weight_partial = np.zeros((n_samples, *self.weight.shape))
        self.input_partial = np.zeros(
            (n_samples, self.n_features, self.n_targets))

        for i in range(self.n_weights_per_target):
            self.weight[i, :] += np.pi / 2
            self.weight_partial[:, i, :] = 1 / 2 * self(inputs)
            self.weight[i, :] += -np.pi
            self.weight_partial[:, i, :] += -1 / 2 * self(inputs)
            self.weight[i, :] += np.pi / 2

        weight_gradient = self.weight_partial * delta.reshape(n_samples, 1, -1)
        if not samplewise:
            weight_gradient = np.mean(weight_gradient, axis=0)

        if not self.last_layer:
            for i in range(self.n_features):
                inputs[:, i] += np.pi / 2
                self.input_partial[:, i, :] = 1 / 2 * self(inputs)
                inputs[:, i] += -np.pi
                self.input_partial[:, i, :] += -1 / 2 * self(inputs)
                inputs[:, i] += np.pi / 2

            delta = np.einsum("ij,ikj->ik", delta, self.input_partial)

        return weight_gradient, delta

    def randomize_weight(self):
        self.weight = np.random.uniform(
            -np.pi, np.pi, (self.n_weights_per_target, self.n_targets))

    def _scale_output(self, output):
        if self.scale == None:
            return output
        elif type(self.scale) != list:
            return self.scale * output
        else:
            a = self.scale[0]
            b = self.scale[1]
            return (b - a) * output + a


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
