import numpy as np
import qiskit as qk
from copy import deepcopy
from tqdm.notebook import tqdm
from optimizers import Adam, GD
from data_encoders import *
from samplers import *
from utils import *


class Ansatz():
    def __call__(self, circuit, data_register, weight):
        n_qubits = data_register.size

        for i in range(n_qubits - 1):
            circuit.cx(data_register[i], data_register[i + 1])

        for i, w in enumerate(weight):
            circuit.ry(w, data_register[i])

        return circuit


class SwapTest():
    def __call__(self, circuit, register_a, register_b, ancilla_swap):
        circuit.h(ancilla_swap)
        for r_a, r_b in zip(register_a, register_b):
            circuit.cswap(ancilla_swap, r_a, r_b)

        circuit.h(ancilla_swap)

        return circuit


class ParallelModel():
    def __init__(self, n_features=None, n_targets=None, reps=1,  backend=None, shots=1000):
        self.parallel_encoder = ParallelEncoder()
        self.encoder = Encoder()
        self.ansatz = Ansatz()
        self.swap_test = SwapTest()

        self.n_features = n_features
        self.n_targets = n_targets
        self.reps = reps

        self.theta = np.random.uniform(
            0, 2 * np.pi, self.n_features * self.reps)
        self.backend = backend
        self.shots = shots

    def predict(self, x):
        n_samples, n_features = x.shape
        y_pred = []
        for i, x_ in enumerate(x):
            features = qk.QuantumRegister(self.n_features, name="features")
            predictions = qk.QuantumRegister(1, name="predictions")
            classical = qk.ClassicalRegister(1)
            registers = [features, predictions, classical]
            circuit = qk.QuantumCircuit(*registers)

            circuit = self.encoder(circuit, features, x_)
            for i in range(self.reps):
                start = i * self.n_features
                end = (i + 1) * self.n_features
                circuit = self.ansatz(circuit, features, self.theta[start:end])

            for i in range(n_features):
                circuit.cx(features[i], predictions)

            circuit.measure(predictions, classical)

            print(circuit)

            job = qk.execute(circuit, self.backend, shots=self.shots)
            counts = job.result().get_counts(circuit)
            """
            if "0" in counts:
                y_pred.append(
                    [2 * np.arccos(np.sqrt(counts["0"] / self.shots))])
            else:
                y_pred.append([np.pi])
            """
            if "0" in counts:
                y_pred.append(
                    [counts["0"] / self.shots])
            else:
                y_pred.append([0])

        return np.array(y_pred)

    def loss(self, x, y):
        n_samples, n_features = x.shape
        _, n_targets = y.shape
        n_ancilla = np.log2(n_samples)

        features = qk.QuantumRegister(n_features, name="features")
        predictions = qk.QuantumRegister(n_targets, name="predictions")
        ancilla_features = qk.QuantumRegister(
            n_ancilla, name="ancilla_feature")

        targets = qk.QuantumRegister(n_targets, name="targets")
        ancilla_targets = qk.QuantumRegister(n_ancilla, name="ancilla_target")

        ancilla_swap = qk.QuantumRegister(1, name="swap")
        classical = qk.ClassicalRegister(1)

        registers = [features, ancilla_features, predictions, targets,
                     ancilla_targets, ancilla_swap, classical]

        circuit = qk.QuantumCircuit(*registers)

        circuit = self.parallel_encoder(circuit, features, ancilla_features, x)
        circuit = self.parallel_encoder(circuit, targets, ancilla_targets, y)

        circuit.barrier()
        for i in range(self.reps):
            start = i * self.n_features
            end = (i + 1) * self.n_features
            circuit = self.ansatz(circuit, features, self.theta[start:end])

        circuit.barrier()

        for i in range(n_features):
            circuit.cx(features[i], predictions)

        register_a = predictions[:] + ancilla_features[:]
        register_b = targets[:] + ancilla_targets[:]
        circuit = self.swap_test(circuit, register_a, register_b, ancilla_swap)
        # circuit.cx(predictions, ancilla_swap)
        # circuit.cx(targets, ancilla_swap)
        circuit.measure(ancilla_swap, classical)

        # print(circuit)

        job = qk.execute(circuit, self.backend, shots=self.shots)
        counts = job.result().get_counts(circuit)
        if "0" in counts:
            loss = counts["0"] / self.shots
        else:
            loss = 0

        return loss

    def gradient(self, x, y):
        weight_gradient = np.zeros_like(self.theta)

        for i in range(len(self.theta)):
            self.theta[i] += np.pi / 2
            weight_gradient[i] += 1 / (2 * np.sqrt(2)) * self.loss(x, y)
            self.theta[i] += -np.pi
            weight_gradient[i] += -1 / (2 * np.sqrt(2)) * self.loss(x, y)
            self.theta[i] += np.pi / 2

        return weight_gradient


class RegularizedModel():
    def __init__(self, n_features=None, n_targets=None, reps=1, alpha=None, train_map=True,  backend=None, shots=1000, optimizer=None):
        self.encoder = RegularizedEncoder()
        self.ansatz = Ansatz()
        self.sampler = Parity()

        self.n_features = n_features
        self.n_targets = n_targets
        self.reps = reps
        self.alpha = alpha
        self.train_map = train_map
        self.backend = backend
        self.shots = shots
        self.optimizer = optimizer

        self.loss = []
        self.map_params = []

        self.n_qubits = n_features + 1
        self.theta = np.random.uniform(
            0, 2 * np.pi, self.n_qubits * self.reps + self.n_features)

        if self.train_map:
            self.theta[:self.n_features] = 0
        else:
            self.theta[:self.n_features] = np.pi / 2

        if not self.optimizer == None:
            self.optimizer.initialize(self.theta.shape)

    def predict(self, x):
        outputs = []
        circuit_list = []
        for i, x_ in enumerate(x):
            data_register = qk.QuantumRegister(self.n_qubits)
            classical = qk.ClassicalRegister(self.n_qubits)
            registers = [data_register, classical]
            circuit = qk.QuantumCircuit(*registers)

            circuit = self.encoder(
                circuit, data_register, x_, self.theta[:self.n_features])

            for j in range(self.reps):
                start = j * self.n_qubits + self.n_features
                end = (j + 1) * self.n_qubits + self.n_features
                circuit = self.ansatz(
                    circuit, data_register, self.theta[start:end])

            circuit.measure(data_register, classical)

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

        return np.array(outputs).reshape(-1, 1)

    def gradient(self, x, y):
        n_samples = x.shape[0]
        y_pred = self.predict(x)
        delta = (y_pred - y)
        weight_gradient = np.zeros((n_samples, len(self.theta)))

        if self.train_map:
            start = 0
        else:
            start = self.n_features

        for i in range(start, len(self.theta)):

            self.theta[i] += np.pi / 2
            weight_gradient[:, i] += 1 / \
                (2 * np.sqrt(2)) * self.predict(x)[:, 0]
            self.theta[i] += -np.pi
            weight_gradient[:, i] += -1 / \
                (2 * np.sqrt(2)) * self.predict(x)[:, 0]
            self.theta[i] += np.pi / 2

        weight_gradient = weight_gradient * delta
        weight_gradient = np.mean(weight_gradient, axis=0)

        return weight_gradient

    def train(self, x, y, epochs=100, verbose=False):
        if verbose:
            dec = tqdm
        else:
            dec = identity

        for i in dec(range(epochs)):
            y_pred = self.predict(x)
            self.loss.append(np.mean((y_pred - y)**2))
            self.map_params.append(deepcopy(self.theta[:self.n_features]))

            gradient = self.gradient(x, y)
            gradient_mod = self.optimizer([gradient])[0]

            penalty = np.zeros_like(gradient)
            if self.train_map:
                penalty[:self.n_features] = self.theta[:self.n_features]

            self.theta -= self.optimizer.lr * gradient_mod + self.alpha * penalty

            if verbose:
                print(f"epoch: {i}, loss: {self.loss[-1]}")

        y_pred = self.predict(x)
        self.loss.append(np.mean((y_pred - y)**2))
        self.map_params.append(self.theta[:self.n_features])
