import numpy as np
import qiskit as qk
from copy import deepcopy
from tqdm.notebook import tqdm
from optimizers import Adam, GD
from data_encoders import *
from samplers import *
from utils import *


class Ansatz():
    def __init__(self, reps):
        self.reps = reps

    def __call__(self, circuit, data_register, weight):

        for i in range(self.reps):
            for j in range(self.n_qubits - 1):
                circuit.cx(data_register[j], data_register[j + 1])

            idx_start = i * self.n_qubits
            idx_end = (i + 1) * self.n_qubits
            for j, w in enumerate(weight[idx_start:idx_end]):
                circuit.ry(w, data_register[j])

        return circuit

    def calculate_n_weights(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_weights_per_target = self.reps * self.n_qubits


class Ansatz2():
    def __call__(self, circuit, data_register, weight):
        n_qubits = data_register.size

        for i, w in enumerate(weight[:n_qubits // 2]):
            circuit.cx(data_register[2 * i], data_register[2 * i + 1])
            circuit.rz(w, data_register[i])
            circuit.cx(data_register[2 * i], data_register[2 * i + 1])

        for i, w in enumerate(weight[n_qubits // 2:]):
            circuit.cx(data_register[2 * i + 1], data_register[2 * i + 2])
            circuit.rz(w, data_register[i])
            circuit.cx(data_register[2 * i + 1], data_register[2 * i + 2])

        return circuit
