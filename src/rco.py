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


class RCO:

    def __init__(self, ansatz, sampler, optimizer, divisor, tol=1e-3):
        self.ansatz = ansatz
        self.sampler = sampler
        self.optimizer = optimizer
        self.divisor = divisor
        self.tol = tol

    def fit(self, circuit):
        self.n_qubits = circuit.num_qubits
        self.n_params = self.ansatz.calculate_n_weights(self.n_qubits)
        self.params = np.random.uniform(-np.pi, np.pi,
                                        (self.divisor, self.n_params))

        circuit_list = self.divide_circuit(circuit, self.divisor)
        for i in range(self.divisor):
            params_cur = params[i]
            self.optimize(circuit_list[i], params[i], params_prev)

    def divide_circuit(self, circuit, divisor):
        circuit_size = len(circuit)
        gates_per_sub_circuit = circuit_size // divisor
        k = 0
        circuit_list = []
        while k < circuit_size:
            _circuit = deepcopy(circuit)
            for i in range(k):
                _circuit.data.pop(0)
            for i in range(circuit_size - gates_per_sub_circuit - k):
                _circuit.data.pop(-1)
            circuit_list.append(_circuit)
            k += gates_per_sub_circuit

        return circuit_list

    def optimize(self, sub_circuit, params, params_prev=None):
