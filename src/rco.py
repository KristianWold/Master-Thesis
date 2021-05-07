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


class Ansatz():
    def __init__(self, blocks=["entangle", "ry", "rz"], reps=1):
        self.blocks = blocks
        self.reps = reps

    def __call__(self, circuit, data_register, weight, inverse=False):
        qregs = circuit.qregs[0]
        _circuit = qk.QuantumCircuit(qregs)

        idx_start = idx_end = 0
        for i in range(self.reps):
            for block in self.blocks:
                if block == "rx":
                    idx_end += self.n_qubits
                    for j, w in enumerate(weight[idx_start:idx_end]):
                        _circuit.rx(w, data_register[j])
                    idx_start = idx_end

                if block == "ry":
                    idx_end += self.n_qubits
                    for j, w in enumerate(weight[idx_start:idx_end]):
                        _circuit.ry(w, data_register[j])
                    idx_start = idx_end

                if block == "rz":
                    idx_end += self.n_qubits
                    for j, w in enumerate(weight[idx_start:idx_end]):
                        _circuit.rz(w, data_register[j])
                    idx_start = idx_end

                if block == "entangle":
                    for j in range(self.n_qubits - 1):
                        _circuit.cx(data_register[j], data_register[j + 1])

        if inverse:
            _circuit = _circuit.inverse()

        return circuit + _circuit

    def calculate_n_weights(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_weights_per_target = 2 * self.reps * self.n_qubits


class RCO:

    def __init__(self, ansatz, n_qubits, sampler, optimizer, divisor, shots, tol=1e-3, warm_start=False):
        self.ansatz = ansatz
        self.sampler = sampler
        self.optimizer = optimizer
        self.divisor = divisor
        self.shots = shots
        self.tol = tol
        self.warm_start = warm_start
        self.n_qubits = n_qubits

        self.ansatz.calculate_n_weights(self.n_qubits)
        self.n_params = self.ansatz.n_weights_per_target
        self.params = np.random.uniform(-np.pi, np.pi,
                                        (self.divisor, self.n_params))

        self.error_sampler = ZeroBit()

    def fit(self, circuit):
        circuit_list = self.divide_circuit(circuit, self.divisor)
        params_prev = None
        print(f"{0}/{self.divisor} iterations")
        for i in range(self.divisor):
            if i != 0:
                params_prev = self.params[i - 1]
                if self.warm_start:
                    self.params[i] = np.copy(params_prev)

            self.optimize(circuit_list[i], self.params[i], params_prev)

            print(f"{i+1}/{self.divisor} iterations")

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
            circuit_list.append(deepcopy(_circuit))
            k += gates_per_sub_circuit

        return circuit_list

    def evaluate(self, target_circuit, params, params_prev=None, return_circuit=False, use_error_measure=False):
        qreg = target_circuit.qregs[0]
        circuit = qk.QuantumCircuit(qreg)

        circuit = self.ansatz(circuit, qreg, params)
        circuit += target_circuit.inverse()
        if params_prev is not None:
            circuit = self.ansatz(circuit, qreg, params_prev, inverse=True)

        creg = qk.ClassicalRegister(qreg.size)
        circuit.add_register(creg)
        circuit.measure(qreg, creg)
        if return_circuit:
            return circuit
        else:
            job = qk.execute(circuit, backend=qk.Aer.get_backend(
                'qasm_simulator'), shots=self.shots)
            counts = job.result().get_counts(circuit)

            if use_error_measure == True:
                output = self.error_sampler(counts)
            else:
                output = self.sampler(counts)

            return output

    def gradient(self, target_circuit, params, params_prev=None):
        backend = qk.Aer.get_backend('qasm_simulator')

        grads = np.zeros(params.shape)
        circuit_plus_list = []
        circuit_minus_list = []
        for i in range(len(params)):
            params[i] += np.pi / 2
            circuit_plus_list.append(self.evaluate(
                target_circuit, params, params_prev, return_circuit=True))

            params[i] -= np.pi
            circuit_minus_list.append(self.evaluate(
                target_circuit, params, params_prev, return_circuit=True))

            params[i] += np.pi / 2

        circuit_list = circuit_plus_list + circuit_minus_list
        transpiled_list = qk.transpile(circuit_list, backend=backend)
        qobject_list = qk.assemble(transpiled_list,
                                   backend=backend,
                                   shots=self.shots,
                                   max_parallel_shots=0,
                                   max_parallel_experiments=0)

        job = backend.run(qobject_list)

        outputs = []
        for circuit in circuit_list:
            counts = job.result().get_counts(circuit)
            outputs.append(self.sampler(counts))

        outputs = np.array(outputs).reshape(2, -1)

        grads = (outputs[0] - outputs[1]) / 2
        return grads

    def optimize(self, target_circuit, params, params_prev=None):
        self.optimizer.initialize([len(params)])
        error = self.evaluate(target_circuit, params,
                              params_prev, use_error_measure=True)
        counter = 1
        while error > self.tol:
            grads = self.gradient(target_circuit, params, params_prev)
            grads = self.optimizer([grads])[0]

            params[:] = params - self.optimizer.lr * grads

            error = self.evaluate(target_circuit, params,
                                  params_prev, use_error_measure=True)

            print(f"{counter}: {error}")
            counter += 1

    def predict(self, target_circuit):
        return self.evaluate(target_circuit, self.params[-1], use_error_measure=True)
