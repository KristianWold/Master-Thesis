import numpy as np
import qiskit as qk
from math import ceil
from math import floor


class Encoder():
    """Quantum circuit for encoding data by Pauli rotations.

    Parameters
    ----------
    mode : str
        Specify what Pauli rotations to use. "x", "y" and "z" corresponds
        to Rx, Ry and Rz rotation, respectivly.
    """

    def __init__(self, mode="y"):
        self.mode = mode

    def __call__(self, circuit, data_register, weight, data):
        """Apply gates to circuit for encoding data.

        Parameters
        ----------
        mode : str
            Specify what Pauli rotations to use. "x", "y" and "z" corresponds
            to Rx, Ry and Rz rotation, respectivly.
        """

        n_qubits = data_register.size
        n_features = data.shape[0]

        for i, x in enumerate(data):
            if self.mode == "x":
                circuit.rx(x, data_register[i])
            if self.mode == "y":
                circuit.ry(x, data_register[i])
            if self.mode == "z":
                circuit.h(data_register[i])
                circuit.rz(x, data_register[i])

        # apply Hadamard on latent qubits
        if n_qubits > n_features:
            for i in range(n_features, n_qubits):
                circuit.h(data_register[i])

        return circuit

    def calculate_n_weights(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_weights_per_target = 0


class RegularizedEncoder():
    def __init__(self, reg=True):
        self.reg = reg

    def __call__(self, circuit, data_register, data, params):
        n_qubits = data_register.size
        n_features = data.shape[0]
        n_params = params.shape[0]

        for i, x in enumerate(data):
            circuit.ry(x, data_register[i])

        if n_qubits > n_features:
            for i in range(n_features, n_qubits):
                circuit.h(data_register[i])

        for i, w in enumerate(params[:n_params // 2]):
            circuit.cx(data_register[2 * i], data_register[2 * i + 1])
            circuit.ry(w, data_register[2 * i])
            circuit.cx(data_register[2 * i], data_register[2 * i + 1])

        for i, w in enumerate(params[n_params // 2:]):
            circuit.cx(data_register[2 * i + 1], data_register[2 * i + 2])
            circuit.ry(w, data_register[2 * i + 1])
            circuit.cx(data_register[2 * i + 1], data_register[2 * i + 2])

        if self.reg:
            for i, x in enumerate(data):
                circuit.ry(-x, data_register[i])
        else:
            for i, x in enumerate(data):
                circuit.ry(x, data_register[i])

        if n_qubits > n_features:
            for i in range(n_features, n_qubits):
                circuit.h(data_register[i])

        return circuit


class ParallelEncoder():
    def __call__(self, circuit, data_register, ancilla, data):
        n_samples, n_features = data.shape
        n_ancilla = ancilla.size

        binary_ref = n_ancilla * [0]
        for i in range(n_samples):
            binary = interger_to_binary(i, n_ancilla)

            for j, (b, b_ref) in enumerate(zip(binary, binary_ref)):
                if b != b_ref:
                    circuit.x(ancilla[j])

            for j in range(n_features):
                circuit.cry(data[i, j], ancilla,
                            data_register[j])

            binary_ref = binary

        circuit.x(ancilla)

        for i in range(n_features - 1):
            circuit.cx(data_register[i], data_register[i + 1])

        return circuit


def amplitude_encoding(data, circuit, storage, inverse=False):
    N = data.shape[0]
    n = int(np.log2(N))

    circuit.ry(calculate_rotation(data, 0, 0), storage[0])
    for i in range(1, n):
        binary_ref = i * [0]
        circuit.x(storage[:i])

        for j in range(2**i):
            binary = interger_to_binary(j, i)

            for k, (b, b_ref) in enumerate(zip(binary, binary_ref)):
                if b != b_ref:
                    circuit.x(storage[k])

            circuit.mcry(calculate_rotation(data, i, j),
                         storage[:i], storage[i], q_ancillae=None)
            binary_ref = binary

    if inverse:
        circuit = circuit.inverse()

    return circuit


def basis_encoding(x):
    M, N = x.shape

    clas_reg = qk.ClassicalRegister(N)
    loading_reg = qk.QuantumRegister(N, name="loading")
    storage_reg = qk.QuantumRegister(N, name="storage")
    ancillas = qk.QuantumRegister(N, name="ancillas")
    branches = qk.QuantumRegister(2, name="branches")

    circuit = qk.QuantumCircuit(
        clas_reg, loading_reg, storage_reg, ancillas, branches)
    circuit.x(branches[1])

    for i in range(M):
        for j in range(N):
            if x[i, j] == 1:
                circuit.x(loading_reg[j])
                circuit.cx(branches[1], storage_reg[j])

        circuit.cx(branches[1], branches[0])
        theta = -1 / np.sqrt(M - i)
        circuit.cry(2 * np.arcsin(theta), branches[0], branches[1])

        circuit.toffoli(loading_reg, storage_reg, ancillas)
        circuit.x(loading_reg)
        circuit.x(storage_reg)
        circuit.toffoli(loading_reg, storage_reg, ancillas)

        circuit.mcx(ancillas, branches[0])

        circuit.toffoli(loading_reg, storage_reg, ancillas)
        circuit.x(loading_reg)
        circuit.x(storage_reg)
        circuit.toffoli(loading_reg, storage_reg, ancillas)

        for j in range(N):
            if x[i, j] == 1:
                circuit.x(loading_reg[j])
                circuit.cx(branches[1], storage_reg[j])

    circuit.measure(storage_reg, clas_reg)
    return circuit


# helper functions
################################################


def calculate_rotation(data, i, j):
    n = int(np.log2(len(data)))

    idx1 = (2 * j + 1) * 2**(n - i - 1)
    idx2 = (j + 1) * 2**(n - i)
    idx3 = j * 2**(n - i)
    idx4 = (j + 1) * 2**(n - i)

    if i == n - 1:
        a1 = data[idx1]
    else:
        a1 = np.sqrt(np.sum(np.abs(data[idx1:idx2])**2))

    a2 = np.sqrt(np.sum(np.abs(data[idx3:idx4])**2))
    if a2 == 0:
        return 0
    else:
        return 2 * np.arcsin(a1 / a2)


def interger_to_binary(integer, digits):
    binary = [int(b) for b in bin(integer)[2:]]
    binary = (digits - len(binary)) * [0] + binary
    return binary


def float_to_binary(x, digits=4):
    binary_rep = []
    if x > 0:
        binary_rep.append(0)
    else:
        binary_rep.append(1)

    x = abs(x)

    for i in range(digits):
        digit = floor(x * 2**(i))
        binary_rep.append(digit)
        x -= digit * 2**(-i)

    return binary_rep


def design_matrix_to_binary(X, digits=4):
    M, N = X.shape
    X_ = []

    for row in X:
        row_ = [float_to_binary(feature, digits) for feature in row]
        X_.append(row_)

    return np.array(X_).reshape(M, (digits + 1) * N)
