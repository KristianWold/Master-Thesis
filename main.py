import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
from qiskit import Aer
from tqdm.notebook import tqdm
import os

import qcnpy as qp

np.random.seed(42)
backend = Aer.get_backend('qasm_simulator')

layer1 = qp.QLayer(n_qubits=1,
                   n_features=1,
                   n_targets=3,
                   encoder=qp.Encoder(),
                   ansatz=qp.Ansatz(blocks=["entangle", "ry"]),
                   sampler=qp.Parity(),
                   scale=[-np.pi, np.pi],
                   backend=backend,
                   shots=0)

layer2 = qp.QLayer(n_qubits=3,
                   n_features=3,
                   n_targets=1,
                   encoder=qp.Encoder(),
                   ansatz=qp.Ansatz(blocks=["entangle", "ry"], reps=2),
                   sampler=qp.Parity(),
                   scale=1,
                   backend=backend,
                   shots=0)


layers = [layer1, layer2]

optimizer = qp.Adam(lr=0.1)
network = qp.NeuralNetwork(layers)

x = np.random.uniform(0, np.pi, 10).reshape(-1, 1)

y_pred = network.predict(x)
print(y_pred)
