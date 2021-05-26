import numpy as np
import qiskit as qk
from math import ceil
from math import floor


class Parity():
    def __call__(self, counts):
        shots = sum(counts.values())
        output = 0
        for bitstring, samples in counts.items():
            if self.parity_of_bitstring(bitstring):
                output += samples

        output = output / shots

        return output

    def observable(self, circuit, data_register):
        return circuit

    def parity_of_bitstring(self, bitstring):
        binary = [int(i) for i in bitstring]
        parity = sum(binary) % 2

        return parity


class ZeroBit():
    def __call__(self, counts):
        shots = sum(counts.values())
        output = 0
        for bitstring, samples in counts.items():
            if "1" not in bitstring:
                output += samples

        output = output / shots

        return 1 - output


class AverageBit():
    def __call__(self, counts):
        shots = sum(counts.values())
        output = 0
        for bitstring, samples in counts.items():
            bits = [int(bit) for bit in bitstring]
            average = sum(bits) / len(bits)
            output += average * samples

        output = output / shots

        return output

    def observable(self, circuit, data_register):

        return circuit


class LastBit():
    def __call__(self, counts):
        shots = sum(counts.values())
        output = 0
        for bitstring, samples in counts.items():
            if bitstring[0] == "1":
                output += samples

        output = output / shots

        return output

    def observable(self, circuit, data_register):

        return circuit
