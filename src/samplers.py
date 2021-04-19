import numpy as np
import qiskit as qk
from math import ceil
from math import floor


class Parity():
    def __call__(self, counts):
        shots = sum(counts.values())
        output = 0
        for bitstring, samples in counts.items():
            if self._parity_of_bitstring(bitstring):
                output += samples

        output = output / shots

        return output

    def _parity_of_bitstring(self, bitstring):
        binary = [int(i) for i in bitstring]
        parity = sum(binary) % 2

        return parity


class ZeroBit():
    def __call__(self, counts):
        shots = sum(counts.values())
        output = 0
        for bitstring, samples in counts.items():
            if not "1" in bitstring:
                output += samples

        output = output / shots

        return output


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


class LastBit():
    def __call__(self, counts):
        shots = sum(counts.values())
        output = 0
        for bitstring, samples in counts.items():
            if bitstring[0] == "1":
                output += samples

        output = output / shots

        return output
