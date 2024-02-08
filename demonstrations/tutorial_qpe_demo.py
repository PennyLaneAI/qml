r"""Intro to Quantum Phase Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Quantum Phase Estimation (QPE) algorithm is one of the most important tools in quantum
computing. Maybe the most important. It solves a deceptively simple task: given an eigenstate of a
unitary operator, find its eigenvalue. This demo explains the basics of the QPE algorithm. After
reading it, you will be able to understand the algorithm and how to implement it in PennyLane.

.. image:: qpe.png

Let’s code it in PennyLane!
"""

import pennylane as qml
import numpy as np

def U(wires):
    return qml.PhaseShift(2 * np.pi / 5, wires=wires)

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit_qpe(estimation_wires):
    # initialize to state |1>
    qml.PauliX(wires=0)

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(U(wires=0), control=estimation_wires)

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)

######################################################################
# Great, now show me the state,
# 
# .. math:: |\phi\rangle 
# 
# and the code.
# 

import matplotlib.pyplot as plt

estimation_wires = range(1, 5)

results = circuit_qpe(estimation_wires)

bit_strings = [f"0.{x:0{len(estimation_wires)}b}" for x in range(len(results))]

plt.bar(bit_strings, results)
plt.xlabel("phase")
plt.ylabel("probability")
plt.xticks(rotation="vertical")
plt.subplots_adjust(bottom=0.3)

plt.show()

######################################################################
# Conclusion
# ~~~~~~~~~~
# 
# This demo presented the “textbook” version of QPE. There are multiple variations, notably iterative
# QPE that uses a single estimation qubit, as well as Bayesian versions that saturate optimal
# prefactors appearing in the total cost. There are also mathematical subtleties about cost and errors
# that are important but out of scope for this demo.
# 
# Finally, there is extensive work on how to implement the unitaries themselves. In quantum chemistry,
# the main strategy is to encode a molecular Hamiltonian into a unitary such that the phases are
# invertible functions of the Hamiltonian eigenvalues. This can be done for instance through the
# exponential mapping, which can be implemented using Hamiltonian simulation techniques. More advanced
# techniques employ a qubitization-based encoding. QPE can then be used to estimate eigenvalues like
# ground-state energies by sampling them with respect to a distribution induced by the input state.
# 

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/photo.txt
