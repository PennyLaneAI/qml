r""".. _mbqc:

Measurement-based quantum computation
=============================

.. meta::
    :property="og:description": Learn about measurement-based quantum computation
    :property="og:image": https://pennylane.ai/qml/_images/mbqc.png

*Author: Radoica Draskic & Joost Bus. Posted: Day Month 2022. Last updated: Day Month 2022.*

"""

##############################################################################
#
# Measurement-based quantum computation
# --------------
# Measurement-based quantum computing (MBQC) is a clever approach towards quantum computing that
# makes use of entanglement as a resource for computation. This method, also referred to as one-way
# quantum computing, is very dissimilar from the gate-based model but is universal nonetheless. In a
# one-way quantum computer, we start out with an entangled state, a so-called cluster state, and
# apply particular single-qubit measurements that correspond to the desired quantum circuit. In
# MBQC, the measurements are the computation and the entanglement of the cluster state is used as a
# resource.

##############################################################################
#
# Cluster states
# --------------
#
# .. figure:: ../demonstrations/mbqc/mbqc_blueprint.png
#    :align: center
#    :width: 60%
#
#    ..
#
#    Cluster state proposed in [#XanaduBlueprint2021]_
#
# To understand how MBQC qubits work, it's good to have an understanding of the cluster state. There
# is not one cluster state, but rather it’s a name for a class of highly entangled multi-qubit
# states. One example of a cluster state would be
# .. math::
#    |\psi⟩=\Pi_{(i,j)\in E(G)}CZ_{ij}|+⟩^{\otimes n},$$
# where :math:`G` is some graph and :math:`E(G)` is the set of its edges.

import networkx as nx

a, b = 10, 2
n = a * b  # number of qubits

G = nx.grid_graph(dim=[a, b])

# .. figure:: ../demonstrations/mbqc/measure_entangle.jpeg
#    :align: center
#    :width: 60%
#
#    ..

import pennylane as qml

qubits = [str(n) for n in G.nodes]

dev = qml.device("default.qubit", wires=qubits)


@qml.qnode(dev)
def cluster_state():
    for node in qubits:
        qml.Hadamard(wires=[node])

    for edge in G.edges:
        i, j = edge
        qml.CZ(wires=[str(i), str(j)])

    return qml.expval(qml.PauliZ(0))


print(qml.draw(cluster_state)())

##############################################################################
# Teleportation
# ----------
# .. figure:: ../demonstrations/mbqc/one-bit-teleportation.PNG
#    :align: center
#    :width: 60%
#
#    ..

import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=2)
input_state = np.array([1, -1], requires_grad=False) / np.sqrt(2)


@qml.qnode(dev)
def one_bit_teleportation(state, theta):
    # Prepare input state
    qml.QubitStateVector(state, wires=0)

    qml.Hadamard(wires=1)
    qml.CZ(wires=[0, 1])

    qml.Hadamard(wires=0)
    qml.PhaseShift(theta, wires=0)

    # measure the first qubit
    m = qml.measure(0)
    qml.cond(m == 1, qml.PauliX)(wires=1)

    return qml.density_matrix(wires=1)


print(qml.draw(one_bit_teleportation, expansion_strategy="device")(input_state, np.pi))


##############################################################################
# References
# ----------
#
#
# .. [#OneWay2021]
#
#     Robert Raussendorf and Hans J. Briegel (2021) "A One-Way Quantum Computer",
#     `Phys. Rev. Lett. 86, 5188
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188>`__.
#
# .. [#XanaduBlueprint2021]
#
#     J. Eli Bourassa, Rafael N. Alexander, Michael Vasmer et al. (2021) "Blueprint for a Scalable Photonic Fault-Tolerant Quantum Computer",
#     `Quantum 5, 392
#     <https://quantum-journal.org/papers/q-2021-02-04-392/>`__.
#
