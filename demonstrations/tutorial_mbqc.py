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
# Measurement-based quantum computing (MBQC) is a clever approach towards quantum computing that
# makes use of *offline* entanglement as a resource for computation. This method, also referred to as one-way
# quantum computing, seems very dissimilar from the gate-based model. However, they can be proven to
# be equivalent and so both are universal. In a one-way quantum computer, we start out with an entangled state, a so-called cluster state, and
# apply particular single-qubit measurements that correspond to the desired quantum circuit. In
# MBQC, the measurements *are* the computation and the entanglement of the cluster state is used as a
# resource.
#
# The structure of this demo will be as follows. First we introduce the concept of a cluster state, 
# the substrate for measurement-based quantum computation. Then, we will move on to explain to 
# implement arbitrary quantum circuits, thus proving that MBQC is also universal. In particular, we 
# will illustrate
# 1. How **information propagates** through the cluster state.
# 2. How **arbitrary qubit rotations** can be implemented.
# 3. How a **two-qubit gate** can be implemented in this scheme.
#
# Once these operations are explained, we will move on to fault-tolerance and sketch how this can be
# achieved through lattice surgery. Throughout this tutorial, we will explain the underlying concepts 
# with the help of some `PennyLane<https://pennylane.readthedocs.io/en/stable/>`__ and 
# `FlamingPy<https://flamingpy.readthedocs.io/en/latest/>`__ code snippets. The latter is a relatively
# new software package developped by Xanadu's architecture team for quantum error correction research.

##############################################################################
#
# Cluster states
# =================
#
# Cluster states are the basis of measurement-based quantum computation. They are a special instance 
# of graph states, a class of entangled multi-qubit states that can be represented by an undirected 
# graph :math:`G = (V,E)` whose vertices are associated with qubits and the edges with entanglement 
# between them. The associated quantum state reads as follows
#
# .. math::    |\psi\rangle=\Pi_{(i,j)\in E(G)}CZ_{ij}|+⟩^{\otimes n}.
#
# The difference between a cluster state and a graph state is that the cluster state is ...
#
# 
# 

import networkx as nx

a, b = 5, 2
n = a * b  # number of qubits

G = nx.grid_graph(dim=[a, b])

##############################################################################
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
# Information propagation and Teleportation
# ======================================
# 
# Measurement-based quantum computation heavily relies on the idea of information propagation. In 
# particular, we make use of a protocol called *teleportation*. 
# Admittedly, this is a rather delusive name, but it is a powerful idea that can be used in many 
# quantum technologies like communication and, indeed, MBQC.
#
# Let us consider a simple example of teleportation. We start with a maximally entangled 2-qubit 
# state, a Bell state. To relate this to the cluster states - this is a cluster state with two nodes
# and one edge connecting them.
# 
# .. figure:: ../demonstrations/mbqc/one-bit-teleportation.png
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
#
# The information propagates through the cluster state as we apply measurements to one end. 
# Interestingly enough, we don not have to prepare all the entanglement at once...
#
# .. figure:: ../demonstrations/mbqc/mbqc_info_flow.png
#    :align: center
#    :width: 60%
#
#    ..
#
#    Cluster state proposed in [#XanaduBlueprint2021]_

##############################################################################
# Universality
# ======================
# Arbitrary single-qubit rotations are an essential operation for a universal quantum computer. In
# MBQC, we can implement arbitrary single-qubit rotations by using the entanglement of the cluster state.

##############################################################################
# Single-qubit rotations
# ----------
# Arbitrary single-qubit rotations are an essential operation for a universal quantum computer. In
# MBQC, we can implement arbitrary single-qubit rotations by using the entanglement of the cluster state.

##############################################################################
# The two-qubit gate: CNOT
# ----------
# The last ingredient for a universal quantum computing scheme is the two-qubit gate. Here, we will
# show how to do a CNOT in the measurement-based framework.

##############################################################################
# Fault-tolerance
# ======================
#
# To mitigate the risk of failure during a quantum computation we require quantum error correction.
# This requires a 3-dimensional cluster state [CITE]
#
# .. figure:: ../demonstrations/mbqc/mbqc_blueprint.png
#    :align: center
#    :width: 60%
#
#    ..
#
#    Cluster state proposed in [#XanaduBlueprint2021]_


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
# .. [#LatticeSurgeryRaussendorf2018]
#
#     Daniel Herr, Alexandru Paler, Simon J. Devitt and Franco Nori (2018) "Lattice Surgery on the Raussendorf Lattice",
#     `IOP Publishing 3, 3
#     <https://arxiv.org/abs/1711.04921>`__.