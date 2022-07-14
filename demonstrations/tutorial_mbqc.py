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
# **Measurement-based quantum computing (MBQC)** is a clever approach towards quantum computing that
# makes use of *offline* entanglement as a resource for computation. This method, also referred to 
# as one-way quantum computing, seems very dissimilar from the gate-based model. However, they can 
# be proven to be equivalent and so both are universal. In a one-way quantum computer, we start out 
# with an entangled state, a so-called cluster state, and apply particular single-qubit measurements 
# that correspond to the desired quantum circuit. In MBQC, the measurements *are* the computation 
# and the entanglement of the cluster state is used as a resource.
#
# The structure of this demo will be as follows. First of all, we introduce the concept of a cluster 
# state, the substrate for measurement-based quantum computation. Then, we will move on to explain 
# how to implement arbitrary quantum circuits, thus proving that MBQC is universal. Lastly, we will 
# consider how fault-tolerance can be achieved in this scheme.
#
# Throughout this tutorial, we will explain the underlying concepts with the help of some code 
# snippets using `PennyLane <https://pennylane.readthedocs.io/en/stable/>`_ and Xanadu's quantum 
# error correction software `FlamingPy <https://flamingpy.readthedocs.io/en/latest/>`_ developed 
# by our architecture team [#XanaduPassiveArchitecture]_.
#
#
#
# .. figure:: ../demonstrations/mbqc/mbqc_info_flow.png
#    :align: center
#    :width: 60%
#
#    ..
#
#    The flow of information in a measurement-based quantum computation [#OneWay2001]_

##############################################################################
#
# Cluster states
# ----------------
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
#
# Now that we have a graph, we can construct the cluster state with PennyLane

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
# --------------------------------------
# 
# Measurement-based quantum computation heavily relies on the idea of information propagation. In 
# particular, we make use of a protocol called *teleportation*. Despite its esoteric name, quantum
# teleportation is very real and it's one of the driving concepts behind MBQC. Moreover, it has applications
# in safe communication protocols that are not possible with classical communication so it's certainly worth to learn about.
# In this protocol, we transport *information* between systems. Note that we do not transport matter. 
# Admittedly, it has a rather delusive name because it is not instantaneous but requires additional 
# classical information to be communicated too. This classical information transfer is naturally 
# limited by the speed of light. 
#
# Quantum Teleportation
# `````````````````````	
# Let us have a deeper look at the principles behind the protocol using a simple example of 
# teleportation. We start with a maximally entangled 2-qubit state, a Bell state. To relate this to 
# the cluster states - this is a cluster state with two nodes and one edge connecting them.
# 
# .. figure:: ../demonstrations/mbqc/one-bit-teleportation.png
#    :align: center
#    :width: 50%
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
# Information propagation
# ``````````````````````
# Essentially, we keep logical information in one end of our cluster state which we progagate to the 
# other end using the teleportation protocol. By choosing adaptive measurements, we can "write" our 
# circuit onto the cluster state. Later, we will see how we can actually do this. 
# 
# It's good to emphasize that the entanglement of the cluster state is created *off-line*. This 
# means that the entanglement is made independently from the computation, like how a blank sheet of 
# paper is made separately from the text of a book. Interestingly enough, we don not have to prepare 
# all the entanglement at once. Just like we can already start printing text upon the first few 
# pages, we can apply measurements to one end of the cluster, while growing it at the same time as 
# shown in the figure below. That is, we can start printing the text on the first few pages while at
# the same time reloading the printer's paper tray!
#
# .. figure:: ../demonstrations/mbqc/measure_entangle.jpeg
#    :align: center
#    :width: 75%
#
#    ..
#
# This feature makes it particularly interesting for photonic-based quantum computers: we can use
# expendable qubits that don't have to stick around for the full calculation. If we can find a 
# reliable way to produce qubits and stitch them together through entanglement, we can use it to 
# produce our cluster state resource! Essentially, we need some kind of qubit factory and a 
# stitching mechanism that puts it all together.
#

##############################################################################
# Universality of MBQC
# ----------------------
# How do we know if this measurement-based scheme is just as powerful as its gate-based brother? We 
# have to prove it! In particular, we want to show that a measurement-based quantum computer is a 
# called quantum Turing machine (WTM) also known as a universal quantum computer. To do this, we
# need to show 3 things
#
# 1. How **information propagates** through the cluster state.
#
# 2. How **arbitrary qubit rotations** can be implemented.
#
# 3. How a **two-qubit gate** can be implemented in this scheme.


##############################################################################
# Single-qubit rotations
# ```````````````````````
# Arbitrary single-qubit rotations are an essential operation for a universal quantum computer. In
# MBQC, we can implement these rotations by using the entanglement of the cluster state.

##############################################################################
# The two-qubit gate: CNOT
# ``````````````````````````	
# The last ingredient for a universal quantum computing scheme is the two-qubit gate. Here, we will
# show how to do a CNOT in the measurement-based framework.

##############################################################################
# Fault-tolerance
# ----------------
#
# To mitigate the physical errors that can (and will) happen during a quantum computation we 
# require error correction, in particular quantum error correction. This is a 
# 
# Error correction is not exclusively for quantum computers; it is also ubiquitous in `"classical" computing 
# <https://www.youtube.com/watch?v=AaZ_RSt0KP8>`_ and communication. However,  it is much more 
# essential in the quantum realm as the systems we work with are much precarious and prone to 
# environmental factors causing errors. This is a scheme that encodes the logical information in a larger system 
# 
# In the measurement-based picture, quantum error correction requires a 3-dimensional cluster state 
# [#XanaduBlueprint]_. The error correcting code that you want to implement dictates the structure 
# of the cluster state. For example, the cluster state that is associated with the surface code is 
# the RHG lattice, named after its architects Raussendorf, Harrington, and Goyal. We can visualize 
# this cluster state with FlamingPy.
#

from flamingpy.codes import SurfaceCode
import matplotlib.pyplot as plt

code_distance = 3
RHG = SurfaceCode(code_distance)
RHG.draw()

# Trying out if plotly offers interactivity on the website #TODO: remove this

import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species')
fig.show()

##############################################################################
#
# .. raw:: html
#    :file: ../demonstrations/mbqc/rhg-graph.html
#

##############################################################################
#
# Xanadu's path towards a fault-tolerant quantum computer is via a measurement-based scheme 
# with a 3-dimensional cluster state using photonics. The main ideas are presented in 
# [#XanaduBlueprint]_, and the corresponding cluster state looks like the figure below.
#
# .. figure:: ../demonstrations/mbqc/mbqc_blueprint.png
#    :align: center
#    :width: 75%
#
#    ..
#
#    Cluster state proposed in [#XanaduBlueprint]_
#
#
#

##############################################################################
#
# .. jupyter-execute::
#    import plotly.graph_objects as go
#    trace = go.Scatter(
#        x=[0, 1, 2, 3, 4, 5],
#        y=[0, 1, 4, 9, 16, 5],
#    )
#    layout = go.Layout(title='Growth')
#    figure = go.Figure(data=[trace], layout=layout)
#    figure.show()
#

##############################################################################
# References
# ----------
#
#
# .. [#OneWay2001]
#
#     Robert Raussendorf and Hans J. Briegel (2001) "A One-Way Quantum Computer",
#     `Phys. Rev. Lett. 86, 5188
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188>`__.
#
# .. [#XanaduBlueprint]
#
#     J. Eli Bourassa, Rafael N. Alexander, Michael Vasmer et al. (2021) "Blueprint for a Scalable Photonic Fault-Tolerant Quantum Computer",
#     `Quantum 5, 392
#     <https://quantum-journal.org/papers/q-2021-02-04-392/>`__.
#
# .. [#XanaduPassiveArchitecture]
#
#     Ilan Tzitrin, Takaya Matsuura, Rafael N. Alexander, Guillaume Dauphinais, J. Eli Bourassa, 
#     Krishna K. Sabapathy, Nicolas C. Menicucci, and Ish Dhand (2021) "Fault-Tolerant Quantum Computation with Static Linear Optics",
#     `PRX Quantum, Vol. 2, No. 4
#     <http://dx.doi.org/10.1103/PRXQuantum.2.040353>`__.
#
# .. [#LatticeSurgeryRaussendorf2018]
#
#     Daniel Herr, Alexandru Paler, Simon J. Devitt and Franco Nori (2018) "Lattice Surgery on the Raussendorf Lattice",
#     `IOP Publishing 3, 3
#     <https://arxiv.org/abs/1711.04921>`__.