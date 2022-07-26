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
# makes use of *offline* entanglement as a resource for computation. If you are more familiar with
# the gate-based model, this method might seem unintuitive to you at first, but the approaches can 
# be proven to be equally powerful. In a one-way quantum computer, we start out
# with an entangled state, a so-called cluster state, and apply particular single-qubit measurements
# that correspond to the desired quantum circuit. In MBQC, the measurements *are* the computation
# and the entanglement of the cluster state is used as a resource.
#
# The structure of this demo will be as follows. First of all, we introduce the concept of a cluster
# state, the substrate for measurement-based quantum computation. Then, we will move on to explain
# how to implement arbitrary quantum circuits, thus proving that MBQC is universal. Lastly, we will
# briefly touch upon how quantum error correction (QEC) is done in this scheme.
#
# Throughout this tutorial, we will explain the underlying concepts with the help of some code
# snippets using `PennyLane <https://pennylane.readthedocs.io/en/stable/>`_. In the section on QEC,
# we will also use Xanadu's quantum error correction simulation software 
# `FlamingPy <https://flamingpy.readthedocs.io/en/latest/>`_ developed by our architecture team 
# [#XanaduPassiveArchitecture]_.
#
#
#
# .. figure:: ../demonstrations/mbqc/mbqc_info_flow.png
#    :align: center
#    :width: 75%
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
# Information propagation and teleportation
# --------------------------------------
#
# Measurement-based quantum computation heavily relies on the idea of information propagation. In
# particular, we make use of a protocol called *teleportation*. Despite its esoteric name, quantum
# teleportation is very real and it's one of the driving concepts behind MBQC. Moreover, it has applications
# in safe communication protocols that are not possible with classical communication so it's certainly worth to learn about.
# In this protocol, we do transport matter but *information* between systems. Admittedly, it has a 
# rather delusive name because it is not instantaneous but requires additional classical information 
# to be communicated too, which is naturally limited by the speed of light.
#
# Quantum Teleportation
# `````````````````````
# Let us have a deeper look at the principles behind the protocol using a simple example of qubit
# teleportation. We start with a maximally entangled 2-qubit state, a Bell state. To relate this to
# cluster states - this is a cluster state with two nodes and one edge connecting them. The circuit 
# for teleportation is shown in the figure below.
#
# .. figure:: ../demonstrations/mbqc/one-bit-teleportation.png
#    :align: center
#    :width: 75%
#
#    ..

import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def one_bit_teleportation(input_state):
    # Prepare input state
    qml.QubitStateVector(input_state, wires=0)
    qml.Hadamard(wires=1)

    qml.CNOT(wires=[1, 0])
    
    # Measure the first qubit and apply an X-gate conditioned on the outcome
    m = qml.measure(wires = [0])
    qml.cond(m == 1, qml.PauliX)(wires=1)
    
    # Return the density matrix of the output state
    return qml.density_matrix(wires = [1])


##############################################################################
#
# Now let's prepare a random qubit state and see if the teleportation protocol is working as 
# expected. To do so, we generate two random complex numbers :math:`a` and :math:`b` and then 
# normalize them to create a valid qubit state :math:`|\psi\rangle = \alpha |0\rangle + \beta |1\rangle`. 
#

# Define a function to show the density matrix for easy comparison
qubit = qml.device("default.qubit", wires=1)

@qml.qnode(qubit)
def density_matrix(input_state):
    qml.QubitStateVector(input_state, wires=0)
    return qml.density_matrix(0)

# Generate random input state
a, b = np.random.random(2) + 1j*np.random.random(2)
norm = np.linalg.norm([a, b])

alpha = a / norm
beta = b / norm

input_state = np.array([alpha, beta])

density_matrix(input_state)

##############################################################################
# We then apply the teleportation protocol and see if the resulting density matrix of the output 
# state of the second qubit is the same as the input state of the first qubit.

one_bit_teleportation(input_state)

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
# 2. How arbitrary **single-qubit rotations** can be implemented.
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
# Quantum error correction
# ----------------
#
# To mitigate the physical errors that can (and will) happen during a quantum computation we
# require some kind of error correction. Error correction is a technique of detecting errors and 
# reconstructing the logical data without losing any information. It is not exclusive to quantum computing;
# it is also ubiquitous in `"classical" computing <https://www.youtube.com/watch?v=AaZ_RSt0KP8>`_
# and communication. However, it is a stringent requisite in the quantum realm as the systems one 
# works with are much more precarious and therefore prone to environmental factors, causing errors.
#
# Due to the peculiarities of quantum physics, we have to be careful when though. First of all, we can
# not simply look inside our quantum computer and see if an error occured. This would collapse the
# wavefunction which caries valuable information. Secondly, we can not make copies of a quantum
# state to create redundancy. This is because of the no-cloning theorem. A whole research field is devoted
# to combat these challenges since Peter Shor's published a seminal paper in 1995 [#ShorQEC1995]_. A 
# full coverage of this topic is beyond the scope of this tutorial, but a good place to start is 
# `Daniel Gottesman's thesis <https://arxiv.org/abs/quant-ph/9705052>`_ or `this blog post by 
# Arthur Pesah <https://arthurpesah.me/blog/2022-01-25-intro-qec-1/>`_ for a more compact 
# introduction. Instead, what we will do here is showing how to implement error correction in the 
# MBQC framework.
#
# In the measurement-based picture, quantum error correction requires a 3-dimensional cluster state
# [#XanaduBlueprint]_. The error correcting code that you want to implement dictates the structure
# of the cluster state. Let's see how we can implement the famous surface code [#FowlerSurfaceCode]_ [#GoogleQEC2022]_ as 
# an example. The cluster state that is associated with this code is known as the the RHG lattice, 
# named after its architects Raussendorf, Harrington, and Goyal. We can visualize this cluster 
# state with FlamingPy.
#

from flamingpy.codes import SurfaceCode
import matplotlib.pyplot as plt

code_distance = 3
RHG = SurfaceCode(code_distance)
# RHG.draw(backend="plotly") #TODO: uncomment this line after merging FP#103

##############################################################################
#
# .. raw:: html
#    :file: ../demonstrations/mbqc/rhg-graph.html
#

##############################################################################
#
# The actual computation is done by performing single-qubit measurements, as illustrated below. At 
# each timestep, we measure all the qubits on one sheet of the lattice. The binary outcomes of these 
# measurements determine the measurement bases for future measurements, and the last sheet of the 
# lattice encodes the result of the computation which can be read out by yet another measurement!
# 
#
# .. figure:: ../demonstrations/mbqc/gif_measuring.gif
#    :align: center
#    :width: 75%
#
#    ..
#
#    Performing a computation with measurements using the RHG lattice. [#XanaduBlueprint]_
#

##############################################################################
#
# Conclusion and further reading
# ---------------------------
#
# Xanadu approach towards a universal quantum computer involves continuous-variable cluster states 
# [#CV-MBQC]. If you would like to learn more about the architecture, you can read our blueprint 
# paper [#XanaduBlueprint]_ and [#XanaduPassiveArchitecture]_. In the meantime on the hardware side, 
# efforts are made to develop the necessary technology. This includes the recent `Borealis 
# experiment <https://xanadu.ai/blog/beating-classical-computers-with-Borealis>`_ [#Borealis]_ where a 
# 3-dimensional photonic graph state was created that was used to demonstrate quantum advantage. 
#

##############################################################################
# References
# ----------
#
#
# .. [#OneWay2001]
#
#     Robert Raussendorf and Hans J. Briegel (2001) *A One-Way Quantum Computer*,
#     `Phys. Rev. Lett. 86, 5188
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188>`__.
#
# .. [#XanaduBlueprint]
#
#     J. Eli Bourassa, Rafael N. Alexander, Michael Vasmer et al. (2021) *Blueprint for a Scalable Photonic Fault-Tolerant Quantum Computer*,
#     `Quantum 5, 392
#     <https://quantum-journal.org/papers/q-2021-02-04-392/>`__.
#
# .. [#XanaduPassiveArchitecture]
#
#     Ilan Tzitrin, Takaya Matsuura, Rafael N. Alexander, Guillaume Dauphinais, J. Eli Bourassa,
#     Krishna K. Sabapathy, Nicolas C. Menicucci, and Ish Dhand (2021) *Fault-Tolerant Quantum Computation with Static Linear Optics*,
#     `PRX Quantum, Vol. 2, No. 4
#     <http://dx.doi.org/10.1103/PRXQuantum.2.040353>`__.
#
# .. [#ShorQEC1995]
#
#     Peter W. Shor (1995) *Scheme for reducing decoherence in quantum computer memory*,
#     `Physical Review A, Vol. 52, Iss. 4
#     <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.52.R2493>`__.
#
# .. [#LatticeSurgeryRaussendorf2018]
#
#     Daniel Herr, Alexandru Paler, Simon J. Devitt and Franco Nori (2018) *Lattice Surgery on the Raussendorf Lattice*,
#     `IOP Publishing 3, 3
#     <https://arxiv.org/abs/1711.04921>`__.
#
# .. [#FowlerSurfaceCode]
#
#     Austin G. Fowler, Matteo Mariantoni, John M. Martinis, Andrew N. Cleland (2012) 
#     *Surface codes: Towards practical large-scale quantum computation*, `arXiv <https://arxiv.org/abs/1208.0928>`__.
#
# .. [#GoogleQEC2022]
#
#     Google Quantum AI (2022) *Suppressing quantum errors by scaling a surface code logical qubit*, `arXiv <https://arxiv.org/pdf/2207.06431.pdf>`__.
#
# .. [#CV-MBQC]
#
#     Nicolas C. Menicucci, Peter van Loock, Mile Gu, Christian Weedbrook, Timothy C. Ralph, and 
#     Michael A. Nielsen (2006) *Universal Quantum Computation with Continuous-Variable Cluster States*, 
#     `arXiv <https://arxiv.org/abs/quant-ph/0605198>`__.
#
# .. [#Borealis]
#
#    Lars S. Madsen, Fabian Laudenbach, Mohsen Falamarzi. Askarani, Fabien Rortais, Trevor Vincent, 
#    Jacob F. F. Bulmer, Filippo M. Miatto, Leonhard Neuhaus, Lukas G. Helt, Matthew J. Collins, 
#    Adriana E. Lita, Thomas Gerrits, Sae Woo Nam, Varun D. Vaidya, Matteo Menotti, Ish Dhand, 
#    Zachary Vernon, Nicolás Quesada & Jonathan Lavoie (2022) *Quantum computational advantage with a 
#    programmable photonic processor* 
#   `Nature 606, 75-81 <https://www.nature.com/articles/s41586-022-04725-x>`__.
#