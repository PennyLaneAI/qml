r""".. _mbqc:

Measurement-based quantum computation
=====================================

.. meta::
    :property="og:description": Learn about measurement-based quantum computation
    :property="og:image": https://pennylane.ai/qml/_images/mbqc.png

.. related::

    tutorial_toric_code Toric code 

*Authors: Joost Bus & Radoica Draškić. Posted: 08 August 2022. Last updated: 08 August 2022.*

"""

##############################################################################
#
# **Measurement-based quantum computing (MBQC)** also known as one-way quantum computing is an
# inventive approach to quantum computing that makes use of *off-line* entanglement as a resource
# for computation. A one-way quantum computer starts out with an entangled state, a so-called
# *cluster state*, and applies particular single-qubit measurements that correspond to the desired quantum circuit. In this context,
# off-line means that the entanglement is created independently from the rest of the
# computation, like how a blank sheet of paper is made separately from the text of a book. Coming
# from the gate-based model, this method might seem unintuitive to you at first, but the approaches
# can be proven to be equally powerful. In MBQC, the measurements *are* the computation and the
# entanglement of the cluster state is used as a resource.
#
# The structure of this demo will be as follows. First, we introduce the concept of a cluster
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
# .. figure:: ../demonstrations/mbqc/DALLE-mbqc.png
#    :align: center
#    :width: 60%
#
#    ..
#
#    In MBQC, seeing is computing!
#

##############################################################################
#
# Cluster states and graph states
# ----------------
#
# *Cluster states* are the universal substrate for measurement-based quantum computation
# [#OneWay2001]_. They are a special instance of *graph states*, a class of entangled multi-qubit
# states that can be represented by an undirected graph :math:`G = (V,E)` whose vertices are
# associated with qubits and the edges with entanglement between them. The associated quantum state
# reads as follows
#
# .. math::    |\Phi\rangle=\Pi_{(i,j)\in E}CZ_{ij}|+⟩^{\otimes n}.
#
# where :math:`n` is the number of qubits, :math:`CZ_{ij}` is the controlled-:math:`Z`` gate between
# qubits :math:`i` and :math:`j`, and :math:`|+\rangle = \frac{1}{\sqrt{2}}\big(|0\rangle + |1\rangle\big)` 
# is the :math:`+1`` eigenstate of the Pauli-:math:`X` operator.
#
# The distinction between a graph state and a cluster state is that the latter has the additional
# condition that the underlying graph has to be a lattice. Physically, this means that TODO ... 
# - require Pauli measurements 
# - fully disentangle
#
# We can also describe the creation of a cluster state in the gate-based model. Let us first
# define a graph we want to look at, and then construct a circuit in PennyLane to create the
# corresponding graph state.
#

import networkx as nx
import matplotlib.pyplot as plt

a, b = 1, 5  # dimensions of the graph (lattice)
G = nx.grid_graph(dim=[a, b])  # there are a * b qubits

plt.figure(figsize=(5, 1))
nx.draw(G, pos={node: node for node in G}, node_size=500, node_color="black")

##############################################################################
#
# This is a fairly simple cluster state, but we `will later see <#single-qubit-rotations>`_ how even
# this simple graph is useful for logical operations. Now that we have defined a graph, we can go ahead
# and define a circuit to prepare the cluster state.
#

import pennylane as qml

qubits = [str(node) for node in G.nodes]
dev = qml.device("default.qubit", wires=qubits)


@qml.qnode(dev)
def cluster_state():
    for node in qubits:
        qml.Hadamard(wires=[node])

    for edge in G.edges:
        i, j = edge
        qml.CZ(wires=[str(i), str(j)])

    return qml.state()


print(qml.draw(cluster_state)())

##############################################################################
#
# Observe that the structure of the circuit is fairly simple. It only requires Hadamard
# gates on each qubit and then a controlled-:math:`Z` gate between connected qubits. This part of
# the computation is not actually computing anything. In fact, aside from the width and depth of 
# the desired quantum circuit, the cluster state generation is essentially independent of the 
# calculation. If you have a reliable way of applying these two operations (Hadamard and 
# controlled-:math:`Z`), you are ready for the next step: worrying about conditional single-qubit 
# measurements.
#

##############################################################################
# Information propagation and teleportation
# ------------------------------------------
#
# Measurement-based quantum computation heavily relies on the idea of information propagation. In
# particular, we make use of a protocol called *quantum teleportation*, one of the driving concepts behind MBQC. Despite its esoteric name, quantum
# teleportation is very real and experimentally demonstrated multiple times in the last few decades 
# [#Hermans2022]_, [#Furusawa1998]_, [#Riebe2004]_, [#Nielsen1998]_. Moreover, it has related applications
# in safe communication protocols that are impossible with classical communication so it's certainly
# worth learning about. In this protocol, we do not transport matter but *information* between systems. Admittedly, it has a
# rather delusive name because it is not instantaneous but requires communication of additional classical information,
# which is naturally limited by the speed of light.
#
# .. figure:: ../demonstrations/mbqc/mbqc_info_flow.png
#    :align: center
#    :width: 75%
#
#    ..
#
#    The flow of information in a measurement-based quantum computation [#OneWay2001]_
#
# One-bit Teleportation
# `````````````````````
# Let us have a deeper look at the principles behind the protocol using a simple example of one-bit
# teleportation. We start with one qubit in the state :math:`|\psi\rangle` that we want to transfer
# to the second qubit initially in the state :math:`|0\rangle`. The figure below represents the 
# one-bit teleportation protocol. The green box represents the creation of a cluster state while
# the red box represents the measurement of a qubit with the appropriate correction applied to 
# the second qubit based on the single bit acquired through the measurement.
#
# .. figure:: ../demonstrations/mbqc/one-bit-teleportation.png
#    :align: center
#    :width: 75%
#
#    ..
#
# Let's implement one-bit teleportation in PennyLane. 

import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def one_bit_teleportation(input_state):
    # Prepare the input state
    qml.QubitStateVector(input_state, wires=0)

    # Prepare the cluster state
    qml.Hadamard(wires=1)
    qml.CZ(wires=[0, 1])

    # Measure the first qubit in the Pauli-X basis
    # and apply an X-gate conditioned on the outcome
    qml.Hadamard(wires=0)
    m = qml.measure(wires=[0])
    qml.cond(m == 1, qml.PauliX)(wires=1)
    qml.Hadamard(wires=1)

    # Return the density matrix of the output state
    return qml.density_matrix(wires=[1])


##############################################################################
#
# Note that we return a `density matrix <https://en.wikipedia.org/wiki/Density_matrix>`_ in the 
# function above because we are dealing with a *quantum channel* that includes a measurement on a 
# subset of the qubits. These matrices allow for description of mixed quantum states and are an
# extension of pure state vectors. Using this formalism, we can describe operations beyond unitaries
# such as the teleportation protocol.
#
# Now, let's prepare a random qubit state and see if the teleportation protocol is working as
# expected. To do so, we generate a random complex vector and normalize it to create a valid
# quantum state :math:`|\psi\rangle = \alpha |0\rangle + \beta |1\rangle`.
# We then apply the teleportation protocol and see if the resulting density matrix of the output
# state of the second qubit is the same as the input state of the first qubit.

# Define helper function for random input state on n qubits
def generate_random_state(n=1):
    input_state = np.random.random(2**n) + 1j * np.random.random(2**n)
    return input_state / np.linalg.norm(input_state)

# Generate a random input state for n=1 qubit
input_state = generate_random_state()

density_matrix = np.outer(input_state, np.conj(input_state))
density_matrix_mbqc = one_bit_teleportation(input_state)

np.allclose(density_matrix, density_matrix_mbqc)

##############################################################################
#
# As we can see, we found that the output state is identical to the input state!
#
# This protocol is one of the main ingredients of one-way quantum computing. Essentially, we
# propagate the information in one end of our cluster state to the other end by using the
# teleportation protocol. In addition, we can "write" our circuit onto the cluster state by
# choosing adaptive measurements. In the next section, we will see how we can actually do this.
#

##############################################################################
# Universality of MBQC
# ----------------------
# How do we know if this measurement-based scheme is just as powerful as its gate-based brother? We
# have to prove it! In particular, we want to show that a measurement-based quantum computer is a
# quantum Turing machine (QTM) also known as a universal quantum computer. To do this, we
# need to show 4 things [#OneWay2001]_:
#
#   1. How **information propagates** through the cluster state.
#
#   2. How arbitrary **single-qubit rotations** can be implemented.
#
#   3. How a **two-qubit gate** can be implemented in this scheme.
#
#   4. How to implement **arbitrary quantum circuits**.
#
# In the previous section, we have already seen how the quantum information propagates from one
# side of the cluster to the other. In this section, we will tackle the remaining parts concerning
# logical operations.
#

##############################################################################
#
# .. _single-qubit-rotations:
#
# Single-qubit rotations
# ```````````````````````
# Arbitrary single-qubit rotations are an essential operation for a universal quantum computer. In
# MBQC, we can implement these rotations by using the entanglement of the cluster state. Any
# single-qubit gate can be represented as a composition of three rotations along two different axes,
# for example :math:`U(\alpha, \beta, \gamma) = R_x(\gamma)R_z(\beta)R_x(\alpha)` where
# :math:`R_x` and :math:`R_z` represent rotations around the :math:`X` and :math:`Z` axis,
# respectively.
#
# We will see that in our measurement-based scheme, this operation can be implemented using a linear
# chain of 5 qubits prepared in a cluster state, as shown in the figure below [#MBQCRealization]_. The first qubit
# :math:`t_\mathrm{in}` is prepared in some input state :math:`|\psi_\mathrm{in}\rangle`,
# and we are interested in the final state of the output qubit :math:`t_\mathrm{out}`.
#
# .. figure:: ../demonstrations/mbqc/single-qubit-rotation.png
#    :align: center
#    :width: 75%
#
#    ..
#
# The input qubit :math:`t_\mathrm{in}`, together with the intermediate qubits :math:`a_1`,
# :math:`a_2`, and :math:`a_3` are then measured in the bases
#
# .. math::
#   \mathcal{B}_j(\theta_j) \equiv \left\{\frac{|0\rangle + e^{i\theta_j}|1\rangle}{\sqrt{2}},
#   \frac{|0\rangle - e^{i\theta_j}|1\rangle}{\sqrt{2}}\right\},
#
# where the angles :math:`\theta_j` depend on prior measurement outcomes and
# are given by
#
# .. math:: \theta_{\mathrm{in}} = 0, \qquad \theta_{1} = (-1)^{m_{\mathrm{in}} + 1} \alpha, \qquad
#   \theta_{2} = (-1)^{m_1} \beta, \quad \text{and} \quad \theta_{3} = (-1)^{m_{\mathrm{in}} + m_2} \gamma
#
# with :math:`m_{\mathrm{in}}, m_1, m_2, m_3 \in \{0, 1\}` being the measurement outcomes on nodes
# :math:`t_\mathrm{in}`, :math:`a_1`, :math:`a_2`, and :math:`a_3`, respectively. Note that the
# measurement basis is adaptive - the measurement on :math:`a_3` for example depends on the outcome
# of earlier measurements on the chain. After these operations, the state of qubit
# :math:`t_\mathrm{out}` is given by
#
# .. math:: |\psi_{\mathrm{out}}\rangle = \hat{U}(\alpha, \beta, \gamma)|\psi_{\mathrm{in}}\rangle 
#    = X^{m_1 + m_3}Z^{m_{\mathrm{in}} + m_2}U(\alpha, \beta, \gamma)
#    |\psi_{\mathrm{in}}\rangle.
#
# Now note that this unitary :math:`\hat{U}` is related to our desired unitary :math:`U` up to
# the first two Pauli terms. Luckily, we can correct for these additional Pauli gates by
# choosing the measurement basis of qubit :math:`t_\mathrm{out}` appropriately or correcting for them classically after
# the quantum computation.
#
# To demonstrate that this actually works, we will use PennyLane. For simplicity, we will just 
# show the ability will to perform single-axis rotations :math:`R_z(\theta)` and 
# :math:`R_x(\theta)` for arbitrary :math:`\theta \in [0, 2 \pi)`. Note that these two operations 
# plus the CNOT also constitute a universal gate set. 
# 
# To start of, we define the :math:`R_z(\theta)` gate using two qubits with the gate-based approach
# so we can later compare our MBQC approach to it.

dev = qml.device('default.qubit', wires=1)

@qml.qnode(dev)
def RZ(theta, input_state):
    # Prepare the input state
    qml.QubitStateVector(input_state, wires=0)
    
    # Perform the Rz rotation
    qml.RZ(theta, wires=0)

    # Return the density matrix of the output state
    return qml.density_matrix(wires=[0])

##############################################################################
#
# Now he have a base case, let's implement an :math:`R_z` gate on an arbitrary state 
# in the MBQC formalism.
#

mbqc_dev = qml.device("default.qubit", wires=2)

@qml.qnode(mbqc_dev)
def RZ_MBQC(theta, input_state):
    # Prepare the input state
    qml.QubitStateVector(input_state, wires=0)

    # Prepare the cluster state
    qml.Hadamard(wires=1)
    qml.CZ(wires=[0, 1])

    # Measure the first qubit an correct the state
    qml.RZ(theta, wires=0)
    qml.Hadamard(wires=0)
    m = qml.measure(wires=[0])

    qml.cond(m == 1, qml.PauliX)(wires=1)
    qml.Hadamard(wires=1)

    # Return the density matrix of the output state
    return qml.density_matrix(wires=[1])

##############################################################################
#
# Next, we will prepare a random input state and compare the two approaches.
#

# Generate a random input state
input_state = generate_random_state()
theta = 2 * np.pi * np.random.random()

np.allclose(RZ(theta, input_state), RZ_MBQC(theta, input_state))

##############################################################################
#
# Seems good! As we can see, the resulting states are practically the same as we wanted to show. 
# For the :math:`R_x(\theta)` gate we take a similar approach.
#

dev = qml.device('default.qubit', wires=1)

@qml.qnode(dev)
def RX(theta, input_state):
    # Prepare the input state
    qml.QubitStateVector(input_state, wires=0)
    
    # Perform the Rz rotation
    qml.RX(theta, wires=0)

    # Return the density matrix of the output state
    return qml.density_matrix(wires=[0])

mbqc_dev = qml.device("default.qubit", wires=3)

@qml.qnode(mbqc_dev)
def RX_MBQC(theta, input_state):
    # Prepare the input state
    qml.QubitStateVector(input_state, wires=0)

    # Prepare the cluster state
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.CZ(wires=[0, 1])
    qml.CZ(wires=[1, 2])

    # Measure the qubits and perform corrections
    qml.Hadamard(wires=0)
    m1 = qml.measure(wires=[0])

    qml.RZ(theta, wires=1)
    qml.cond(m1 == 1, qml.RX)(-2 * theta, wires=2)
    qml.Hadamard(wires=1)
    m2 = qml.measure(wires=[1])

    qml.cond(m2 == 1, qml.PauliX)(wires=2)
    qml.cond(m1 == 1, qml.PauliZ)(wires=2)
    
    # Return the density matrix of the output state
    return qml.density_matrix(wires=[2])

##############################################################################
#
# Finally, we again compare the two implementations with random state as an input.
#

# Generate a random input state
input_state = generate_random_state()
theta = 2 * np.pi * np.random.random()

np.allclose(RX(theta, input_state), RX_MBQC(theta, input_state))

##############################################################################
#
# Perfect! We have shown that we can implement any single-axis rotation on an arbitrary state in the 
# MBQC formalism. In the following section we will look at a two-qubit gate to complete our 
# universal gate set.

##############################################################################
# The two-qubit gate: CNOT
# ``````````````````````````
# The second ingredient for a universal quantum computing scheme is the two-qubit gate. Here, we will
# show how to perform a CNOT operation in the measurement-based framework. The input state is given on two qubits,
# control qubit :math:`c` and target qubit :math:`t_\mathrm{in}`. Preparing the cluster state shown in
# the figure below, and measuring qubits :math:`t_\mathrm{in}` and :math:`a` in the Hadamard basis,
# we implement the CNOT gate between qubits :math:`c` and :math:`t_\mathrm{out}` up to Pauli corrections [#MBQCRealization]_.
#
# .. figure:: ../demonstrations/mbqc/cnot.png
#    :align: center
#    :width: 50%
#
#    ..
#
# Let's see how one can do this in PennyLane.

# As sanity check, we will implement a CNOT gate on an arbitrary state.
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def CNOT(input_state):
    # Prepare the input state
    qml.QubitStateVector(input_state, wires=[0, 1])
    qml.CNOT(wires=[0, 1])

    return qml.density_matrix(wires=[0, 1])

# Let's now implement a CNOT in MBQC formalism!
# We will associate qubits c, t_in, a, and t_out in the figure with qubits 0, 1, 2, 3, respectively.
mbqc_dev = qml.device("default.qubit", wires=4)

@qml.qnode(mbqc_dev)
def CNOT_MBQC(input_state):
    # Prepare the input state
    qml.QubitStateVector(input_state, wires=[0, 1])

    # Prepare the cluster state
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    qml.CZ(wires=[2, 0])
    qml.CZ(wires=[2, 1])
    qml.CZ(wires=[2, 3])

    # Measure the qubits in the appropriate bases
    qml.Hadamard(wires=1)
    m1 = qml.measure(wires=[1])
    qml.Hadamard(wires=2)
    m2 = qml.measure(wires=[2])

    # Correct the state
    qml.cond(m1 == 1, qml.PauliZ)(wires=0)
    qml.cond(m2 == 1, qml.PauliX)(wires=3)
    qml.cond(m1 == 1, qml.PauliZ)(wires=3)

    # Return the density matrix of the output state
    return qml.density_matrix(wires=[0, 3])

##############################################################################
# Now let's prepare a random input state and check our implementation.

# Generate a random 2-qubit state
input_state = generate_random_state(n=2)

np.allclose(CNOT(input_state), CNOT_MBQC(input_state))

##############################################################################
# Arbitrary quantum circuits
# ```````````````````````````
# Once we have established the ability to implement arbitrary single-qubit rotations and a two-qubit
# gate, the final step is to show that we can implement arbitrary quantum circuits. To do so,
# we simply have to note that we have a *universal gate set* [#DiVincenzo]_. However, you might
# wonder - how many resources do these cluster states require?
#
# The number of qubits needed to construct a circuit can grow to be very large, as it depends on the
# amount of logical gates. At this point, it's good to reiterate that the entanglement of the cluster
# state is created *off-line*. This means that the entanglement is made independently from the
# computation, like how a blank sheet of paper is made separately from the text of a book.
# Interestingly enough, we do not have to prepare all the entanglement at once. Just like we can
# already start printing text upon the first few pages, we can apply measurements to one end of the
# cluster, while growing it at the same time, as shown in the figure below. That is, we can start
# printing the text on the first few pages while at the same time reloading the printer's paper
# tray!
#
# .. figure:: ../demonstrations/mbqc/measure_entangle.jpeg
#    :align: center
#    :width: 75%
#
#    ..
#
#   We can also consume the cluster state while we grow it. The blue qubits are in a cluster state, 
#   where the bonds between them represent entanglement. The gray qubits have been measured, 
#   destroying the entanglement and removing them from the cluster. At the same time, the green 
#   qubits are being added to the cluster by entangling them with it. Prior measurement outcomes 
#   determine the basis for future measurements [#OpticalQuantumComputing].
#
# This feature makes it particularly interesting for photonic-based quantum computers: we can use
# expendable qubits that don't have to stick around for the full calculation. If we can find a
# reliable way to produce qubits and stitch them together through entanglement, we can use it to
# produce our cluster state resource! Essentially, we need some kind of qubit factory and a
# stitching mechanism that puts it all together. The stitching mechanism depends on the physical 
# platform, and can for example be implemented with an Ising interaction [#OneWay2001]_ or in the
# context of photonics by interfering two modes with a beamsplitter.
#

##############################################################################
# Quantum error correction
# -------------------------
#
# To mitigate the physical errors that can (and will) happen during a quantum computation, we
# require some kind of error correction. Error correction is a technique for detecting errors and
# reconstructing the logical data without losing any information. It is not exclusive to quantum computing;
# it is also ubiquitous in "classical" computing and communication where one also has to deal `with 
# noise coming from the environment <https://www.youtube.com/watch?v=AaZ_RSt0KP8>`_. However, it is 
# a stringent requirement in the quantum realm as the systems one works with are much more 
# precarious and therefore prone to environmental factors, causing errors.
#
# Due to the peculiarities of quantum physics, we have to be careful when implementing error 
# correction. First of all, we can not simply look inside our quantum computer and see if an error occurred. This would collapse the
# wavefunction which carries valuable information. Secondly, we can not make copies of a quantum
# state to create redundancy. This is because of the no-cloning theorem. A whole research field is devoted
# to combating these challenges since Peter Shor published the seminal paper in 1995 [#ShorQEC1995]_.
# Full coverage of this topic is beyond the scope of this tutorial, but a good place to start is
# `Daniel Gottesman's thesis <https://arxiv.org/abs/quant-ph/9705052>`_ or `this blog post by
# Arthur Pesah <https://arthurpesah.me/blog/2022-01-25-intro-qec-1/>`_ for a more compact
# introduction. Instead, what we will do here is show how to implement error correction in the
# MBQC framework by using the surface code [#FowlerSurfaceCode]_ [#GoogleQEC2022]_ as
# an example.
#
# .. figure:: ../demonstrations/mbqc/surfacecode.jpg
#    :align: center
#    :width: 50%
#
#    ..
#
#    A distance-3 surface code
#
# In the measurement-based picture, quantum error correction requires a 3-dimensional cluster state
# [#XanaduBlueprint]_. The error correcting code that you want to implement dictates the structure
# of the cluster state. The cluster state that is associated with this code is known as the RHG lattice,
# named after its architects Raussendorf, Harrington, and Goyal. We can visualize this cluster
# state with FlamingPy.
#

from flamingpy.codes import SurfaceCode

code_distance = 3
RHG = SurfaceCode(code_distance)
# RHG.draw(backend="plotly", label=["type"], showbackground=True) #TODO: uncomment this line after merging FP#103

##############################################################################
#
# .. raw:: html
#    :file: ../demonstrations/mbqc/rhg-graph.html
#
#
# For the sake of intuition, you can think of the graph shown above as having two spatial dimensions (:math:`x`
# and :math:`y`) and one temporal dimension (:math:`z`). The cluster state alternates between *primal* and *dual sheets*, shown in the
# figure above on the :math:`xy`-plane. In principle, any quantum error correction code can be `foliated <https://arxiv.org/abs/1607.02579>` into 
# a graph state for measurement-based QEC [#FoliatedQuantumCodes]_. However, the foliations are particularly nice for `CSS 
# codes <https://errorcorrectionzoo.org/c/css>`_, named after Calderbank, Shor and Steane. CSS codes have stabilizers that exclusively contain
# :math:`X`-checks *or* :math:`Z`-checks, and include the *surface code* and *colour code* families. 
# For these CSS codes, you can view the primal and dual sheets as measuring the 
# :math:`Z`-stabilizers and :math:`X`-stabilizers, respectively.
#
# The computation and error correction are again performed with single-qubit measurements, as illustrated below.
# At each timestep, we measure all the qubits on one sheet of the lattice. The binary
# outcomes of these measurements determine the measurement bases for future measurements, and the
# last sheet of the lattice encodes the result of the computation which can be read out by yet
# another measurement.
#
# .. figure:: ../demonstrations/mbqc/gif_measuring.gif
#    :align: center
#    :width: 75%
#
#    ..
#
#    Performing an error corrected computation with measurements using the RHG lattice. [#XanaduBlueprint]_
#


##############################################################################
#
# Conclusion
# -------------------------------
#
# We have learned that a one-way quantum computer capable of cluster state 
# entanglement together with adaptive arbitrary single-qubit measurements allows for universal 
# quantum computation. The MBQC framework is a powerful quantum computing approach, particularly 
# useful in platforms that allow for many expendable qubits, and offers
# several advantages over the gate-based model. For example, it circumvents the need for applying 
# in-line entangling gates which are often the most noisy operations in gate-based quantum computers 
# with trapped-ions or superconducting circuits. Instead, the required entanglement is created 
# off-line which is often simpler to implement. 
#
# .. container:: alert alert-block alert-info
# 
#   **A system that is capable of creating an entangled cluster state and performing adaptive 
#   arbitrary single-qubit measurements on it is a universal quantum computer!**
#
# In this demo, we assumed that the system is capable of performing arbitrary
# single-qubit measurements. This is not a strict requirement, as one can acquire the same 
# capabilities by sprinkling *magic states* into the cluster state. A discussion of this topic is 
# beyond the scope of this tutorial, but a good place to start is `this 
# paper <https://arxiv.org/abs/quant-ph/0403025>`_ [#MagicStates].
#
# Xanadu's approach toward a universal quantum computer involves *continuous-variable* cluster states
# [#CV-MBQC]_. If you would like to learn more about the architecture, you can read our blueprint
# papers [#XanaduBlueprint]_ and [#XanaduPassiveArchitecture]_. We highly recommend watching `this 
# video <https://www.youtube.com/watch?v=SD6TH7GZ1rM&feature=emb_title>`_ outlining the main ideas. 
# On the hardware side, efforts are made to develop the necessary technology. This includes the recent `Borealis
# experiment <https://xanadu.ai/blog/beating-classical-computers-with-Borealis>`_ [#Borealis]_ where
# a 3-dimensional photonic graph state was created that was used to demonstrate quantum advantage.
#

##############################################################################
# References
# ------------
#
#
# .. [#OneWay2001]
#
#     Robert Raussendorf and Hans J. Briegel (2001) *A One-Way Quantum Computer*,
#     `Phys. Rev. Lett. 86, 5188
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188>`_.
#
# .. [#MBQCRealization]
#
#     Swapnil Nitin Shah (2021) *Realizations of Measurement Based Quantum Computing*,
#     `arXiv <https://arxiv.org/pdf/2112.11601.pdf>`_.
#
# .. [#XanaduBlueprint]
#
#   J. Eli Bourassa, Rafael N. Alexander, Michael Vasmer, Ashlesha Patil, Ilan Tzitrin, 
#   Takaya Matsuura, Daiqin Su, Ben Q. Baragiola, Saikat Guha, Guillaume Dauphinais, Krishna K. 
#   Sabapathy, Nicolas C. Menicucci, and Ish Dhand (2021) *Blueprint for a Scalable Photonic Fault-Tolerant Quantum Computer*,
#   `Quantum 5, 392 <https://quantum-journal.org/papers/q-2021-02-04-392/>`_.
#
# .. [#XanaduPassiveArchitecture]
#
#     Ilan Tzitrin, Takaya Matsuura, Rafael N. Alexander, Guillaume Dauphinais, J. Eli Bourassa,
#     Krishna K. Sabapathy, Nicolas C. Menicucci, and Ish Dhand (2021) *Fault-Tolerant Quantum Computation with Static Linear Optics*,
#     `PRX Quantum, Vol. 2, No. 4
#     <http://dx.doi.org/10.1103/PRXQuantum.2.040353>`_.
#
# .. [#ShorQEC1995]
#
#     Peter W. Shor (1995) *Scheme for reducing decoherence in quantum computer memory*,
#     `Physical Review A, Vol. 52, Iss. 4
#     <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.52.R2493>`_.
#
# .. [#LatticeSurgeryRaussendorf2018]
#
#     Daniel Herr, Alexandru Paler, Simon J. Devitt and Franco Nori (2018) *Lattice Surgery on the Raussendorf Lattice*,
#     `IOP Publishing 3, 3
#     <https://arxiv.org/abs/1711.04921>`_.
#
# .. [#FowlerSurfaceCode]
#
#     Austin G. Fowler, Matteo Mariantoni, John M. Martinis, Andrew N. Cleland (2012)
#     *Surface codes: Towards practical large-scale quantum computation*, `arXiv <https://arxiv.org/abs/1208.0928>`_.
#
# .. [#GoogleQEC2022]
#
#     Google Quantum AI (2022) *Suppressing quantum errors by scaling a surface code logical qubit*, `arXiv <https://arxiv.org/pdf/2207.06431.pdf>`_.
#
# .. [#CV-MBQC]
#
#     Nicolas C. Menicucci, Peter van Loock, Mile Gu, Christian Weedbrook, Timothy C. Ralph, and
#     Michael A. Nielsen (2006) *Universal Quantum Computation with Continuous-Variable Cluster States*,
#     `arXiv <https://arxiv.org/abs/quant-ph/0605198>`_.
#
# .. [#Borealis]
#
#     Lars S. Madsen, Fabian Laudenbach, Mohsen Falamarzi. Askarani, Fabien Rortais, Trevor Vincent,
#     Jacob F. F. Bulmer, Filippo M. Miatto, Leonhard Neuhaus, Lukas G. Helt, Matthew J. Collins,
#     Adriana E. Lita, Thomas Gerrits, Sae Woo Nam, Varun D. Vaidya, Matteo Menotti, Ish Dhand,
#     Zachary Vernon, Nicolás Quesada & Jonathan Lavoie (2022) *Quantum computational advantage with a
#     programmable photonic processor*, `Nature 606, 75-81
#     <https://www.nature.com/articles/s41586-022-04725-x>`_.
#
# .. [#DiVincenzo]
#
#    David P. DiVincenzo. (2000) *The Physical Implementation of Quantum Computation*,
#    `arXiv <https://arxiv.org/abs/quant-ph/0002077>`_.
#
# .. [#Furusawa1998]
#
#    A. Furusawa, J. L. Sørensen, S. L. Braunstein, C. A. Fuchs,H. J. Kimble, E. S. Polzik. (1998) 
#    *Unconditional Quantum Teleportation*, `Science Vol 282, Issue 5389 <https://www.science.org/doi/10.1126/science.282.5389.706>`_.
#
#
# .. [#Nielsen1998]
#
#   M. A. Nielsen, E. Knill & R. Laflamme. (1998) *Complete quantum teleportation using nuclear 
#   magnetic resonance*, `Nature volume 396, 52–55 <https://www.nature.com/articles/23891>`_.
#
# .. [#Hermans2022]
#
#    S. L. N. Hermans, M. Pompili, H. K. C. Beukers, S. Baier, J. Borregaard & R. Hanson. (2022) 
#    *Qubit teleportation between non-neighbouring nodes in a quantum network*, 
#    `Nature 605, 663–668 <https://www.nature.com/articles/s41586-022-04697-y>`_.
#
# .. [#Riebe2004]
#
#   M. Riebe, H. Häffner, C. F. Roos, W. Hänsel, J. Benhelm, G. P. T. Lancaster, T. W. Körber, 
#   C. Becher, F. Schmidt-Kaler, D. F. V. James & R. Blatt. (2002) *Deterministic quantum 
#   teleportation with atoms*, `Nature 429, 734-737 <https://www.nature.com/articles/nature02570>`_.
#
# .. [#FoliatedQuantumCodes]
#
#   A. Bolt, G. Duclos-Cianci, D. Poulin, T. M. Stace. (2016) *Foliated Quantum Codes*,
#    `arXiv <https://arxiv.org/abs/1607.02579>`_.
#
# .. [#MagicStates]
#
#   Sergey Bravyi and Alexei Kitaev. (2004) *Universal quantum computation with ideal Clifford gates and noisy ancillas*,
#   `arXiv <https://arxiv.org/abs/quant-ph/0403025>`_.
#
# .. [#OpticalQuantumComputing]
#
#   Jeremy L. O'Brien. (2007) *Optical quantum computing.*, `Science Vol. 318, Issue 5856, 1567-1570 
#   <https://www.science.org/doi/10.1126/science.1142892>`_.
#

##############################################################################
#
# About the authors
# ----------------
#
# .. bio:: Joost Bus
#    :photo: ../_static/authors/jbus.webp
#
#    I am a MSc student in Quantum Engineering at ETH Zürich who likes to explore how to wield quantum physics for technology. This summer, I am working with the architecture team on FlamingPy as a Xanadu resident.
#
# .. bio:: Radoica Draškić
#    :photo: ../_static/authors/radoica_draskic.jpg
#
#    I am a trained theoretical physicist and a wannabe computer scientist. I am currently working as a summer resident at Xanadu.
