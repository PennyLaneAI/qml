"""
The Quantum Graph Recurrent Neural Network
===========================================

.. meta::
    :property="og:description": Using a quantum graph recurrent neural network to learn quantum dynamics.
    :property="og:image": https://pennylane.ai/qml/_images/qgrnn_thumbnail.png

*Author: Jack Ceroni. Posted: 27 July 2020. Last updated: 26 Oct 2020.*

"""

######################################################################
# This demonstration investigates quantum graph
# recurrent neural networks (QGRNN), which are the quantum analogue of a
# classical graph recurrent neural network, and a subclass of the more
# general quantum graph
# neural network ansatz. Both the QGNN and QGRNN were introduced in
# `this paper (2019) <https://arxiv.org/abs/1909.12264>`__.

######################################################################
# The Idea
# --------
#


######################################################################
# A graph is defined as a set of *nodes* along with a set of
# **edges**, which represent connections between nodes.
# Information can be encoded into graphs by assigning numbers
# to nodes and edges, which we call **weights**.
# It is usually convenient to think of a graph visually:
#
# .. image:: ../demonstrations/qgrnn/graph.png
#     :width: 70%
#     :align: center
#
# In recent years, the concept of a
# `graph neural network <https://arxiv.org/abs/1812.08434>`__ (GNN) has been
# receiving a lot of attention from the machine learning community.
# A GNN seeks
# to learn a representation (a mapping of data into a
# low-dimensional vector space) of a given graph with feature vectors assigned
# to nodes and edges. Each of the vectors in the learned
# representation preserves not only the features, but also the overall
# topology of the graph, i.e., which nodes are connected by edges. The
# quantum graph neural network attempts to do something similar, but for
# features that are quantum-mechanical; for instance, a
# collection of quantum states.
#


######################################################################
# Consider the class of qubit Hamiltonians that are *quadratic*, meaning that
# the terms of the Hamiltonian represent either interactions between two
# qubits, or the energy of individual qubits.
# This class of Hamiltonians is naturally described by graphs, with
# second-order terms between qubits corresponding to weighted edges between
# nodes, and first-order terms corresponding to node weights.
#
# A well known example of a quadratic Hamiltonian is the transverse-field
# Ising model, which is defined as
#
# .. math::
#
#     \hat{H}_{\text{Ising}}(\boldsymbol\theta) \ = \ \displaystyle\sum_{(i, j) \in E}
#     \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \
#     \displaystyle\sum_{i} X_{i},
#
# where :math:`\boldsymbol\theta \ = \ \{\theta^{(1)}, \ \theta^{(2)}\}`.
# In this Hamiltonian, the set :math:`E` that determines which pairs of qubits
# have :math:`ZZ` interactions can be represented by the set of edges for some graph. With
# the qubits as nodes, this graph is called the *interaction graph*.
# The :math:`\theta^{(1)}` parameters correspond to the edge weights and
# the :math:`\theta^{(2)}`
# parameters correspond to weights on the nodes.
#


######################################################################
# This result implies that we can think about *quantum circuits* with
# graph-theoretic properties. Recall that the time-evolution operator
# with respect to some Hamiltonian :math:`H` is defined as:
#
# .. math:: U \ = \ e^{-it H}.
#
# Thus, we have a clean way of taking quadratic Hamiltonians and turning
# them into unitaries (quantum circuits) that preserve the same correspondance to a graph.
# In the case of the Ising Hamiltonian, we have:
#
# .. math::
#
#     U_{\text{Ising}} \ = \ e^{-it \hat{H}_{\text{Ising}} (\boldsymbol\theta)} \ = \ \exp \Big[ -it
#     \Big( \displaystyle\sum_{(i, j) \in E} \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \
#     \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \ \displaystyle\sum_{i} X_{i} \Big) \Big]
#
# In general, this kind of unitary is very difficult to implement on a quantum computer.
# However, we can approximate it using the `Trotter-Suzuki decomposition
# <https://en.wikipedia.org/wiki/Time-evolving_block_decimation#The_Suzuki-Trotter_expansion>`__:
#
# .. math::
#
#     \exp \Big[ -it \Big( \displaystyle\sum_{(i, j) \in E} \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \
#     \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \ \displaystyle\sum_{i} X_{i} \Big) \Big]
#     \ \approx \ \displaystyle\prod_{k \ = \ 1}^{t / \Delta} \Bigg[ \displaystyle\prod_{j \ = \
#     1}^{Q} e^{-i \Delta \hat{H}_{\text{Ising}}^{j}(\boldsymbol\theta)} \Bigg]
#
# where :math:`\hat{H}_{\text{Ising}}^{j}(\boldsymbol\theta)` is the :math:`j`-th term of the
# Ising Hamiltonian and :math:`\Delta` is some small number.
#
# This circuit is a specific instance of the **Quantum Graph
# Recurrent Neural Network**, which in general is defined as a variational ansatz of
# the form
#
# .. math::
#
#     U_{H}(\boldsymbol\mu, \ \boldsymbol\gamma) \ = \ \displaystyle\prod_{i \ = \ 1}^{P} \Bigg[
#     \displaystyle\prod_{j \ = \ 1}^{Q} e^{-i \gamma_j H^{j}(\boldsymbol\mu)} \Bigg],
#
# for some parametrized quadratic Hamiltonian, :math:`H(\boldsymbol\mu)`.

######################################################################
# Using the QGRNN
# ^^^^^^^^^^^^^^^^
#

######################################################################
# Since the QGRNN ansatz is equivalent to the
# approximate time evolution of some quadratic Hamiltonian, we can use it
# to learn the dynamics of a quantum system.
#
# Continuing with the Ising model example, let's imagine we have some system
# governed by :math:`\hat{H}_{\text{Ising}}(\boldsymbol\alpha)` for an unknown set of
# target parameters,
# :math:`\boldsymbol\alpha` and an unknown interaction graph :math:`G`. Let's also
# suppose we have access to copies of some
# low-energy, non-ground state of the target Hamiltonian, :math:`|\psi_0\rangle`. In addition,
# we have access to a collection of time-evolved states,
# :math:`\{ |\psi(t_1)\rangle, \ |\psi(t_2)\rangle, \ ..., \ |\psi(t_N)\rangle \}`, defined by:
#
# .. math:: |\psi(t_k)\rangle \ = \ e^{-i t_k \hat{H}_{\text{Ising}}(\boldsymbol\alpha)} |\psi_0\rangle.
#
# We call the low-energy states and the collection of time-evolved states *quantum data*.
# From here, we randomly pick a number of time-evolved states
# from our collection. For any state that we choose, which is
# evolved to some time :math:`t_k`, we compare it
# to
#
# .. math::
#
#     U_{\hat{H}_{\text{Ising}}}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle \ \approx \ e^{-i t_k
#     \hat{H}_{\text{Ising}}(\boldsymbol\mu)} |\psi_0\rangle.
#
# This is done by feeding one of the copies of :math:`|\psi_0\rangle` into a quantum circuit
# with the QGRNN ansatz, with some guessed set of parameters :math:`\boldsymbol\mu`
# and a guessed interaction graph, :math:`G'`.
# We then use a classical optimizer to maximize the average
# "similarity" between the time-evolved states and the states prepared
# with the QGRNN.
#
# As the QGRNN states becomes more similar to
# each time-evolved state for each sampled time, it follows that
# :math:`\boldsymbol\mu \ \rightarrow \ \boldsymbol\alpha`
# and we are able to learn the unknown parameters of the Hamiltonian.
#
# .. figure:: ../demonstrations/qgrnn/qgrnn3.png
#     :width: 90%
#     :align: center
#
#     A visual representation of one execution of the QGRNN for one piece of quantum data.
#


######################################################################
# Learning an Ising Model with the QGRNN
# ---------------------------------------
#


######################################################################
# We now attempt to use the QGRNN to learn the parameters corresponding
# to an arbitrary transverse-field Ising model Hamiltonian.
#


######################################################################
# Getting Started
# ^^^^^^^^^^^^^^^^
#


######################################################################
# We begin by importing the necessary dependencies:
#


import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
import scipy
import networkx as nx
import copy


######################################################################
# We also define some fixed values that are used throughout
# the simulation.
#


qubit_number = 4
qubits = range(qubit_number)


######################################################################
# In this
# simulation, we don't have quantum data readily available to pass into
# the QGRNN, so we have to generate it ourselves. To do this, we must
# have knowledge of the target interaction graph and the target Hamiltonian.
#
# Let us use the following cyclic graph as the target interaction graph
# of the Ising Hamiltonian:
#


ising_graph = nx.cycle_graph(qubit_number)

print(f"Edges: {ising_graph.edges}")
nx.draw(ising_graph)


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      Edges: [(0, 1), (0, 3), (1, 2), (2, 3)]
#


#######################################################################
# .. figure:: ../demonstrations/qgrnn/graph1.png
#      :width: 70%
#      :align: center
#


######################################################################
# We can then initialize the “unknown” target parameters that describe the
# target Hamiltonian, :math:`\boldsymbol\alpha \ = \ \{\alpha^{(1)}, \ \alpha^{(2)}\}`.
# Recall from the introduction that we have defined our parametrized
# Ising Hamiltonian to be of the form:
#
# .. math::
#
#     \hat{H}_{\text{Ising}}(\boldsymbol\theta) \ = \ \displaystyle\sum_{(i, j) \in E}
#     \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \
#     \displaystyle\sum_{i} X_{i},
#
# where :math:`E` is the set of edges in the interaction graph, and
# :math:`X_i` and :math:`Z_i` are the Pauli-X and Pauli-Z on the
# :math:`i`-th qubit.
#
# For this tutorial, we choose the target parameters by sampling from
# a uniform probability distribution ranging from :math:`-2` to :math:`2`, with
# two-decimal precision.
#

target_weights = [0.56, 1.24, 1.67, -0.79]
target_bias = [-1.44, -1.43, 1.18, -0.93]


######################################################################
# In theory, these parameters can
# be any value we want, provided they are reasonably small enough that the QGRNN can reach them
# in a tractable number of optimization steps.
# In `matrix_params`, the first list represents the :math:`ZZ` interaction parameters and
# the second list represents the single-qubit `Z` parameters.
#
# Finally,
# we use this information to generate the matrix form of the
# Ising model Hamiltonian in the computational basis:
#


def create_hamiltonian_matrix(n_qubits, graph, weights, bias):

    full_matrix = np.zeros((2 ** n_qubits, 2 ** n_qubits))

    # Creates the interaction component of the Hamiltonian
    for i, edge in enumerate(graph.edges):
        interaction_term = 1
        for qubit in range(0, n_qubits):
            if qubit in edge:
                interaction_term = np.kron(interaction_term, qml.PauliZ.matrix)
            else:
                interaction_term = np.kron(interaction_term, np.identity(2))
        full_matrix += weights[i] * interaction_term

    # Creates the bias components of the matrix
    for i in range(0, n_qubits):
        z_term = x_term = 1
        for j in range(0, n_qubits):
            if j == i:
                z_term = np.kron(z_term, qml.PauliZ.matrix)
                x_term = np.kron(x_term, qml.PauliX.matrix)
            else:
                z_term = np.kron(z_term, np.identity(2))
                x_term = np.kron(x_term, np.identity(2))
        full_matrix += bias[i] * z_term + x_term

    return full_matrix


# Prints a visual representation of the Hamiltonian matrix
ham_matrix = create_hamiltonian_matrix(qubit_number, ising_graph, target_weights, target_bias)
plt.matshow(ham_matrix, cmap="hot")
plt.show()


######################################################################
# .. figure:: ../demonstrations/qgrnn/ising_hamiltonian.png
#      :width: 50%
#      :align: center
#


######################################################################
# Preparing Quantum Data
# ^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# The collection of quantum data needed to run the QGRNN has two components:
# (i) copies of a low-energy state, and (ii) a collection of time-evolved states, each of which are
# simply the low-energy state evolved to different times.
# The following is a low-energy state of the target Hamiltonian:
#

low_energy_state = [
    (-0.054661080280306085 + 0.016713907320174026j),
    (0.12290003656489545 - 0.03758500591109822j),
    (0.3649337966440005 - 0.11158863596657455j),
    (-0.8205175732627094 + 0.25093231967092877j),
    (0.010369790825776609 - 0.0031706387262686003j),
    (-0.02331544978544721 + 0.007129899300113728j),
    (-0.06923183949694546 + 0.0211684344103713j),
    (0.15566094863283836 - 0.04760201916285508j),
    (0.014520590919500158 - 0.004441887836078486j),
    (-0.032648113364535575 + 0.009988590222879195j),
    (-0.09694382811137187 + 0.02965579457620536j),
    (0.21796861485652747 - 0.06668776658411019j),
    (-0.0027547112135013247 + 0.0008426289322652901j),
    (0.006193695872468649 - 0.0018948418969390599j),
    (0.018391279795405405 - 0.005625722994009138j),
    (-0.041350974715649635 + 0.012650711602265649j),
]


######################################################################
# This state can be obtained by using a decoupled version of the
# :doc:`Variational Quantum Eigensolver </demos/tutorial_vqe>` algorithm (VQE).
# Essentially, we choose a
# VQE ansatz such that the circuit cannot learn the exact ground state,
# but it can get fairly close. Another way to arrive at the same result is
# to perform VQE with a reasonable ansatz, but to terminate the algorithm
# before it converges to the ground state. If we used the exact ground state
# :math:`|\psi_0\rangle`, the time-dependence would be trivial and the
# data would not provide enough information about the Hamiltonian parameters.
#
# We can verify that this is a low-energy
# state by numerically finding the lowest eigenvalue of the Hamiltonian
# and comparing it to the energy expectation of this low-energy state:
#


res = np.vdot(low_energy_state, (ham_matrix @ low_energy_state))
energy_exp = np.real_if_close(res)
print(f"Energy Expectation: {energy_exp}")


ground_state_energy = np.real_if_close(min(np.linalg.eig(ham_matrix)[0]))
print(f"Ground State Energy: {ground_state_energy}")


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      Energy Expectation: -7.244508985189116
#      Ground State Energy: -7.3306896612912436
#


######################################################################
# We have in fact found a low-energy, non-ground state,
# as the energy expectation is slightly greater than the energy of the true ground
# state. This, however, is only half of the information we need. We also require
# a collection of time-evolved, low-energy states.
# Evolving the low-energy state forward in time is fairly straightforward: all we
# have to do is multiply the initial state by a time-evolution unitary. This operation
# can be defined as a custom gate in PennyLane:
#


def state_evolve(hamiltonian, qubits, time):

    U = scipy.linalg.expm(-1j * hamiltonian * time)
    qml.QubitUnitary(U, wires=qubits)


######################################################################
# We don't actually generate time-evolved quantum data quite yet,
# but we now have all the pieces required for its preparation.
#


######################################################################
# Learning the Hamiltonian
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# With the quantum data defined, we are able to construct the QGRNN and
# learn the target Hamiltonian.
# Each of the exponentiated
# Hamiltonians in the QGRNN ansatz,
# :math:`\hat{H}^{j}_{\text{Ising}}(\boldsymbol\mu)`, are the
# :math:`ZZ`, :math:`Z`, and :math:`X` terms from the Ising
# Hamiltonian. This gives:
#


def qgrnn_layer(weights, bias, qubits, graph, trotter_step):

    # Applies a layer of RZZ gates (based on a graph)
    for i, edge in enumerate(graph.edges):
        qml.MultiRZ(2 * weights[i] * trotter_step, wires=(edge[0], edge[1]))

    # Applies a layer of RZ gates
    for i, qubit in enumerate(qubits):
        qml.RZ(2 * bias[i] * trotter_step, wires=qubit)

    # Applies a layer of RX gates
    for qubit in qubits:
        qml.RX(2 * trotter_step, wires=qubit)


######################################################################
# As was mentioned in the first section, the QGRNN has two
# registers. In one register, some piece of quantum data
# :math:`|\psi(t)\rangle` is prepared and in the other we have
# :math:`U_{H}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle`. We need a
# way to measure the similarity between these states.
# This can be done by using the fidelity, which is
# simply the modulus squared of the inner product between the states,
# :math:`| \langle \psi(t) | U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle |^2`.
# To calculate this value, we use a `SWAP
# test <https://en.wikipedia.org/wiki/Swap_test>`__ between the registers:
#


def swap_test(control, register1, register2):

    qml.Hadamard(wires=control)
    for reg1_qubit, reg2_qubit in zip(register1, register2):
        qml.CSWAP(wires=(control, reg1_qubit, reg2_qubit))
    qml.Hadamard(wires=control)


######################################################################
# After performing this procedure, the value returned from a measurement of the circuit is
# :math:`\langle Z \rangle`, with respect to the ``control`` qubit.
# The probability of measuring the :math:`|0\rangle` state
# in this control qubit is related to both the fidelity
# between registers and :math:`\langle Z \rangle`. Thus, with a bit of algebra,
# we find that :math:`\langle Z \rangle` is equal to the fidelity.
#
# Before creating the full QGRNN and the cost function, we
# define a few more fixed values. Among these is a "guessed"
# interaction graph, which we set to be a
# `complete graph <https://en.wikipedia.org/wiki/Complete_graph>`__.
#  This choice is motivated by the fact that any target interaction graph will be a subgraph
# of this initial guess. Part of the idea behind the QGRNN is that
# we don’t know the interaction graph, and it has to be learned. In this case, the graph
# is learned *automatically* as the target parameters are optimized. The
# :math:`\boldsymbol\mu` parameters that correspond to edges that don't exist in
# the target graph will simply approach :math:`0`.
#

# Defines some fixed values

reg1 = tuple(range(qubit_number))  # First qubit register
reg2 = tuple(range(qubit_number, 2 * qubit_number))  # Second qubit register

control = 2 * qubit_number  # Index of control qubit
trotter_step = 0.01  # Trotter step size

# Defines the interaction graph for the new qubit system

new_ising_graph = nx.complete_graph(reg2)

print(f"Edges: {new_ising_graph.edges}")
nx.draw(new_ising_graph)


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      Edges: [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]
#
# .. figure:: ../demonstrations/qgrnn/graph2.png
#      :width: 70%
#      :align: center
#


######################################################################
# With this done, we implement the QGRNN circuit for some given time value:
#


def qgrnn(weights, bias, time=None):

    # Prepares the low energy state in the two registers
    qml.QubitStateVector(np.kron(low_energy_state, low_energy_state), wires=reg1 + reg2)

    # Evolves the first qubit register with the time-evolution circuit to
    # prepare a piece of quantum data
    state_evolve(ham_matrix, reg1, time)

    # Applies the QGRNN layers to the second qubit register
    depth = time / trotter_step  # P = t/Delta
    for _ in range(0, int(depth)):
        qgrnn_layer(weights, bias, reg2, new_ising_graph, trotter_step)

    # Applies the SWAP test between the registers
    swap_test(control, reg1, reg2)

    # Returns the results of the SWAP test
    return qml.expval(qml.PauliZ(control))


######################################################################
# We have the full QGRNN circuit, but we still need to define a cost function.
# We know that
# :math:`| \langle \psi(t) | U_{H}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle |^2`
# approaches :math:`1` as the states become more similar and approaches
# :math:`0` as the states become orthogonal. Thus, we choose
# to minimize the quantity
# :math:`-| \langle \psi(t) | U_{H}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle |^2`.
# Since we are interested in calculating this value for many different
# pieces of quantum data, the final cost function is the average
# negative fidelity* between registers:
#
# .. math::
#
#     \mathcal{L}(\boldsymbol\mu, \ \Delta) \ = \ - \frac{1}{N} \displaystyle\sum_{i \ = \ 1}^{N} |
#     \langle \psi(t_i) | \ U_{H}(\boldsymbol\mu, \ \Delta) \ |\psi_0\rangle |^2,
#
# where we use :math:`N` pieces of quantum data.
#
# Before creating the cost function, we must define a few more fixed
# variables:
#

N = 15  # The number of pieces of quantum data that are used for each step
max_time = 0.1  # The maximum value of time that can be used for quantum data


######################################################################
# We then define the negative fidelity cost function:
#
rng = np.random.default_rng(seed=42)


def cost_function(weight_params, bias_params):

    # Randomly samples times at which the QGRNN runs
    times_sampled = rng.random(size=N) * max_time

    # Cycles through each of the sampled times and calculates the cost
    total_cost = 0
    for dt in times_sampled:
        result = qgrnn_qnode(weight_params, bias_params, time=dt)
        total_cost += -1 * result

    return total_cost / N


######################################################################
# Next we set up for optimization.
#

# Defines the new device
qgrnn_dev = qml.device("default.qubit", wires=2 * qubit_number + 1)

# Defines the new QNode
qgrnn_qnode = qml.QNode(qgrnn, qgrnn_dev)

# This is a LONG simulation.  If you just want to test that the code runs, try fewer steps
# steps = 10
steps = 300

optimizer = qml.AdamOptimizer(stepsize=0.5)

weights = rng.random(size=len(new_ising_graph.edges)) - 0.5
bias = rng.random(size=qubit_number) - 0.5

initial_weights = copy.copy(weights)
initial_bias = copy.copy(bias)

######################################################################
# All that remains is executing the optimization loop.

for i in range(0, steps):
    (weights, bias), cost = optimizer.step_and_cost(cost_function, weights, bias)

    # Prints the value of the cost function
    if i % 5 == 0:
        print(f"Cost at Step {i}: {cost}")
        print(f"Weights at Step {i}: {weights}")
        print(f"Bias at Step {i}: {bias}")
        print("---------------------------------------------")


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#       Cost at Step 0: -0.9803638573791904
#       Weights at Step 0: [-0.22604317  0.4388776   0.85859736  0.69736712  0.09417674 -0.02437703]
#       Bias at Step 0: [-0.23885902 -0.21393414  0.12811164  0.45038514]
#       ---------------------------------------------
#       Cost at Step 5: -0.9806500098112143
#       Weights at Step 5: [-1.29827824  1.52426565  1.81163837  1.86438612  1.04288314 -0.98516982]
#       Bias at Step 5: [-1.27134815 -1.36193505  1.31543373  1.32653424]
#       ---------------------------------------------
#       Cost at Step 10: -0.9648857984236838
#       Weights at Step 10: [-1.41068173  1.67469055  1.64410873  2.23403518  0.87027277 -0.84569027]
#       Bias at Step 10: [-1.28108438 -1.67672193  1.74541519  0.99186816]
#       ---------------------------------------------
#       Cost at Step 15: -0.9909075076678547
#       Weights at Step 15: [-0.99423966  1.31509032  0.97714182  2.16814495  0.20032301 -0.2265963 ]
#       Bias at Step 15: [-0.74022191 -1.53032257  1.76965387  0.16183654]
#       ---------------------------------------------
#       Cost at Step 20: -0.996600820482348
#       Weights at Step 20: [-0.46419217  0.84550568  0.43286203  1.9380202  -0.348272    0.25976054]
#       Bias at Step 20: [-0.12768221 -1.22298499  1.62611891 -0.4553839 ]
#       ---------------------------------------------
#       Cost at Step 25: -0.9926133497439015
#       Weights at Step 25: [-0.09015325  0.52906304  0.38274877  1.70919239 -0.41967469  0.26112821]
#       Bias at Step 25: [ 0.2326128  -0.93936924  1.45717265 -0.4540624 ]
#       ---------------------------------------------
#       Cost at Step 30: -0.998494633104677
#       Weights at Step 30: [ 0.0123472   0.47133081  0.7428572   1.56943045 -0.11998299 -0.09999201]
#       Bias at Step 30: [ 0.22093363 -0.77889245  1.33551884 -0.01237726]
#       ---------------------------------------------
#       Cost at Step 35: -0.9987664875905385
#       Weights at Step 35: [-0.04974879  0.55838325  1.09528857  1.50649468  0.13247182 -0.39189813]
#       Bias at Step 35: [ 0.01173716 -0.72349491  1.25413089  0.33198678]
#       ---------------------------------------------
#       Cost at Step 40: -0.9976572253882187
#       Weights at Step 40: [-0.14927     0.66157373  1.19794921  1.48297288  0.10734093 -0.3831593 ]
#       Bias at Step 40: [-0.21739349 -0.72790187  1.18290627  0.29425225]
#       ---------------------------------------------
#       Cost at Step 45: -0.9996712273475132
#       Weights at Step 45: [-0.20674402  0.70360037  1.0807707   1.48455205 -0.12668579 -0.15106377]
#       Bias at Step 45: [-0.35389918 -0.76974032  1.11755141 -0.02878183]
#       ---------------------------------------------
#       Cost at Step 50: -0.9995584075846883
#       Weights at Step 50: [-0.21856857  0.69313289  0.9951204   1.51327524 -0.27695939 -0.00304774]
#       Bias at Step 50: [-0.39501335 -0.83717604  1.08135172 -0.25421503]
#       ---------------------------------------------
#       Cost at Step 55: -0.9993417338158986
#       Weights at Step 55: [-0.19958537  0.65456745  1.0540154   1.56588539 -0.22806521 -0.06304728]
#       Bias at Step 55: [-0.37502663 -0.92029696  1.08093403 -0.2300362 ]
#       ---------------------------------------------
#       Cost at Step 60: -0.9997702657469879
#       Weights at Step 60: [-0.15016116  0.59649054  1.17917889  1.61891613 -0.0895     -0.22355593]
#       Bias at Step 60: [-0.3186739  -0.99107611  1.09766839 -0.08892556]
#       ---------------------------------------------
#       Cost at Step 65: -0.9996436456556428
#       Weights at Step 65: [-0.08220889  0.53561352  1.22392446  1.64470734 -0.04968037 -0.29147776]
#       Bias at Step 65: [-0.26354431 -1.02050635  1.10560821 -0.06357961]
#       ---------------------------------------------
#       Cost at Step 70: -0.999782972597875
#       Weights at Step 70: [-0.02812171  0.50207641  1.18712749  1.64861258 -0.12278163 -0.24907531]
#       Bias at Step 70: [-0.25578322 -1.01899708  1.10378897 -0.1755329 ]
#       ---------------------------------------------
#       Cost at Step 75: -0.9998161136786834
#       Weights at Step 75: [-0.01479171  0.51567978  1.17552912  1.65394365 -0.18664303 -0.21758165]
#       Bias at Step 75: [-0.32014432 -1.01791815  1.10519351 -0.27848449]
#       ---------------------------------------------
#       Cost at Step 80: -0.9998843655288555
#       Weights at Step 80: [-0.01884375  0.54281647  1.22344387  1.66582127 -0.17956552 -0.25497115]
#       Bias at Step 80: [-0.40362773 -1.02780124  1.10836139 -0.29496652]
#       ---------------------------------------------
#       Cost at Step 85: -0.9999098350330778
#       Weights at Step 85: [-0.00379234  0.54313665  1.2733362   1.67338414 -0.15161841 -0.30903377]
#       Bias at Step 85: [-0.45061752 -1.03818189  1.10130953 -0.28564262]
#       ---------------------------------------------
#       Cost at Step 90: -0.9999044415579313
#       Weights at Step 90: [ 0.04088165  0.50692558  1.28240681  1.67288196 -0.15109114 -0.33205828]
#       Bias at Step 90: [-0.4519275  -1.04504232  1.07941985 -0.3132239 ]
#       ---------------------------------------------
#       Cost at Step 95: -0.999906122283825
#       Weights at Step 95: [ 0.08554249  0.46656083  1.26840277  1.6755038  -0.16748469 -0.33495651]
#       Bias at Step 95: [-0.45081552 -1.0592657   1.05546404 -0.36819017]
#       ---------------------------------------------
#       Cost at Step 100: -0.9999067038489897
#       Weights at Step 100: [ 0.10504756  0.45151576  1.27495676  1.68957857 -0.16364088 -0.35685113]
#       Bias at Step 100: [-0.48320092 -1.0860277   1.04231719 -0.40281373]
#       ---------------------------------------------
#       Cost at Step 105: -0.9999258608983025
#       Weights at Step 105: [ 0.11286395  0.45096054  1.29489123  1.70583499 -0.14822871 -0.39062082]
#       Bias at Step 105: [-0.53290732 -1.11214824  1.03558616 -0.42231884]
#       ---------------------------------------------
#       Cost at Step 110: -0.9999159642471992
#       Weights at Step 110: [ 0.13106534  0.44316269  1.30188589  1.7135758  -0.14449277 -0.41395624]
#       Bias at Step 110: [-0.57365278 -1.12589957  1.02588522 -0.45380079]
#       ---------------------------------------------
#       Cost at Step 115: -0.9999295051359973
#       Weights at Step 115: [ 0.16456919  0.42229382  1.29615959  1.71331202 -0.14936482 -0.43044986]
#       Bias at Step 115: [-0.60049609 -1.12909729  1.01159669 -0.4958413 ]
#       ---------------------------------------------
#       Cost at Step 120: -0.9999329105144759
#       Weights at Step 120: [ 0.19394551  0.40471337  1.2991934   1.71575281 -0.1428606  -0.45732438]
#       Bias at Step 120: [-0.63053342 -1.13507053  1.00043562 -0.52433156]
#       ---------------------------------------------
#       Cost at Step 125: -0.9999672151181849
#       Weights at Step 125: [ 0.21407342  0.39421863  1.30670262  1.72320127 -0.13102448 -0.48725764]
#       Bias at Step 125: [-0.6687698  -1.14749235  0.99304941 -0.54780452]
#       ---------------------------------------------
#       Cost at Step 130: -0.9999498639111295
#       Weights at Step 130: [ 0.23251895  0.38323767  1.30565485  1.73043603 -0.12634796 -0.50754102]
#       Bias at Step 130: [-0.70519062 -1.16065043  0.9853521  -0.57984279]
#       ---------------------------------------------
#       Cost at Step 135: -0.9999460668283626
#       Weights at Step 135: [ 0.25315     0.36897848  1.30157699  1.73591416 -0.12368359 -0.52479079]
#       Bias at Step 135: [-0.73738032 -1.17210895  0.97628525 -0.6137189 ]
#       ---------------------------------------------
#       Cost at Step 140: -0.9999760263168519
#       Weights at Step 140: [ 0.27319219  0.35494171  1.30465117  1.74054957 -0.11422205 -0.5482196 ]
#       Bias at Step 140: [-0.76767167 -1.18178289  0.96794962 -0.63664615]
#       ---------------------------------------------
#       Cost at Step 145: -0.9999620825798221
#       Weights at Step 145: [ 0.29291312  0.34115888  1.30743903  1.7441703  -0.10602196 -0.56989588]
#       Bias at Step 145: [-0.79755643 -1.18957764  0.96017139 -0.65982932]
#       ---------------------------------------------
#       Cost at Step 150: -0.9999528162713989
#       Weights at Step 150: [ 0.31193841  0.32791071  1.30767856  1.747724   -0.10220887 -0.58675715]
#       Bias at Step 150: [-0.8277568  -1.19660646  0.9533145  -0.68795875]
#       ---------------------------------------------
#       Cost at Step 155: -0.9999676206975212
#       Weights at Step 155: [ 0.32929841  0.31576924  1.3101255   1.75246696 -0.0974324  -0.60396184]
#       Bias at Step 155: [-0.85776514 -1.2043529   0.94822294 -0.71402521]
#       ---------------------------------------------
#       Cost at Step 160: -0.9999688897065233
#       Weights at Step 160: [ 0.34617048  0.30341272  1.31549083  1.75789129 -0.09127326 -0.62192276]
#       Bias at Step 160: [-0.88499311 -1.21222407  0.94414483 -0.73671672]
#       ---------------------------------------------
#       Cost at Step 165: -0.9999678828349428
#       Weights at Step 165: [ 0.36424544  0.28904681  1.3183802   1.76299152 -0.08653562 -0.6379972 ]
#       Bias at Step 165: [-0.91076279 -1.21990516  0.93990101 -0.76174131]
#       ---------------------------------------------
#       Cost at Step 170: -0.9999718518505569
#       Weights at Step 170: [ 0.38058818  0.27530965  1.31811484  1.76729325 -0.08126922 -0.65326086]
#       Bias at Step 170: [-0.93762726 -1.22741419  0.93605274 -0.78587399]
#       ---------------------------------------------
#       Cost at Step 175: -0.999969992884057
#       Weights at Step 175: [ 0.39531011  0.26293084  1.32005541  1.7712584  -0.0738556  -0.67032688]
#       Bias at Step 175: [-0.96623738 -1.23426398  0.9328937  -0.80654326]
#       ---------------------------------------------
#       Cost at Step 180: -0.9999810640341682
#       Weights at Step 180: [ 0.41111346  0.24939364  1.32150139  1.7744836  -0.06804758 -0.68598269]
#       Bias at Step 180: [-0.99446201 -1.23978732  0.92974762 -0.82904396]
#       ---------------------------------------------
#       Cost at Step 185: -0.9999732059499762
#       Weights at Step 185: [ 0.42663809  0.23569253  1.323561    1.77787964 -0.06331664 -0.7001218 ]
#       Bias at Step 185: [-1.02027229 -1.24495257  0.9273182  -0.8511393 ]
#       ---------------------------------------------
#       Cost at Step 190: -0.9999904686011623
#       Weights at Step 190: [ 0.4407972   0.22274166  1.32741124  1.78217538 -0.0581927  -0.71407865]
#       Bias at Step 190: [-1.04467037 -1.2507425   0.92600868 -0.87121712]
#       ---------------------------------------------
#       Cost at Step 195: -0.9999828831078126
#       Weights at Step 195: [ 0.45318723  0.21066667  1.32928285  1.78607162 -0.05386645 -0.72584502]
#       Bias at Step 195: [-1.06684646 -1.25630331  0.92524928 -0.88990376]
#       ---------------------------------------------
#       Cost at Step 200: -0.9999814691774112
#       Weights at Step 200: [ 0.46483484  0.19879592  1.33066437  1.78942676 -0.04940039 -0.73713324]
#       Bias at Step 200: [-1.08841247 -1.26129482  0.92474857 -0.90725602]
#       ---------------------------------------------
#       Cost at Step 205: -0.9999831444629906
#       Weights at Step 205: [ 0.47584426  0.18719266  1.33154192  1.79202379 -0.04462708 -0.74822003]
#       Bias at Step 205: [-1.10977785 -1.265427    0.92453063 -0.92310712]
#       ---------------------------------------------
#       Cost at Step 210: -0.9999790517608405
#       Weights at Step 210: [ 0.48584547  0.17597834  1.33028455  1.79378517 -0.03978138 -0.75852048]
#       Bias at Step 210: [-1.13101187 -1.26898531  0.92482346 -0.93780681]
#       ---------------------------------------------
#       Cost at Step 215: -0.9999847111950139
#       Weights at Step 215: [ 0.4946749   0.16562713  1.32915967  1.79528287 -0.03462316 -0.76851533]
#       Bias at Step 215: [-1.15132516 -1.27221485  0.92584491 -0.9503349 ]
#       ---------------------------------------------
#       Cost at Step 220: -0.9999855424488905
#       Weights at Step 220: [ 0.50303178  0.15574802  1.32886423  1.79677025 -0.03058318 -0.77711746]
#       Bias at Step 220: [-1.16955409 -1.27502675  0.92729975 -0.96212831]
#       ---------------------------------------------
#       Cost at Step 225: -0.9999863772257213
#       Weights at Step 225: [ 0.51097377  0.14603986  1.32919454  1.79836519 -0.02733105 -0.78463519]
#       Bias at Step 225: [-1.18628365 -1.27783948  0.92913454 -0.97340009]
#       ---------------------------------------------
#       Cost at Step 230: -0.9999898484097556
#       Weights at Step 230: [ 0.51836647  0.13626055  1.32967368  1.80000495 -0.02394918 -0.79187192]
#       Bias at Step 230: [-1.20276401 -1.28103879  0.93145415 -0.98384786]
#       ---------------------------------------------
#       Cost at Step 235: -0.9999890657624451
#       Weights at Step 235: [ 0.52564729  0.12600301  1.32958604  1.80138672 -0.02082854 -0.79856878]
#       Bias at Step 235: [-1.2195148  -1.28431275  0.93422156 -0.9943289 ]
#       ---------------------------------------------
#       Cost at Step 240: -0.999988297622605
#       Weights at Step 240: [ 0.53229253  0.11610906  1.32908336  1.80229993 -0.01749836 -0.80503539]
#       Bias at Step 240: [-1.23599684 -1.28731013  0.93747248 -1.00346756]
#       ---------------------------------------------
#       Cost at Step 245: -0.9999882440932375
#       Weights at Step 245: [ 0.53847247  0.10723924  1.33070351  1.80329887 -0.01467023 -0.81108244]
#       Bias at Step 245: [-1.25069646 -1.28978222  0.94081211 -1.01123413]
#       ---------------------------------------------
#       Cost at Step 250: -0.9999868214455343
#       Weights at Step 250: [ 0.54460039  0.0980518   1.33056073  1.80410968 -0.01354454 -0.81496581]
#       Bias at Step 250: [-1.26394352 -1.29235223  0.94434983 -1.02022419]
#       ---------------------------------------------
#       Cost at Step 255: -0.9999884214814994
#       Weights at Step 255: [ 0.54943379  0.08996909  1.33186446  1.8051759  -0.01056943 -0.82027142]
#       Bias at Step 255: [-1.27708148 -1.29545081  0.94840617 -1.02592635]
#       ---------------------------------------------
#       Cost at Step 260: -0.9999905893984568
#       Weights at Step 260: [ 0.55392271  0.08211047  1.33002746  1.8051647  -0.00912692 -0.82330494]
#       Bias at Step 260: [-1.28877696 -1.29778044  0.9520992  -1.03219995]
#       ---------------------------------------------
#       Cost at Step 265: -0.9999894660037792
#       Weights at Step 265: [ 0.55772452  0.07551996  1.33104998  1.80525639 -0.00672651 -0.82745976]
#       Bias at Step 265: [-1.29986939 -1.29978116  0.95585094 -1.03589225]
#       ---------------------------------------------
#       Cost at Step 270: -0.9999873354125405
#       Weights at Step 270: [ 0.56170933  0.06903136  1.33175345  1.80537894 -0.00624539 -0.82966868]
#       Bias at Step 270: [-1.30907139 -1.30160912  0.9593855  -1.04092885]
#       ---------------------------------------------
#       Cost at Step 275: -0.9999910126657187
#       Weights at Step 275: [ 0.56476346  0.06288961  1.33121415  1.80555827 -0.00483336 -0.83207027]
#       Bias at Step 275: [-1.3181107  -1.30426179  0.96340533 -1.04444787]
#       ---------------------------------------------
#       Cost at Step 280: -0.9999911807478907
#       Weights at Step 280: [ 0.56790545  0.05710628  1.33293783  1.80572737 -0.00337721 -0.83486294]
#       Bias at Step 280: [-1.32665878 -1.30655911  0.96725682 -1.04712595]
#       ---------------------------------------------
#       Cost at Step 285: -0.9999870533227305
#       Weights at Step 285: [ 0.57142315  0.05115447  1.33329561  1.80547158 -0.00412561 -0.83538511]
#       Bias at Step 285: [-1.33391608 -1.30836527  0.97068764 -1.05192006]
#       ---------------------------------------------
#       Cost at Step 290: -0.999991853358171
#       Weights at Step 290: [ 0.57314839  0.04641148  1.33277082  1.80515273 -0.00185471 -0.83794182]
#       Bias at Step 290: [-1.3420092  -1.3110848   0.97494496 -1.05274072]
#       ---------------------------------------------
#       Cost at Step 295: -0.9999907581161837
#       Weights at Step 295: [ 5.75810409e-01  4.17373794e-02  1.33385607e+00  1.80469466e+00
#        -1.37560830e-03 -8.39439431e-01]
#       Bias at Step 295: [-1.34861394 -1.31269908  0.97854738 -1.05489467]
#       ---------------------------------------------

######################################################################
# With the learned parameters, we construct a visual representation
# of the Hamiltonian to which they correspond and compare it to the
# target Hamiltonian, and the initial guessed Hamiltonian:
#

new_ham_matrix = create_hamiltonian_matrix(
    qubit_number, nx.complete_graph(qubit_number), weights, bias
)

init_ham = create_hamiltonian_matrix(
    qubit_number, nx.complete_graph(qubit_number), initial_weights, initial_bias
)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 6))

axes[0].matshow(ham_matrix, vmin=-7, vmax=7, cmap="hot")
axes[0].set_title("Target", y=1.13)

axes[1].matshow(init_ham, vmin=-7, vmax=7, cmap="hot")
axes[1].set_title("Initial", y=1.13)

axes[2].matshow(new_ham_matrix, vmin=-7, vmax=7, cmap="hot")
axes[2].set_title("Learned", y=1.13)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()


######################################################################
# .. figure:: ../demonstrations/qgrnn/hamiltonian_comparison.png
#      :width: 100%
#      :align: center
#


######################################################################
# These images look very similar, indicating that the QGRNN has done a good job
# learning the target Hamiltonian.
#
# We can also look
# at the exact values of the target and learned parameters.
# Recall how the target
# interaction graph has :math:`4` edges while the complete graph has :math:`6`.
# Thus, as the QGRNN converges to the optimal solution, the weights corresponding to
# edges :math:`(1, 3)` and :math:`(2, 0)` in the complete graph should go to :math:`0`, as
# this indicates that they have no effect, and effectively do not exist in the learned
# Hamiltonian.

# We first pick out the weights of edges (1, 3) and (2, 0)
# and then remove them from the list of target parameters

weights_noedge = []
weights_edge = []
for ii, edge in enumerate(new_ising_graph.edges):
    if (edge[0] - qubit_number, edge[1] - qubit_number) in ising_graph.edges:
        weights_edge.append(weights[ii])
    else:
        weights_noedge.append(weights[ii])

######################################################################
# Then, we print all of the weights:
#

print("Target parameters \tLearned parameters")
print("\t Weights:")
for ii_target, ii_learned in zip(target_weights, weights_edge):
    print(f"{ii_target}\t\t\t{ii_learned}")

print("\t Bias:")
for ii_target, ii_learned in zip(target_bias, bias):
    print(f"{ii_target}\t\t\t{ii_learned}")

print(f"\nNon-Existing Edge Parameters: {weights_noedge}")


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#       Target parameters       Learned parameters
#                Weights:
#       0.56                    0.5782895244479087
#       1.24                    1.3350283296762822
#       1.67                    1.8044804399858207
#       -0.79                   -0.8395497395039521
#                Bias:
#       -1.44                   -1.3529586931946445
#       -1.43                   -1.313891802560241
#       1.18                    0.9811173767093699
#       -0.93                   -1.0579331212003404
#
#       Non-Existing Edge Parameters: [tensor(0.03795849, requires_grad=True), tensor(-0.00229918, requires_grad=True)]
#


######################################################################
# The weights of edges :math:`(1, 3)` and :math:`(2, 0)`
# are very close to :math:`0`, indicating we have learned the cycle graph
# from the complete graph. In addition, the remaining learned weights
# are fairly close to those of the target Hamiltonian.
# Thus, the QGRNN is functioning properly, and has learned the target
# Ising Hamiltonian to a high
# degree of accuracy!
#

######################################################################
# References
# ----------
#
# 1. Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., &
#    Hidary, J. (2019). Quantum Graph Neural Networks. arXiv preprint
#    `arXiv:1909.12264 <https://arxiv.org/abs/1909.12264>`__.
#
