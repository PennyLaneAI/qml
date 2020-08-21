"""
The Quantum Graph Recurrent Neural Network
===========================================

.. meta::
    :property="og:description": Using a quantum graph recurrent neural network to learn quantum dynamics.
    :property="og:image": https://pennylane.ai/qml/_images/qgrnn_thumbnail.png

*Author: Jack Ceroni*

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


matrix_params = [[0.56, 1.24, 1.67, -0.79], [-1.44, -1.43, 1.18, -0.93]]


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


def create_hamiltonian_matrix(n, graph, params):

    matrix = np.zeros((2 ** n, 2 ** n))

    # Creates the interaction component of the Hamiltonian
    for count, i in enumerate(graph.edges):
        m = 1
        for j in range(0, n):
            if i[0] == j or i[1] == j:
                m = np.kron(m, qml.PauliZ.matrix)
            else:
                m = np.kron(m, np.identity(2))
        matrix += params[0][count] * m

    # Creates the bias components of the matrix
    for i in range(0, n):
        m1 = m2 = 1
        for j in range(0, n):
            if j == i:
                m1 = np.kron(m1, qml.PauliZ.matrix)
                m2 = np.kron(m2, qml.PauliX.matrix)
            else:
                m1 = np.kron(m1, np.identity(2))
                m2 = np.kron(m2, np.identity(2))
        matrix += (params[1][i] * m1 + m2)

    return matrix

# Prints a visual representation of the Hamiltonian matrix
ham_matrix = create_hamiltonian_matrix(qubit_number, ising_graph, matrix_params)
plt.matshow(ham_matrix, cmap='hot')
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
    (-0.054661080280306085+0.016713907320174026j), 
    (0.12290003656489545-0.03758500591109822j), 
    (0.3649337966440005-0.11158863596657455j), 
    (-0.8205175732627094+0.25093231967092877j), 
    (0.010369790825776609-0.0031706387262686003j), 
    (-0.02331544978544721+0.007129899300113728j), 
    (-0.06923183949694546+0.0211684344103713j), 
    (0.15566094863283836-0.04760201916285508j), 
    (0.014520590919500158-0.004441887836078486j), 
    (-0.032648113364535575+0.009988590222879195j), 
    (-0.09694382811137187+0.02965579457620536j), 
    (0.21796861485652747-0.06668776658411019j), 
    (-0.0027547112135013247+0.0008426289322652901j), 
    (0.006193695872468649-0.0018948418969390599j), 
    (0.018391279795405405-0.005625722994009138j), 
    (-0.041350974715649635+0.012650711602265649j)
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
#      Energy Expectation: -7.244508985189101
#      Ground State Energy: -7.330689661291242
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

    U = scipy.linalg.expm(-1j* hamiltonian * time)
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


def qgrnn_layer(param1, param2, qubits, graph, trotter_step):

    # Applies a layer of RZZ gates (based on a graph)
    for count, i in enumerate(graph.edges):
        qml.MultiRZ(2 * param1[count] * trotter_step, wires=[i[0], i[1]])

    # Applies a layer of RZ gates
    for count, i in enumerate(qubits):
        qml.RZ(2 * param2[count] * trotter_step, wires=i)

    # Applies a layer of RX gates
    for i in qubits:
        qml.RX(2 * trotter_step, wires=i)


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
    for i in range(0, len(register1)):
        qml.CSWAP(wires=[int(control), register1[i], register2[i]])
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
# interaction graph, which we set to be a complete graph. This choice
# is motivated by the fact that any target interaction graph will be a subgraph
# of this initial guess. Part of the idea behind the QGRNN is that
# we don’t know the interaction graph, and it has to be learned. In this case, the graph
# is learned *automatically* as the target parameters are optimized. The
# :math:`\boldsymbol\mu` parameters that correspond to edges that don't exist in
# the target graph will simply approach :math:`0`.
#

# Defines some fixed values

reg1 = list(range(qubit_number))  # First qubit register
reg2 = list(range(qubit_number, 2 * qubit_number))  # Second qubit register

control = 2 * qubit_number  # Index of control qubit
trotter_step = 0.01  # Trotter step size

# Defines the interaction graph for the new qubit system

new_ising_graph = nx.Graph()
new_ising_graph.add_nodes_from(range(qubit_number, 2 * qubit_number))
new_ising_graph.add_edges_from([(4, 5), (5, 6), (6, 7), (4, 6), (7, 4), (5, 7)])

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


def qgrnn(params1, params2, time=None):

    # Prepares the low energy state in the two registers
    qml.QubitStateVector(np.kron(low_energy_state, low_energy_state), wires=reg1+reg2)

    # Evolves the first qubit register with the time-evolution circuit to
    # prepare a piece of quantum data
    state_evolve(ham_matrix, reg1, time)

    # Applies the QGRNN layers to the second qubit register
    depth = time / trotter_step  # P = t/Delta
    for i in range(0, int(depth)):
        qgrnn_layer(params1, params2, reg2, new_ising_graph, trotter_step)

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


def cost_function(params):

    global iterations

    # Separates the parameter list
    weight_params = params[0:6]
    bias_params = params[6:10]

    # Randomly samples times at which the QGRNN runs
    times_sampled = [np.random.uniform() * max_time for i in range(0, N)]

    # Cycles through each of the sampled times and calculates the cost
    total_cost = 0
    for i in times_sampled:
        result = qnode(weight_params, bias_params, time=i)
        total_cost += -1 * result

    # Prints the value of the cost function
    if iterations % 5 == 0:
        print(
            "Fidelity at Step " + str(iterations) + ": " + str((-1 * total_cost / N)._value)
            )
        print("Parameters at Step " + str(iterations) + ": " + str(params._value))
        print("---------------------------------------------")

    iterations += 1

    return total_cost / N


######################################################################
# The last step is to define the new device and QNode, and execute the
# optimizer. We use Adam,
# with a step-size of :math:`0.5`:
#

# Defines the new device

qgrnn_dev = qml.device("default.qubit", wires=2 * qubit_number + 1)

# Defines the new QNode

qnode = qml.QNode(qgrnn, qgrnn_dev)

iterations = 0
optimizer = qml.AdamOptimizer(stepsize=0.5)
steps = 300
qgrnn_params = list([np.random.randint(-20, 20)/50 for i in range(0, 10)])
init = copy.copy(qgrnn_params)

# Executes the optimization method

for i in range(0, steps):
    qgrnn_params = optimizer.step(cost_function, qgrnn_params)


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
# 
#  .. code-block:: none
#
#      Fidelity at Step 0: 0.9918346272467008
#      Parameters at Step 0: [-0.18, 0.3, 0.32, -0.08, 0.22, 0.28, 0.0, 0.04, -0.26, -0.34]
#      ---------------------------------------------
#      Fidelity at Step 1: 0.9982958958675725
#      Parameters at Step 1: [-0.6799981741676951, 0.7999982222290656, 0.8199988800891312, 0.4199985008189975, 0.7199987506613634, -0.2199988652959371, -0.4999965632247722, -0.4599971332618679, 0.2399970101066845, 0.1599984088859147]
#      ---------------------------------------------
#      Fidelity at Step 2: 0.990490615156777
#      Parameters at Step 2: [-1.0098221172022992, 1.1336805430938313, 1.1430272204750902, 0.7644358118180916, 1.0421203687079919, -0.5468227660845358, -0.8209215857161252, -0.8014895139877191, 0.5869815972734128, 0.47692466959999924]
#      ---------------------------------------------
#      Fidelity at Step 3: 0.9762351051354551
#      Parameters at Step 3: [-1.2299363744628509, 1.3661666433666075, 1.3415230886864924, 1.030456044669323, 1.2378781697230516, -0.7575172669172114, -1.0130201555167162, -1.0581775588121731, 0.8610525437144596, 0.6563901663908569]
#      ---------------------------------------------
#      Fidelity at Step 4: 0.9784260543002337
#      Parameters at Step 4: [-1.3467580918064672, 1.5070467394602267, 1.4191992723432134, 1.2363074986621367, 1.3105574652271725, -0.8578216277740998, -1.0787194096943145, -1.2453915798444228, 1.082957803098034, 0.7007286251069664]
#      ---------------------------------------------
#      Fidelity at Step 5: 0.9760924766644138
#      Parameters at Step 5: [-1.3975050155126143, 1.5874154907714464, 1.4253477829297985, 1.3972088582056272, 1.310768053020758, -0.8907455190190542, -1.0697339005273567, -1.3825354197935542, 1.264473178391723, 0.6691484100078874]
#      ---------------------------------------------
#      Fidelity at Step 6: 0.9791342696605839
#      Parameters at Step 6: [-1.3916054226388823, 1.6158640373620028, 1.3729978909141598, 1.5194931625164307, 1.2515225501166842, -0.8679964251616565, -0.9987099125166444, -1.4761909187915563, 1.4117656894362285, 0.5767242683721397]
#      ---------------------------------------------
#      Fidelity at Step 7: 0.9787421089376458
#      Parameters at Step 7: [-1.3419795434786161, 1.6037009979053718, 1.2793404242059645, 1.6096566051847816, 1.1498502520134959, -0.8049890167819886, -0.8822831819835857, -1.5340642754199918, 1.530248443597091, 0.44241619983839797]
#      ---------------------------------------------
#      Fidelity at Step 8: 0.9820315749997621
#      Parameters at Step 8: [-1.2488997936415513, 1.5514040173101722, 1.1474714836863231, 1.6683325278610117, 1.0085180485340484, -0.7045659446371773, -0.7220052474377402, -1.5561429872624357, 1.6212374403240934, 0.2699385653926958]
#      ---------------------------------------------
#      Fidelity at Step 9: 0.9906449268083709
#      Parameters at Step 9: [-1.1168651248534744, 1.4629872578303456, 0.9855814851930917, 1.6969881487826513, 0.8353653676865135, -0.57428268457325, -0.524589866529066, -1.5440633182793588, 1.68617934363009, 0.06842824811303153]
#      ---------------------------------------------
#      Fidelity at Step 10: 0.99582195327501
#      Parameters at Step 10: [-0.964031092107903, 1.354674684306868, 0.814393606152222, 1.7042986703019174, 0.650940374604191, -0.43328323068490904, -0.31101171334022504, -1.5090252857025748, 1.731326080636684, -0.1402782545374336]
#      ---------------------------------------------
#      Fidelity at Step 11: 0.9961926541501808
#      Parameters at Step 11: [-0.8036684420752858, 1.238468601757792, 0.648795252860385, 1.6970845839900839, 0.46994924470041266, -0.29523086642904206, -0.0966467937581893, -1.4596190993696592, 1.7617532916742076, -0.34043090654836505]
#      ---------------------------------------------
#      Fidelity at Step 12: 0.9983158950218793
#      Parameters at Step 12: [-0.6312660733884548, 1.1116800858652522, 0.4972845049612349, 1.6699268716653712, 0.2975512423191221, -0.16726719667887802, 0.11884361341846977, -1.3889433232896733, 1.773276350359844, -0.5223183946029325]
#      ---------------------------------------------
#      Fidelity at Step 13: 0.9967000138255077
#      Parameters at Step 13: [-0.4676461753735439, 0.9917792527825551, 0.36890033998622973, 1.6368149409927473, 0.1458560667857181, -0.058079424182490846, 0.3155504611747129, -1.3145273809480482, 1.7766599241241958, -0.6769959196369855]
#      ---------------------------------------------
#      Fidelity at Step 14: 0.9945001638616079
#      Parameters at Step 14: [-0.31794007513498357, 0.8838593182597574, 0.274152383590202, 1.5986807415684292, 0.023998424267510157, 0.02334325807987936, 0.484463374084153, -1.2376354781826886, 1.7724156182238344, -0.792919523723297]
#      ---------------------------------------------
#      Fidelity at Step 15: 0.9913345835898904
#      Parameters at Step 15: [-0.18832820993560723, 0.7934498472478128, 0.2192041202380814, 1.5582912583375383, -0.06231146312915761, 0.07191800841202414, 0.6173058799196036, -1.1617573784335757, 1.7624859746860937, -0.8642178255228796]
#      ---------------------------------------------
#      Fidelity at Step 16: 0.9889565034466684
#      Parameters at Step 16: [-0.08560961522854411, 0.7266587755855446, 0.21178056237429993, 1.5174343908488523, -0.10602854111569737, 0.08120012393307255, 0.7038563895141848, -1.089491099254555, 1.7477885167776666, -0.8832941887432142]
#      ---------------------------------------------
#      Fidelity at Step 17: 0.9925729160049567
#      Parameters at Step 17: [-0.01530712818391683, 0.688186047241436, 0.2546346524352683, 1.478007577277352, -0.10461799053446165, 0.04908916100233172, 0.7372715330783467, -1.023558091459499, 1.72924671045939, -0.8487271410430316]
#      ---------------------------------------------
#      Fidelity at Step 18: 0.9929504851720798
#      Parameters at Step 18: [0.030032491434869538, 0.6705379228419254, 0.32736077326483304, 1.4416928946464962, -0.07563674255845146, -0.007189452883322685, 0.7339963088570366, -0.965283352682299, 1.7091399150791209, -0.7831498975610103]
#      ---------------------------------------------
#      Fidelity at Step 19: 0.9965405348346449
#      Parameters at Step 19: [0.049351359919739754, 0.6743590703990037, 0.4278756421344065, 1.4090284612866684, -0.02163430171942262, -0.08514460113615971, 0.693373791202934, -0.9156335387084984, 1.6874429857641309, -0.6900461171429496]
#      ---------------------------------------------
#      Fidelity at Step 20: 0.9966665201201759
#      Parameters at Step 20: [0.05278414037474171, 0.690320672688792, 0.5389740191005465, 1.3798222192408045, 0.04246657361378194, -0.1704614872588347, 0.6327107569705728, -0.8732384948075647, 1.665370037015241, -0.5873476637991509]
#      ---------------------------------------------
#      Fidelity at Step 21: 0.9985601378649517
#      Parameters at Step 21: [0.036744860702025185, 0.7211548377193117, 0.6624797948658447, 1.3541418526093136, 0.11662055186990422, -0.2630293857462455, 0.5463270732728669, -0.8391813354875706, 1.641691390380928, -0.47517726904059726]
#      ---------------------------------------------
#      Fidelity at Step 22: 0.9989726690473872
#      Parameters at Step 22: [0.0099853113729, 0.7590819662221863, 0.786559829395425, 1.331306986027622, 0.19078421742491747, -0.3533747180286354, 0.44798226535748187, -0.811516385783778, 1.6171600388645548, -0.3653610979369781]
#      ---------------------------------------------
#      Fidelity at Step 23: 0.999047032083147
#      Parameters at Step 23: [-0.026725503966848224, 0.8028384497425541, 0.9074123764373417, 1.3108986407404344, 0.2601589904988729, -0.43682334045593063, 0.3386032585919626, -0.7903088349898605, 1.5906634543837104, -0.2636408610946702]
#      ---------------------------------------------
#      Fidelity at Step 24: 0.9987705154428619
#      Parameters at Step 24: [-0.07086221861622898, 0.8494889117930519, 1.0190044786807104, 1.292312641664198, 0.31796110307647885, -0.5068103469275773, 0.22176642815390943, -0.7752768448722682, 1.561023616126524, -0.1781655177903615]
#      ---------------------------------------------
#      Fidelity at Step 25: 0.9988100329375428
#      Parameters at Step 25: [-0.11779944704122991, 0.8944605594186567, 1.1143459278348542, 1.2748222812428165, 0.35783783243731665, -0.5572620061886935, 0.10437661028468458, -0.765460064780762, 1.52771962327713, -0.11647520632647335]
#      ---------------------------------------------
#      Fidelity at Step 26: 0.9985757201454402
#      Parameters at Step 26: [-0.1620749955989192, 0.9340651187861451, 1.1928230771213455, 1.2582183861869087, 0.3819403680851732, -0.5908063527356947, -0.0052458861753093555, -0.7587107231668222, 1.4928816938483498, -0.07549378112681823]
#      ---------------------------------------------
#      Fidelity at Step 27: 0.9976302566535219
#      Parameters at Step 27: [-0.20132070244284592, 0.9664369998249582, 1.2536060950940575, 1.2422878678254046, 0.3904514403341251, -0.6078223076765568, -0.10356163561282461, -0.7542507212062988, 1.4569850261955124, -0.05477860988229539]
#      ---------------------------------------------
#      Fidelity at Step 28: 0.9983882356575454
#      Parameters at Step 28: [-0.23255417027769784, 0.987815950434486, 1.2909889004408277, 1.2265674025512432, 0.37752274884372017, -0.6025799670865759, -0.1859803393905809, -0.752257031132375, 1.4186113777017493, -0.06138955414451268]
#      ---------------------------------------------
#      Fidelity at Step 29: 0.9978671373289247
#      Parameters at Step 29: [-0.2566856271475478, 1.0006492046651103, 1.3125780415283235, 1.2115530344170848, 0.3520816340739027, -0.5841604105400039, -0.2540974192141527, -0.7518168751351452, 1.380097816028637, -0.0842201282781858]
#      ---------------------------------------------
#      Fidelity at Step 30: 0.9974929437984331
#      Parameters at Step 30: [-0.27124830423752155, 1.001805037820717, 1.3148505271380608, 1.1972028099405516, 0.3108355440242823, -0.5493333047969007, -0.30386761874823015, -0.7534487198975702, 1.340572547803093, -0.12730767306292412]
#      ---------------------------------------------
#      Fidelity at Step 31: 0.9987904515719516
#      Parameters at Step 31: [-0.27331253281188894, 0.9875366027856964, 1.2947325486214651, 1.183881268184288, 0.2512304445465742, -0.49559813556132276, -0.3302261854169147, -0.7582194250788556, 1.2993209544142166, -0.1938896756224774]
#      ---------------------------------------------
#      Fidelity at Step 32: 0.9990327822738706
#      Parameters at Step 32: [-0.26795379123456736, 0.9646777852539683, 1.2642320526274793, 1.172242080195256, 0.18538030083281698, -0.4350974227693908, -0.3416288874251854, -0.7653120532473738, 1.2589997312286312, -0.26913506648646457]
#      ---------------------------------------------
#      Fidelity at Step 33: 0.9996630784088545
#      Parameters at Step 33: [-0.25498616785829376, 0.9326639083646915, 1.2250032532283779, 1.1631299925830865, 0.11566158561604338, -0.370123583270212, -0.3372023926341056, -0.7759491224912733, 1.219973281394287, -0.35046354300650573]
#      ---------------------------------------------
#      Fidelity at Step 34: 0.9995646137376791
#      Parameters at Step 34: [-0.23931125925929084, 0.8979390115443661, 1.18517083195646, 1.1562392191477264, 0.049407091226932126, -0.3080683093766753, -0.3256263558112948, -0.7885433711096579, 1.1837377130938038, -0.42860701069958995]
#      ---------------------------------------------
#      Fidelity at Step 35: 0.9996465904308229
#      Parameters at Step 35: [-0.21955712185077875, 0.8579684028917567, 1.1461751770792745, 1.153596712136502, -0.010617573336647104, -0.25156592871335065, -0.3032907108221233, -0.806211033906433, 1.150742645372461, -0.5010224120719031]
#      ---------------------------------------------
#      Fidelity at Step 36: 0.9994156035961047
#      Parameters at Step 36: [-0.19868629481528227, 0.8165858084509231, 1.1121003293501985, 1.1548201771770399, -0.06107266557805362, -0.20402041326536352, -0.2755913764320364, -0.827857105634376, 1.1216160185109565, -0.5633578965665793]
#      ---------------------------------------------
#      Fidelity at Step 37: 0.9993165244122285
#      Parameters at Step 37: [-0.1779889897836405, 0.7749703004823995, 1.0870344529373086, 1.1611755563451476, -0.09753314112151815, -0.16984814116498834, -0.24407479905269586, -0.8549288607994256, 1.0972908304783051, -0.6106194470473059]
#      ---------------------------------------------
#      Fidelity at Step 38: 0.9996276075039349
#      Parameters at Step 38: [-0.15915307597784445, 0.7354708885031871, 1.072332703887529, 1.1720124226103679, -0.11964806996345806, -0.14955951002483486, -0.2122772329988582, -0.8862271155143, 1.0778211101348336, -0.6420226972112081]
#      ---------------------------------------------
#      Fidelity at Step 39: 0.9990796004168293
#      Parameters at Step 39: [-0.142525631080849, 0.6994887039221028, 1.0634937491532876, 1.1842620087197029, -0.13411912965193842, -0.13670846657035737, -0.1829062246986572, -0.9176867201800266, 1.0615334788002615, -0.6645525202471699]
#      ---------------------------------------------
#      Fidelity at Step 40: 0.9991081504737204
#      Parameters at Step 40: [-0.12978142843589885, 0.6682722355308486, 1.067173079974738, 1.2008548828199337, -0.13307343789244414, -0.13917897231406814, -0.15733943829829156, -0.9527752044623491, 1.0504893600171643, -0.6695713382045293]
#      ---------------------------------------------
#      Fidelity at Step 41: 0.9994006109153413
#      Parameters at Step 41: [-0.12115567316443504, 0.6424397976328438, 1.0824788700855221, 1.2211349307434558, -0.11825805101753337, -0.15543553765641152, -0.1367627010490742, -0.9905091080053223, 1.0443773039911841, -0.6588951948060155]
#      ---------------------------------------------
#      Fidelity at Step 42: 0.9994774988967781
#      Parameters at Step 42: [-0.11571425014381993, 0.6214062734740367, 1.1051772812625795, 1.2432828086933019, -0.09506646644529715, -0.1802470006258153, -0.12088226665017486, -1.0287060346219037, 1.0418592834612284, -0.6384405334852695]
#      ---------------------------------------------
#      Fidelity at Step 43: 0.9994472365267485
#      Parameters at Step 43: [-0.11343136573993991, 0.6053750054673392, 1.1345586980326912, 1.2670816384279806, -0.0645914205975793, -0.21272564092891239, -0.11021940656600411, -1.0669698557220983, 1.042885163548047, -0.6093818606417603]
#      ---------------------------------------------
#      Fidelity at Step 44: 0.9997444727808593
#      Parameters at Step 44: [-0.11493137902455217, 0.5956040002841007, 1.1720296957398517, 1.2933199126033281, -0.025840937747204952, -0.25441678362223674, -0.10664260949985985, -1.1056396577415977, 1.0485202738504023, -0.5705427079314583]
#      ---------------------------------------------
#      Fidelity at Step 45: 0.99988159413535
#      Parameters at Step 45: [-0.1181641198295314, 0.5899525287207167, 1.2117280803719108, 1.3197972842747712, 0.014517460349924967, -0.29859722278693634, -0.10769621831761414, -1.1424715828989647, 1.0567394027897858, -0.5293778103982675]
#      ---------------------------------------------
#      Fidelity at Step 46: 0.9998573514345785
#      Parameters at Step 46: [-0.12175656133105636, 0.5867654275419255, 1.2499010203636594, 1.3451356909319643, 0.05248875951969164, -0.3410987405398154, -0.11141149487295963, -1.176254626328141, 1.0661176290234677, -0.4903933522447272]
#      ---------------------------------------------
#      Fidelity at Step 47: 0.9998292007097693
#      Parameters at Step 47: [-0.12487131421130249, 0.5859454109604079, 1.284613548937081, 1.3689706154338066, 0.0852664799312279, -0.379774628371075, -0.11801000957728758, -1.2060029474462826, 1.0767548088559875, -0.45664961742963583]
#      ---------------------------------------------
#      Fidelity at Step 48: 0.9997365697977294
#      Parameters at Step 48: [-0.12654857023039606, 0.586429393841211, 1.313585556214431, 1.3905244254034577, 0.11041058221068717, -0.4121675022983979, -0.12627431817292728, -1.230996950201297, 1.087855998062673, -0.43089443071099853]
#      ---------------------------------------------
#      Fidelity at Step 49: 0.9997528113932673
#      Parameters at Step 49: [-0.12584831617269926, 0.5874242095681329, 1.3349964372480259, 1.4091922470968539, 0.12563466677970891, -0.43619264265287366, -0.13540569126088883, -1.2504754631024455, 1.0988857079981462, -0.41566649516961174]
#      ---------------------------------------------
#      Fidelity at Step 50: 0.9995771492573322
#      Parameters at Step 50: [-0.12308220081986324, 0.5882956053892081, 1.3498112566839737, 1.4250593957050575, 0.1327111946665708, -0.45293666365919977, -0.14442696202432756, -1.2653186413203268, 1.109248000229557, -0.4091049260651626]
#      ---------------------------------------------
#      Fidelity at Step 51: 0.9996582070356129
#      Parameters at Step 51: [-0.1167464756855651, 0.5885180846568769, 1.3553538821219644, 1.437311180522153, 0.12773708284845237, -0.45934041737317816, -0.153191885507882, -1.2738828621497664, 1.1187160226528092, -0.41542403253932575]
#      ---------------------------------------------
#      Fidelity at Step 52: 0.9996955432159077
#      Parameters at Step 52: [-0.10771034556633162, 0.5880448359118233, 1.3537658157861867, 1.4464530329091365, 0.11384863623479319, -0.45786035704962497, -0.16146420142746504, -1.2774690460612694, 1.1271196329598958, -0.43122725720747185]
#      ---------------------------------------------
#      Fidelity at Step 53: 0.9997991619034833
#      Parameters at Step 53: [-0.09627329672367939, 0.5868963942580755, 1.3462380053806309, 1.4527280567240575, 0.09247903186228626, -0.44976296196126986, -0.1693346453161187, -1.2765780952688226, 1.1344036427582234, -0.45493168233995446]
#      ---------------------------------------------
#      Fidelity at Step 54: 0.9996928121478973
#      Parameters at Step 54: [-0.08345955687710035, 0.5853498150099569, 1.3350713003972525, 1.456813442186792, 0.06682602891064508, -0.43769031387004786, -0.1769697735889833, -1.2725521174090646, 1.1406837825893554, -0.48306702794086753]
#      ---------------------------------------------
#      Fidelity at Step 55: 0.9998899365649595
#      Parameters at Step 55: [-0.06751148109701582, 0.583214665299736, 1.3183722482806561, 1.4575330258805324, 0.033384228036375245, -0.41943921471437035, -0.18540640162811964, -1.2629340228252863, 1.1456369135708773, -0.5191713211994088]
#      ---------------------------------------------
#      Fidelity at Step 56: 0.9998624340125136
#      Parameters at Step 56: [-0.05131752850714959, 0.5811990289809753, 1.3013023835929207, 1.4569478769178683, -0.0004006580174740204, -0.4007625392880318, -0.19430310803813716, -1.251647479542463, 1.1498364761138966, -0.5553743416586863]
#      ---------------------------------------------
#      Fidelity at Step 57: 0.9999132721039935
#      Parameters at Step 57: [-0.034655781202251414, 0.5797721835560955, 1.284992930419999, 1.4547640992807522, -0.03367428060470313, -0.3831140718715586, -0.20534639523532677, -1.2379072945156397, 1.1532505168291682, -0.5904570588092755]
#      ---------------------------------------------
#      Fidelity at Step 58: 0.9998775148230059
#      Parameters at Step 58: [-0.018859726755114874, 0.579075691631346, 1.271532927647943, 1.4521537374570472, -0.06335771928892965, -0.36867465578447683, -0.21781255421386045, -1.2238816583685157, 1.1562215101775397, -0.6213031578896728]
#      ---------------------------------------------
#      Fidelity at Step 59: 0.9998457347810404
#      Parameters at Step 59: [-0.0043922861721972534, 0.5793702002250531, 1.2627489076560674, 1.4496008479846716, -0.087487791151152, -0.3593321319413558, -0.2321843903928341, -1.2102636747738713, 1.1588836805134917, -0.6457585646371806]
#      ---------------------------------------------
#      Fidelity at Step 60: 0.9997727108398051
#      Parameters at Step 60: [0.008381004961572968, 0.5806993140477965, 1.2595896924677998, 1.447526804103137, -0.10478843190020545, -0.35617567351770024, -0.2484759318134681, -1.1977686151876175, 1.1613353047857047, -0.6624979557651268]
#      ---------------------------------------------
#      Fidelity at Step 61: 0.999780827709714
#      Parameters at Step 61: [0.01909797144686237, 0.5832292737274423, 1.2640247275721554, 1.4463805516185642, -0.1129425547030803, -0.36150916550403545, -0.267404289385826, -1.187080581646848, 1.1635990684701452, -0.6689774147180123]
#      ---------------------------------------------
#      Fidelity at Step 62: 0.999808168158736
#      Parameters at Step 62: [0.027905032228909082, 0.5864669710827904, 1.2750805933139595, 1.4463033399265706, -0.11250744283286447, -0.37457023406410855, -0.2881605783982173, -1.1786398687238406, 1.1656020826645257, -0.666008871067185]
#      ---------------------------------------------
#      Fidelity at Step 63: 0.9998548091256995
#      Parameters at Step 63: [0.035299063031327474, 0.5897168777071272, 1.2912297644365056, 1.4472746060795163, -0.10510414420013081, -0.39368682521839066, -0.3096536254084067, -1.1725501854957112, 1.1671566218333023, -0.6555791753225414]
#      ---------------------------------------------
#      Fidelity at Step 64: 0.9999037069964792
#      Parameters at Step 64: [0.0418815267714097, 0.5922150299101131, 1.3102058029478096, 1.449115390383419, -0.09285188170899843, -0.416680689996342, -0.33075519781109286, -1.168752259973914, 1.168045873571082, -0.640236123403879]
#      ---------------------------------------------
#      Fidelity at Step 65: 0.9999246406433501
#      Parameters at Step 65: [0.04829108705674348, 0.5933961363308191, 1.3295024801701152, 1.451494818076525, -0.07868535063518896, -0.44066890713809964, -0.3503195623475613, -1.1667777631995717, 1.1681448292091399, -0.6233418461259689]
#      ---------------------------------------------
#      Fidelity at Step 66: 0.9998866529728345
#      Parameters at Step 66: [0.054990646400591335, 0.5929714866395723, 1.3475420631521589, 1.4541780082209224, -0.06478406516638432, -0.4635982065859475, -0.367668914093558, -1.1661843607147455, 1.1674078817134135, -0.6073574766751283]
#      ---------------------------------------------
#      Fidelity at Step 67: 0.9999098513016205
#      Parameters at Step 67: [0.06305970317875366, 0.5894425173204019, 1.362279806065894, 1.457201816175589, -0.05293376636334743, -0.4837715235887784, -0.38173243002587237, -1.1674220982195875, 1.1651062924666915, -0.594733597363039]
#      ---------------------------------------------
#      Fidelity at Step 68: 0.9998660737935083
#      Parameters at Step 68: [0.07207131811893552, 0.5836763740576104, 1.3731394847850424, 1.4602536559593045, -0.04417940302564178, -0.500178959137041, -0.39288037868101516, -1.169729608023709, 1.161707190111347, -0.5863686970050932]
#      ---------------------------------------------
#      Fidelity at Step 69: 0.9999129411159435
#      Parameters at Step 69: [0.08262588608079113, 0.574810479604329, 1.3780673500067586, 1.4633174442657115, -0.04043810439157602, -0.5109593138014746, -0.40059235668512994, -1.1733561658529568, 1.156749837842348, -0.5847647387158366]
#      ---------------------------------------------
#      Fidelity at Step 70: 0.9999204175249918
#      Parameters at Step 70: [0.09353640799320935, 0.5647304323063623, 1.3794737376476547, 1.4663251015063614, -0.03994340760723002, -0.5179451077951711, -0.40634844089445255, -1.1775732993830843, 1.151165123449546, -0.5873654574983501]
#      ---------------------------------------------
#      Fidelity at Step 71: 0.9999188294218071
#      Parameters at Step 71: [0.10439909428941072, 0.5539946810211575, 1.3778815045617863, 1.4692807280461864, -0.04216929238300462, -0.521672517131992, -0.4107648593458895, -1.1822576620042096, 1.1451991959499215, -0.5934612963667989]
#      ---------------------------------------------
#      Fidelity at Step 72: 0.9998604104737938
#      Parameters at Step 72: [0.1150711419134814, 0.5428094649719107, 1.3736920817157159, 1.4723117761807147, -0.047000405497170064, -0.5223555994970089, -0.414287638179804, -1.187491318807621, 1.1388891215693562, -0.6029561557441782]
#      ---------------------------------------------
#      Fidelity at Step 73: 0.9999430308973932
#      Parameters at Step 73: [0.1256137433970921, 0.5304311299324498, 1.364379429769975, 1.4758445198215686, -0.05652412917964197, -0.5178998350909616, -0.4174842021241018, -1.1944011422191803, 1.131549276053809, -0.618992065967143]
#      ---------------------------------------------
#      Fidelity at Step 74: 0.9999164917980096
#      Parameters at Step 74: [0.1352240671048545, 0.5186721462999183, 1.3544791514439096, 1.4795190877049202, -0.06662586038443774, -0.5124582922772923, -0.42092281691525507, -1.2016505027993059, 1.124376089573319, -0.6358352457562463]
#      ---------------------------------------------
#      Fidelity at Step 75: 0.9999447368621411
#      Parameters at Step 75: [0.1433517389997682, 0.5077377878377025, 1.3439833619291048, 1.4839036749660373, -0.07720888813729462, -0.5061759141332505, -0.4260363440256178, -1.2102020610623234, 1.1171661798181443, -0.6538662573020569]
#      ---------------------------------------------
#      Fidelity at Step 76: 0.9999371167409344
#      Parameters at Step 76: [0.1500140190045107, 0.4980752316911107, 1.3345300738606132, 1.4887111518450673, -0.08677048568017348, -0.5005964210616465, -0.43256554322458635, -1.219315335276509, 1.1103510852672205, -0.6708789782047512]
#      ---------------------------------------------
#      Fidelity at Step 77: 0.9999422443337805
#      Parameters at Step 77: [0.1548386249560046, 0.4900488617967078, 1.3268507273328314, 1.49410275596857, -0.09430363221716345, -0.4967256792177868, -0.4413051029923251, -1.2291959841357525, 1.1040339321643222, -0.6857726663067597]
#      ---------------------------------------------
#      Fidelity at Step 78: 0.9999238559846356
#      Parameters at Step 78: [0.15816023191744086, 0.4835880087905452, 1.3215248542660816, 1.499899606624128, -0.09948259802800598, -0.4949866314201788, -0.45179307232821825, -1.239408282708757, 1.0983243101263098, -0.6979516019681357]
#      ---------------------------------------------
#      Fidelity at Step 79: 0.9998947527867448
#      Parameters at Step 79: [0.1598049053610818, 0.47891485857613925, 1.3193948259446773, 1.5062519491417448, -0.10136287159746196, -0.49640463581192196, -0.4647400306369159, -1.2500852220050942, 1.0932619266576435, -0.7063956771055033]
#      ---------------------------------------------
#      Fidelity at Step 80: 0.9999239374484553
#      Parameters at Step 80: [0.15968640176034107, 0.47623857742848363, 1.3215516035038206, 1.5133445108042078, -0.09869941981715959, -0.5023790625052149, -0.4810894704739998, -1.2613481265326896, 1.0888831966015242, -0.709782922818542]
#      ---------------------------------------------
#      Fidelity at Step 81: 0.999948470725103
#      Parameters at Step 81: [0.15903121689591235, 0.47455604185793715, 1.3263353473424122, 1.5205295668382561, -0.09350921353968285, -0.5109460782478329, -0.4986121270465538, -1.2722717718240208, 1.085019857104728, -0.7100977773081497]
#      ---------------------------------------------
#      Fidelity at Step 82: 0.9999366087435407
#      Parameters at Step 82: [0.15850219731430015, 0.4732269015028527, 1.3323677097014557, 1.5273815258143946, -0.08721313304564966, -0.5206595953771328, -0.5159691422708877, -1.2823461352567127, 1.08150103237917, -0.7087956210664609]
#      ---------------------------------------------
#      Fidelity at Step 83: 0.9999578753805831
#      Parameters at Step 83: [0.15876122870928364, 0.4717539889208967, 1.3389858240280035, 1.5336558651468708, -0.07983759313470387, -0.531645417340478, -0.5331916992279223, -1.2912136020041343, 1.078157462787459, -0.7059802614905673]
#      ---------------------------------------------
#      Fidelity at Step 84: 0.9999562859848723
#      Parameters at Step 84: [0.15998950538828352, 0.46987222594640976, 1.3453745056781694, 1.5392301575813734, -0.0727606140942332, -0.5424516612950055, -0.5491990471098366, -1.2987740326465576, 1.0749304304852945, -0.7031076480354954]
#      ---------------------------------------------
#      Fidelity at Step 85: 0.9999228784481042
#      Parameters at Step 85: [0.16259932393819493, 0.46723784978087973, 1.3510020439148975, 1.5439781690404197, -0.06661231505507895, -0.5525379692017052, -0.5636186962362523, -1.3048278506884046, 1.0717015957679998, -0.7009024320958638]
#      ---------------------------------------------
#      Fidelity at Step 86: 0.9999267299255452
#      Parameters at Step 86: [0.16836953998618337, 0.4625743331186897, 1.354370824497011, 1.547342987670039, -0.06270296629337258, -0.5610776835106112, -0.5757496874720254, -1.3083583567906822, 1.068061298085744, -0.7009301098350672]
#      ---------------------------------------------
#      Fidelity at Step 87: 0.9999539062194203
#      Parameters at Step 87: [0.17706182486991237, 0.4559909932922185, 1.3550043012605686, 1.5493683852536209, -0.06147149350529153, -0.567534538603623, -0.5855838960013352, -1.3095163493047912, 1.0640384884320284, -0.7037221397941468]
#      ---------------------------------------------
#      Fidelity at Step 88: 0.9999375340731136
#      Parameters at Step 88: [0.1868411682003187, 0.4487769452876402, 1.354198171678452, 1.5506991362680713, -0.06179202566244289, -0.572554209006431, -0.5939668641662594, -1.3094725665544014, 1.0600401284357561, -0.708003630693759]
#      ---------------------------------------------
#      Fidelity at Step 89: 0.9999452209763122
#      Parameters at Step 89: [0.19823455064505652, 0.44058748194066133, 1.3509939304430245, 1.5511797992868592, -0.06428899198213606, -0.5757041510507358, -0.601117863559753, -1.3079073427777828, 1.0559608562613465, -0.7146251024356468]
#      ---------------------------------------------
#      Fidelity at Step 90: 0.9999533531894214
#      Parameters at Step 90: [0.21043439829521546, 0.4320595524880997, 1.3463316808943995, 1.5512027849227317, -0.06823345854195782, -0.5776080355311207, -0.6077936351302407, -1.305379521016447, 1.0519900966765465, -0.7228294616693073]
#      ---------------------------------------------
#      Fidelity at Step 91: 0.9999482377985015
#      Parameters at Step 91: [0.22258131884660334, 0.42383575872142953, 1.3410192677440258, 1.5511322451252496, -0.07274782428256824, -0.5789636125943975, -0.614715204002582, -1.302483776213029, 1.0483402298473565, -0.7316818371041457]
#      ---------------------------------------------
#      Fidelity at Step 92: 0.9999576342273339
#      Parameters at Step 92: [0.23412628512023217, 0.41639544583068705, 1.3358218745171693, 1.5513108208713804, -0.07713588301552442, -0.5804647359056426, -0.6227699371318457, -1.29963212390933, 1.0451605613454913, -0.7405203785739566]
#      ---------------------------------------------
#      Fidelity at Step 93: 0.9999727698188148
#      Parameters at Step 93: [0.24453212095330634, 0.4100441400134776, 1.3315228065798237, 1.5519269263778077, -0.08074493839523109, -0.5825878464036203, -0.6319817828330716, -1.2972190092792801, 1.0425036708669353, -0.7485460033992009]
#      ---------------------------------------------
#      Fidelity at Step 94: 0.9999536250599137
#      Parameters at Step 94: [0.2536722703488573, 0.4047163590726245, 1.3280973887080771, 1.5528955787872132, -0.08347035077885467, -0.5852439420251464, -0.6417777542876416, -1.2953414143018134, 1.0403125102601947, -0.755512034118745]
#      ---------------------------------------------
#      Fidelity at Step 95: 0.9999374137250309
#      Parameters at Step 95: [0.26103507537725984, 0.4009139089583318, 1.3261420351258584, 1.5546937868438435, -0.08446635447884701, -0.5893556788424157, -0.65363447984888, -1.2945073876821842, 1.0388087430656394, -0.7607797218924264]
#      ---------------------------------------------
#      Fidelity at Step 96: 0.9999374430735666
#      Parameters at Step 96: [0.26627778353324005, 0.3989196213921769, 1.3262941049535335, 1.5577437040216018, -0.0830341263130707, -0.5956816237748449, -0.668542648219778, -1.2951902905794082, 1.0381178975629617, -0.7638221259352528]
#      ---------------------------------------------
#      Fidelity at Step 97: 0.9999649962536792
#      Parameters at Step 97: [0.2697847785808088, 0.39826938809292944, 1.3281747322189905, 1.561947624046751, -0.07949687637201738, -0.6038947845882106, -0.6859370315702844, -1.2973315413297546, 1.038061526677841, -0.7650278039241306]
#      ---------------------------------------------
#      Fidelity at Step 98: 0.9999680100655207
#      Parameters at Step 98: [0.27271796356922473, 0.39790234793181534, 1.330559862331582, 1.566421385013026, -0.07551557234134834, -0.6122839897830252, -0.7031685695024871, -1.2999467407306893, 1.0381884951626206, -0.7657447634063109]
#      ---------------------------------------------
#      Fidelity at Step 99: 0.9999595967720151
#      Parameters at Step 99: [0.2754978092770754, 0.39740785939847073, 1.3329187656288328, 1.5709769654558308, -0.07156215758556388, -0.6203943327686694, -0.719667344436063, -1.3028301018713266, 1.038348575369489, -0.7664758729296259]
#      ---------------------------------------------
#      Fidelity at Step 100: 0.9999691823205074
#      Parameters at Step 100: [0.2787195506309466, 0.39613177135110544, 1.3345447334346552, 1.5756188983879296, -0.06806192154371864, -0.627881145467017, -0.7351851429420309, -1.3059974283184748, 1.0383453956803301, -0.7679358743646278]
#      ---------------------------------------------
#      Fidelity at Step 101: 0.9999619187430177
#      Parameters at Step 101: [0.28242894636354865, 0.3941413617317143, 1.335479267494555, 1.5800706763364583, -0.06526765767770006, -0.6344739933622472, -0.7492072726913137, -1.3091003914506294, 1.038150294608112, -0.7701324583381439]
#      ---------------------------------------------
#      Fidelity at Step 102: 0.9999527247255329
#      Parameters at Step 102: [0.2871335910640331, 0.39093393639234225, 1.3353433022708152, 1.584278472971393, -0.06355716737124557, -0.6398872952154574, -0.7614264174470093, -1.3120450306286773, 1.0375848139552188, -0.7736042137182007]
#      ---------------------------------------------
#      Fidelity at Step 103: 0.9999515087413056
#      Parameters at Step 103: [0.2933050376932996, 0.3860404487541233, 1.3337755584108504, 1.5881898954764302, -0.06324796246744374, -0.6438856506025266, -0.7716056939368332, -1.3147448766603498, 1.0364772154865878, -0.7788250794420518]
#      ---------------------------------------------
#      Fidelity at Step 104: 0.9999675825710751
#      Parameters at Step 104: [0.3008025878139133, 0.37959472310447573, 1.3309036751336876, 1.5917705693945272, -0.06415186981343302, -0.6466373568076824, -0.7799700701153659, -1.317171006296594, 1.0348254855853776, -0.7855574595540972]
#      ---------------------------------------------
#      Fidelity at Step 105: 0.999963746275596
#      Parameters at Step 105: [0.3088450633120314, 0.3724504767796725, 1.3275806019833747, 1.5951023660493973, -0.06558320660968625, -0.6487570170855949, -0.7872103416969299, -1.3194046097416412, 1.032895159344219, -0.7928421744279232]
#      ---------------------------------------------
#      Fidelity at Step 106: 0.9999700820140445
#      Parameters at Step 106: [0.3171910993892471, 0.3648639835104649, 1.3243046328861554, 1.5982874179327842, -0.0671603503119517, -0.6506556187610064, -0.7938392326798476, -1.3215284432992322, 1.0307296748903971, -0.8002770965620059]
#      ---------------------------------------------
#      Fidelity at Step 107: 0.9999719230711291
#      Parameters at Step 107: [0.32535216077061707, 0.3573429169358276, 1.3214957744448017, 1.6013913477063242, -0.06846711483943341, -0.6527000200317139, -0.8003590156529284, -1.3236232324916415, 1.028486597834076, -0.8073183277096091]
#      ---------------------------------------------
#      Fidelity at Step 108: 0.9999704259176454
#      Parameters at Step 108: [0.3329986920706661, 0.3501863547749545, 1.3192680004739101, 1.6044694211790305, -0.06917601723569679, -0.6551678922869517, -0.8072809087079148, -1.3257835416202473, 1.0262475057816136, -0.813620320598846]
#      ---------------------------------------------
#      Fidelity at Step 109: 0.9999707168742463
#      Parameters at Step 109: [0.33998041479735225, 0.34364365212175774, 1.3181363309815315, 1.6076105586239462, -0.06917365171601826, -0.658241855114755, -0.8147859085270887, -1.3280243362252648, 1.0241020915541021, -0.8190015935969763]
#      ---------------------------------------------
#      Fidelity at Step 110: 0.9999620589656709
#      Parameters at Step 110: [0.3460423583913361, 0.3378693854411911, 1.317849107573066, 1.6108044671696549, -0.06823790133041444, -0.662061454074891, -0.8233949240935231, -1.3304100784557746, 1.0220571285441684, -0.8232926159057777]
#      ---------------------------------------------
#      Fidelity at Step 111: 0.9999584134890723
#      Parameters at Step 111: [0.35107705015323165, 0.3329995731612195, 1.318762522003613, 1.6141968846358399, -0.06612681776667563, -0.6669555529780533, -0.8338466286064476, -1.3330303389384777, 1.020113758621977, -0.8263798910593158]
#      ---------------------------------------------
#      Fidelity at Step 112: 0.9999738155717663
#      Parameters at Step 112: [0.3551015929094727, 0.3289772288315669, 1.320341831355963, 1.6177230807833847, -0.06299787239164033, -0.6727464025727425, -0.8464442930459265, -1.3358252556034642, 1.0182343928369553, -0.8285450055481617]
#      ---------------------------------------------
#      Fidelity at Step 113: 0.9999756646761303
#      Parameters at Step 113: [0.35874167432563875, 0.3253652359278932, 1.3222239301261025, 1.6211422839553296, -0.05984117723587664, -0.6784929360424193, -0.8592066116829397, -1.338460774651548, 1.0164419406180722, -0.8305134681878302]
#      ---------------------------------------------
#      Fidelity at Step 114: 0.9999632639053777
#      Parameters at Step 114: [0.36224136399520773, 0.3219381474441058, 1.3240473921763805, 1.6243697990828445, -0.05697943055729214, -0.6838898111277703, -0.8717711089033766, -1.3408279751056573, 1.0146925419723878, -0.8326343910256307]
#      ---------------------------------------------
#      Fidelity at Step 115: 0.9999668306911149
#      Parameters at Step 115: [0.3661666312383166, 0.31816758562783154, 1.3251122730719682, 1.6273587352316352, -0.054927480071705025, -0.6885799157666435, -0.8845328813242739, -1.3427311562359716, 1.012805347737836, -0.8358068058694881]
#      ---------------------------------------------
#      Fidelity at Step 116: 0.9999754262812004
#      Parameters at Step 116: [0.37067123661742024, 0.3139181266791284, 1.3252926209140343, 1.63006555267173, -0.05381331395093721, -0.6924085884467933, -0.8968698569941312, -1.3441423791478109, 1.0108210366991226, -0.8400656201548946]
#      ---------------------------------------------
#      Fidelity at Step 117: 0.9999695765369236
#      Parameters at Step 117: [0.37551193859655563, 0.30942234244863465, 1.3250715263661343, 1.632541596793348, -0.053322135969472426, -0.6956156209933725, -0.9083586665283568, -1.3451943259101402, 1.008852882232944, -0.8448540224550405]
#      ---------------------------------------------
#      Fidelity at Step 118: 0.9999704499616886
#      Parameters at Step 118: [0.3809867404554024, 0.30431566381658504, 1.3237609869734193, 1.6347345280858476, -0.05357966866203419, -0.6980657635836122, -0.9190687419076997, -1.3458179041827725, 1.0068498778842818, -0.8504909559059793]
#      ---------------------------------------------
#      Fidelity at Step 119: 0.9999755406629498
#      Parameters at Step 119: [0.38704494386526944, 0.2987017809134776, 1.322100112216846, 1.6368192018235117, -0.054239309681938824, -0.7001794349389332, -0.9290719847797951, -1.3461376589728842, 1.0048849088674854, -0.8565799861007863]
#      ---------------------------------------------
#      Fidelity at Step 120: 0.9999789011267138
#      Parameters at Step 120: [0.39331096352645406, 0.29294291177961673, 1.320681871596173, 1.6389160289170315, -0.054877407594790976, -0.7023326927126207, -0.938493381656362, -1.3463437842275554, 1.0030652486348512, -0.8625479083263631]
#      ---------------------------------------------
#      Fidelity at Step 121: 0.9999805974747978
#      Parameters at Step 121: [0.3994724184276103, 0.28723095564053364, 1.3192844618119195, 1.6410257256243213, -0.055214914824290494, -0.7046612022543123, -0.9475386902958111, -1.3465682713464002, 1.001459711273059, -0.868078671789324]
#      ---------------------------------------------
#      Fidelity at Step 122: 0.9999776972201694
#      Parameters at Step 122: [0.4053697020523372, 0.2817518952184357, 1.3183268871909553, 1.643233147188186, -0.05510610404915501, -0.7073342903646016, -0.9563234811807184, -1.3468978140778394, 1.0001022397220145, -0.872971845704736]
#      ---------------------------------------------
#      Fidelity at Step 123: 0.9999730869531723
#      Parameters at Step 123: [0.41099360416098013, 0.27661557993975416, 1.3183903011479048, 1.6456545435677146, -0.05453484459865441, -0.710475460425631, -0.9649009871180091, -1.3473801603951079, 0.9990075498368671, -0.8771861687839271]
#      ---------------------------------------------
#      Fidelity at Step 124: 0.9999757043274963
#      Parameters at Step 124: [0.4162510926470969, 0.2717324710945155, 1.3191867431897601, 1.6484648379221867, -0.053046171510678926, -0.7144703154559339, -0.9739679574720589, -1.3482207595415412, 0.9982798148899359, -0.8805092467341217]
#      ---------------------------------------------
#      Fidelity at Step 125: 0.9999831071267143
#      Parameters at Step 125: [0.4212157294059932, 0.2671278073829876, 1.3208485576256803, 1.6515742056714695, -0.051072873534062604, -0.7189571892077739, -0.9831871286985219, -1.3492992985000976, 0.9978150247357753, -0.8832925936026619]
#      ---------------------------------------------
#      Fidelity at Step 126: 0.9999800502741776
#      Parameters at Step 126: [0.4258304803154631, 0.2628015348419753, 1.322576103509064, 1.6547282217328532, -0.04901727259077763, -0.7234002955708773, -0.9922258325738191, -1.3504893545077803, 0.9975241476585869, -0.8858653921420795]
#      ---------------------------------------------
#      Fidelity at Step 127: 0.999984253640506
#      Parameters at Step 127: [0.43014442630061056, 0.25868690873059363, 1.3240528613827867, 1.657857631389491, -0.047087685568584815, -0.7275751147813105, -1.00102221378504, -1.3517377997881197, 0.9973592845901739, -0.8884676881460879]
#      ---------------------------------------------
#      Fidelity at Step 128: 0.9999815621781017
#      Parameters at Step 128: [0.4342474947003735, 0.25468250766471245, 1.3250608082262136, 1.6609254323711848, -0.0454602413512257, -0.7313106186446107, -1.0095491550284497, -1.353004449699596, 0.9972786993423773, -0.8913329456660455]
#      ---------------------------------------------
#      Fidelity at Step 129: 0.9999749203279624
#      Parameters at Step 129: [0.43828868033187524, 0.2507339797713747, 1.3259657360952608, 1.6639692251407665, -0.04423732639627555, -0.7346198312241684, -1.017741785522529, -1.3542354623451425, 0.997236371907943, -0.8945655022370658]
#      ---------------------------------------------
#      Fidelity at Step 130: 0.9999834155837806
#      Parameters at Step 130: [0.44249513928258005, 0.24653040306162236, 1.3262773111368178, 1.6670270506989011, -0.04370634050950086, -0.7372511491001004, -1.0258454978354004, -1.3554121508254215, 0.9971815425965311, -0.8987263635407147]
#      ---------------------------------------------
#      Fidelity at Step 131: 0.9999843047093948
#      Parameters at Step 131: [0.44662627031508617, 0.2423288408400945, 1.3261078794234142, 1.6699493390380813, -0.04356251616217691, -0.7393949056552869, -1.0336202982832152, -1.3565045133903642, 0.9971185609488726, -0.9032365078623498]
#      ---------------------------------------------
#      Fidelity at Step 132: 0.9999805708115902
#      Parameters at Step 132: [0.4506341367367542, 0.23819936793492752, 1.3256677578018876, 1.6727233225737634, -0.043638344726336875, -0.7412146163040286, -1.0410454472963724, -1.3575100642862223, 0.9970466843308186, -0.9078502956878469]
#      ---------------------------------------------
#      Fidelity at Step 133: 0.9999848139657832
#      Parameters at Step 133: [0.4545912429808756, 0.23404123760127696, 1.324996072175742, 1.6753963110904075, -0.043834865568968186, -0.7428366929831622, -1.048332836540861, -1.3584348303868834, 0.9969536660917491, -0.9125615675357505]
#      ---------------------------------------------
#      Fidelity at Step 134: 0.9999825718980218
#      Parameters at Step 134: [0.4584279371592802, 0.2300414525837428, 1.3247252928222295, 1.677991163691852, -0.043977558780503384, -0.7444909542643103, -1.0553028345072406, -1.3592844893711566, 0.9968544812772046, -0.9170150826462645]
#      ---------------------------------------------
#      Fidelity at Step 135: 0.9999838951828618
#      Parameters at Step 135: [0.4621382229469667, 0.22606720373351205, 1.3244691931108554, 1.6804903156912678, -0.04388265097535359, -0.7462825168234908, -1.062277085779582, -1.360073745565299, 0.9967386980878402, -0.9211551550856614]
#      ---------------------------------------------
#      Fidelity at Step 136: 0.9999824784588653
#      Parameters at Step 136: [0.46573094404052606, 0.22218648245433192, 1.3246036871228646, 1.6829350170970572, -0.043495549937033674, -0.7483250061268732, -1.0691916011324232, -1.3608082940377562, 0.9966148553885258, -0.924864944920965]
#      ---------------------------------------------
#      Fidelity at Step 137: 0.9999844204058552
#      Parameters at Step 137: [0.4691665762480434, 0.21829418758173436, 1.3246903022951173, 1.6852722136598322, -0.04263885474266841, -0.7506944700595976, -1.0763112061299038, -1.3614873941276366, 0.9964682866850386, -0.928060163579988]
#      ---------------------------------------------
#      Fidelity at Step 138: 0.999981074378395
#      Parameters at Step 138: [0.4724835716645587, 0.2145049377725422, 1.3251259128289934, 1.6875346505803719, -0.041523943498307715, -0.7532784602034222, -1.0833319831342216, -1.3620989339182574, 0.9963078754896275, -0.9308430754925966]
#      ---------------------------------------------
#      Fidelity at Step 139: 0.9999795341590785
#      Parameters at Step 139: [0.47570079563668827, 0.2106587381633647, 1.3254196683382355, 1.6896652488659856, -0.04007702689921555, -0.7560711026919471, -1.0905213768357342, -1.362611511128764, 0.9961036723792209, -0.9332696152915033]
#      ---------------------------------------------
#      Fidelity at Step 140: 0.9999889992016171
#      Parameters at Step 140: [0.47898375000729126, 0.2067913373020297, 1.3262842079716255, 1.6917932283805504, -0.03851562036395674, -0.7590541683418879, -1.097685648625456, -1.3630140081676458, 0.9958597307579654, -0.935521013127338]
#      ---------------------------------------------
#      Fidelity at Step 141: 0.999987152994515
#      Parameters at Step 141: [0.4820979864855765, 0.2030562123457122, 1.3268750998011958, 1.6937422986913524, -0.03705166232259487, -0.7618206345540147, -1.1045286788348445, -1.3633311333858664, 0.9956043895794627, -0.9377007330127332]
#      ---------------------------------------------
#      Fidelity at Step 142: 0.9999837616045216
#      Parameters at Step 142: [0.4852152795745083, 0.19933276937130542, 1.3274710255573734, 1.6956023299086598, -0.03577461253637882, -0.7643978601280781, -1.1111527147563243, -1.363538821944383, 0.9953193029481325, -0.9399955381588703]
#      ---------------------------------------------
#      Fidelity at Step 143: 0.9999814459257635
#      Parameters at Step 143: [0.48839916484759044, 0.19550491221578772, 1.3277946418636415, 1.6973704451352887, -0.034798868893791066, -0.7666648753845143, -1.1176717543209973, -1.3636173732484602, 0.9949934317787731, -0.9426268272160425]
#      ---------------------------------------------
#      Fidelity at Step 144: 0.9999850374815137
#      Parameters at Step 144: [0.49165454993302277, 0.19157234440392285, 1.3278888659113894, 1.6990801687879322, -0.03415337895580718, -0.7686115048888433, -1.1241119264423507, -1.3635854280358073, 0.994637952532866, -0.9456455164014339]
#      ---------------------------------------------
#      Fidelity at Step 145: 0.9999861831973936
#      Parameters at Step 145: [0.4948516637510453, 0.18771062041704265, 1.3279261867569543, 1.700750702088882, -0.03371833064936751, -0.7703351317991944, -1.130353693041737, -1.3635155517339068, 0.9943020382966017, -0.9488152011860097]
#      ---------------------------------------------
#      Fidelity at Step 146: 0.9999849895773791
#      Parameters at Step 146: [0.4979179564776596, 0.1839432903772568, 1.3277729869065453, 1.7023684683632287, -0.033421590116644154, -0.7718574676594108, -1.1364400887029087, -1.3634364571146622, 0.9939995210390251, -0.9520625040412555]
#      ---------------------------------------------
#      Fidelity at Step 147: 0.9999867654127862
#      Parameters at Step 147: [0.5008672133519527, 0.18029084997712289, 1.3277142937941975, 1.704005832696915, -0.03317473744913395, -0.7733145786739568, -1.1424194449958749, -1.3633827960967202, 0.9937519907295783, -0.955299499225928]
#      ---------------------------------------------
#      Fidelity at Step 148: 0.9999887913045802
#      Parameters at Step 148: [0.503651682297394, 0.17679445321469484, 1.3277508023942615, 1.7056658715580102, -0.032895307435838114, -0.774767538847392, -1.1482963708561775, -1.3633806485383175, 0.9935767702098116, -0.958419192994982]
#      ---------------------------------------------
#      Fidelity at Step 149: 0.999985035956237
#      Parameters at Step 149: [0.506274433952535, 0.1734509832221617, 1.327941771421969, 1.7073681258447921, -0.032527136133335424, -0.7762749830110723, -1.154080538510611, -1.3634484542330767, 0.9934845033090591, -0.9613654595093701]
#      ---------------------------------------------
#      Fidelity at Step 150: 0.9999878877045175
#      Parameters at Step 150: [0.508749462189302, 0.17018243939953917, 1.328248119010844, 1.7091637132690014, -0.03196923377526107, -0.7779311273700311, -1.1599394135997274, -1.363615635769636, 0.9934987056822612, -0.9641197343895163]
#      ---------------------------------------------
#      Fidelity at Step 151: 0.9999872199533634
#      Parameters at Step 151: [0.5110306887058557, 0.16704257626956867, 1.328412707125246, 1.7109354590868575, -0.03128792053631156, -0.7796047782485971, -1.1656706186958778, -1.3638443013173356, 0.9935892840802129, -0.9666568943461408]
#      ---------------------------------------------
#      Fidelity at Step 152: 0.9999846672973843
#      Parameters at Step 152: [0.5132716374928659, 0.16400649350758884, 1.3290566977033718, 1.7127763720480835, -0.030506251567427905, -0.7814239682976281, -1.171216767607195, -1.3641084371106356, 0.9937345066138964, -0.9689872541512644]
#      ---------------------------------------------
#      Fidelity at Step 153: 0.9999867351528788
#      Parameters at Step 153: [0.5154380642884647, 0.16101081197847836, 1.329681664784569, 1.7146132461957844, -0.029644046624950387, -0.7832743982470529, -1.1766086526335258, -1.3643966310971727, 0.9939355408878404, -0.9711722947871008]
#      ---------------------------------------------
#      Fidelity at Step 154: 0.9999862241484375
#      Parameters at Step 154: [0.5174946573552897, 0.15803941295449758, 1.3298958677221986, 1.7163569021894192, -0.028766525059498282, -0.7850159870432324, -1.1817991369158976, -1.3646823960544083, 0.9941758238430728, -0.9732732065211951]
#      ---------------------------------------------
#      Fidelity at Step 155: 0.9999830014109053
#      Parameters at Step 155: [0.5195040670339732, 0.15504745752530472, 1.329762082040617, 1.7180207634581337, -0.02790811151081738, -0.7866483216268434, -1.1867998664498791, -1.3649424237865677, 0.9944472518870449, -0.9753472278100357]
#      ---------------------------------------------
#      Fidelity at Step 156: 0.9999864511000253
#      Parameters at Step 156: [0.521716676233046, 0.15176583730252977, 1.32957639767777, 1.719723619205503, -0.02705181275335927, -0.7883125895766997, -1.1918166235153695, -1.3651207890826964, 0.9947525014995762, -0.977549037634112]
#      ---------------------------------------------
#      Fidelity at Step 157: 0.9999826513650695
#      Parameters at Step 157: [0.5239964399597866, 0.14844427349129943, 1.329535680851092, 1.7214165735621738, -0.026241741176397944, -0.7899683259992069, -1.1967061184083099, -1.365230019906992, 0.995068887255867, -0.979764123843502]
#      ---------------------------------------------
#      Fidelity at Step 158: 0.9999882498026169
#      Parameters at Step 158: [0.5262861536543998, 0.14504362765356396, 1.3292303431819674, 1.7230464001713983, -0.025476035965192906, -0.7915423527347953, -1.2016220671627142, -1.365248690384345, 0.9954066878505755, -0.982048867852817]
#      ---------------------------------------------
#      Fidelity at Step 159: 0.9999857804291704
#      Parameters at Step 159: [0.5285783161964774, 0.14174543081676313, 1.3293491757849916, 1.7246782438510426, -0.02475599728313822, -0.7931466879198291, -1.206408566689147, -1.3652104189458465, 0.995741455288656, -0.9842759850445346]
#      ---------------------------------------------
#      Fidelity at Step 160: 0.9999873512494414
#      Parameters at Step 160: [0.5307594882904066, 0.13849537736632228, 1.3291680039841776, 1.7262214114795207, -0.024073128205469314, -0.7946463806378471, -1.2112733331631036, -1.3651025135799986, 0.9960957414433877, -0.9865214428462055]
#      ---------------------------------------------
#      Fidelity at Step 161: 0.9999886461524623
#      Parameters at Step 161: [0.5328458040346352, 0.13532214379578952, 1.328966681660508, 1.7277096320780783, -0.023409888019261096, -0.7961063813366783, -1.2162034351667637, -1.364932886473924, 0.996457568976065, -0.9887460480457212]
#      ---------------------------------------------
#      Fidelity at Step 162: 0.9999902032474028
#      Parameters at Step 162: [0.5348340653861403, 0.13225505344324726, 1.328847208373762, 1.7291647442219011, -0.02275233771033779, -0.7975569869396825, -1.2211816293722073, -1.3647211936191672, 0.9968308645104235, -0.9909191256609178]
#      ---------------------------------------------
#      Fidelity at Step 163: 0.9999889132157431
#      Parameters at Step 163: [0.5366843777290922, 0.12933385256430174, 1.3286998070140525, 1.7305499865191734, -0.022101471015799637, -0.7989569513789231, -1.2260919913498218, -1.3644891300526791, 0.9972069997644573, -0.9929949238177518]
#      ---------------------------------------------
#      Fidelity at Step 164: 0.999989502893319
#      Parameters at Step 164: [0.5385235144431655, 0.12657531222828117, 1.3291695036954272, 1.731969724717606, -0.021470826093723947, -0.8004348306344797, -1.2308069913603101, -1.3642503959184866, 0.9975698806550973, -0.9949470347889933]
#      ---------------------------------------------
#      Fidelity at Step 165: 0.9999843597471785
#      Parameters at Step 165: [0.5402555117761144, 0.12386269780869166, 1.3294479006466213, 1.7333139145782528, -0.020846006937108762, -0.801838413403773, -1.2355098657138137, -1.3639852050723429, 0.9979320419695976, -0.9968610486132228]
#      ---------------------------------------------
#      Fidelity at Step 166: 0.9999892815397706
#      Parameters at Step 166: [0.5419763943442601, 0.12107381862368885, 1.3296300899917965, 1.7346385301625344, -0.020234793790248003, -0.8032058316245257, -1.2402922602982387, -1.3636810725471509, 0.9982968131525182, -0.9988249045527734]
#      ---------------------------------------------
#      Fidelity at Step 167: 0.9999868476099629
#      Parameters at Step 167: [0.5436811281664722, 0.118345847639199, 1.3299671848217822, 1.7359542632234002, -0.01966693203088115, -0.8045450432272869, -1.2448980877577691, -1.3633785403963068, 0.9986538360575402, -1.0007502576104403]
#      ---------------------------------------------
#      Fidelity at Step 168: 0.999988211891881
#      Parameters at Step 168: [0.5453907419233148, 0.11562234491250503, 1.3303961942731146, 1.737252260673979, -0.019144809206808718, -0.8058398287593344, -1.249321291250448, -1.3630740644029284, 0.9989949914004753, -1.002662997387518]
#      ---------------------------------------------
#      Fidelity at Step 169: 0.9999928232199977
#      Parameters at Step 169: [0.5470298089619259, 0.11294845433054324, 1.3306209946783156, 1.7384770651864752, -0.018675792225203727, -0.8070111888957845, -1.2535154412184717, -1.3627845973676904, 0.9993249606501416, -1.0045453852920951]
#      ---------------------------------------------
#      Fidelity at Step 170: 0.9999887321883951
#      Parameters at Step 170: [0.5486670595875374, 0.11037292141096658, 1.3311378474407562, 1.739711282598617, -0.01825573027893023, -0.8081728492461591, -1.2574421460778868, -1.3625190854901312, 0.9996410514327894, -1.0063623030994178]
#      ---------------------------------------------
#      Fidelity at Step 171: 0.9999920762371789
#      Parameters at Step 171: [0.5502349300327534, 0.10779993295101048, 1.3313410830723522, 1.7408785660065607, -0.01787842457962252, -0.8092012976753657, -1.26118384893814, -1.362280184910283, 0.9999541327857622, -1.008177363562292]
#      ---------------------------------------------
#      Fidelity at Step 172: 0.9999868204329156
#      Parameters at Step 172: [0.5518084300106664, 0.10527005598833068, 1.3317547409661767, 1.7420676848269074, -0.017533696191392507, -0.8102232480918031, -1.2647257065399566, -1.3620720984869334, 1.0002613731708783, -1.0099566960612834]
#      ---------------------------------------------
#      Fidelity at Step 173: 0.9999913559311914
#      Parameters at Step 173: [0.5533847676018934, 0.1026629991782988, 1.3321004972759545, 1.7432814900492706, -0.01719997151269994, -0.811207265200446, -1.2681875031077672, -1.3619056071299658, 1.0005797750452579, -1.011770234432686]
#      ---------------------------------------------
#      Fidelity at Step 174: 0.9999913214040246
#      Parameters at Step 174: [0.5548821612889252, 0.10011060185804076, 1.3323360404668545, 1.7444747043954105, -0.01687189563828634, -0.8121357317212428, -1.2715295445300454, -1.3617802772055234, 1.000907299920358, -1.0135406958891202]
#      ---------------------------------------------
#      Fidelity at Step 175: 0.9999860300741775
#      Parameters at Step 175: [0.5563210452484355, 0.0976058452896269, 1.3326247104410003, 1.7456829155834255, -0.01652412298005788, -0.8130683214357861, -1.2748043515034768, -1.3616955370247923, 1.0012482944684757, -1.0152569520851642]
#      ---------------------------------------------
#      Fidelity at Step 176: 0.9999869130137046
#      Parameters at Step 176: [0.5576730167381697, 0.09506111198725749, 1.332764702975683, 1.7469248277458853, -0.016100468936757248, -0.8140193217285754, -1.2782028243333425, -1.361661260556583, 1.0016330592746054, -1.0169543314815557]
#      ---------------------------------------------
#      Fidelity at Step 177: 0.9999877217147968
#      Parameters at Step 177: [0.5589274852929289, 0.09258159463488037, 1.3328613000011675, 1.7481529641823588, -0.015629042269218567, -0.8149771282707365, -1.2816037761035932, -1.361648318523111, 1.0020367177663734, -1.0185712748653197]
#      ---------------------------------------------
#      Fidelity at Step 178: 0.9999872658409166
#      Parameters at Step 178: [0.5602276069540725, 0.09013573257411094, 1.3335546522621782, 1.7494895597083502, -0.015105424148538587, -0.8161053194032152, -1.2850558032987738, -1.3616335521110063, 1.0024591565155927, -1.0201225639762055]
#      ---------------------------------------------
#      Fidelity at Step 179: 0.9999875682031998
#      Parameters at Step 179: [0.5614533265372388, 0.08765171564005021, 1.333926828438151, 1.750790180328323, -0.01456972519685228, -0.8171790361138784, -1.2886525881175384, -1.361586512781489, 1.0029150679738548, -1.0217108790652585]
#      ---------------------------------------------
#      Fidelity at Step 180: 0.9999901610429807
#      Parameters at Step 180: [0.5626325362460528, 0.08529226687868985, 1.334338451916551, 1.75203049108332, -0.01408242264275763, -0.8181988672647009, -1.2920751971204063, -1.3615175757134808, 1.0033517777316945, -1.0232391229953068]
#      ---------------------------------------------
#      Fidelity at Step 181: 0.9999924934552326
#      Parameters at Step 181: [0.5637724360868879, 0.08301598861453113, 1.3346625456129624, 1.7531977142841724, -0.013653694695672566, -0.8191386884772068, -1.2953676293760934, -1.3614089750352865, 1.0037750754379051, -1.0247469653569756]
#      ---------------------------------------------
#      Fidelity at Step 182: 0.9999896337416619
#      Parameters at Step 182: [0.5648823856161473, 0.08081632764422314, 1.3349008987446196, 1.7542874802301704, -0.013285444609570781, -0.8199977576502516, -1.298518194796666, -1.3612559517700367, 1.0041812470182039, -1.02623464086704]
#      ---------------------------------------------
#      Fidelity at Step 183: 0.9999886552526498
#      Parameters at Step 183: [0.5659564124151729, 0.07863183162668098, 1.3348109422735874, 1.7552690439422955, -0.012978665788255614, -0.8207337329914272, -1.3015909474075096, -1.361033993599378, 1.004578488805329, -1.0277434686827096]
#      ---------------------------------------------
#      Fidelity at Step 184: 0.9999896963669672
#      Parameters at Step 184: [0.5670407395555168, 0.0764062248474207, 1.3345306186077732, 1.7561739406827335, -0.012682726906812027, -0.8214300293705896, -1.304667668449513, -1.3607179965360254, 1.0049775004301271, -1.0292546807793863]
#      ---------------------------------------------
#      Fidelity at Step 185: 0.9999892555185942
#      Parameters at Step 185: [0.5681657453963701, 0.0742498846666968, 1.3346425503959116, 1.7570930899011632, -0.012365431398008667, -0.822222072957483, -1.3076478743542113, -1.3603657170326973, 1.0053656310008658, -1.030665191966978]
#      ---------------------------------------------
#      Fidelity at Step 186: 0.9999873008804145
#      Parameters at Step 186: [0.5692469331809209, 0.07214852738005786, 1.334726456522285, 1.7579694445496343, -0.01199912569255848, -0.8230416451427469, -1.310601063990293, -1.359981830405764, 1.005761025868443, -1.031971760971129]
#      ---------------------------------------------
#      Fidelity at Step 187: 0.9999902918781893
#      Parameters at Step 187: [0.5703082261411033, 0.0700651796476419, 1.334896749157116, 1.7588420377632485, -0.011550108087819182, -0.8239479011347973, -1.3136022649324255, -1.3595645721844367, 1.006177612859388, -1.0331729024355931]
#      ---------------------------------------------
#      Fidelity at Step 188: 0.9999885080739047
#      Parameters at Step 188: [0.5713983164659022, 0.06808727518476335, 1.3355403183830272, 1.7597628961459177, -0.011100614964728651, -0.8249430455093314, -1.3164774463852682, -1.35915795659551, 1.0065828112639668, -1.0342885329375568]
#      ---------------------------------------------
#      Fidelity at Step 189: 0.9999862800292353
#      Parameters at Step 189: [0.572434804434997, 0.06612847682904653, 1.3359856559587966, 1.7606464812958231, -0.010677275161958846, -0.825864146318007, -1.3193148628423925, -1.3587522835441175, 1.0069937130870932, -1.0354150953922756]
#      ---------------------------------------------
#      Fidelity at Step 190: 0.999986941039955
#      Parameters at Step 190: [0.5734157712423219, 0.0642115062042154, 1.3362412133150423, 1.7614910453000487, -0.01031387695515812, -0.8266793210934649, -1.3220600557304993, -1.358363678558335, 1.0074003729498613, -1.0365720553497828]
#      ---------------------------------------------
#      Fidelity at Step 191: 0.9999890579779093
#      Parameters at Step 191: [0.5742763144740772, 0.062338167532080394, 1.3359942272632255, 1.762246736010568, -0.01000955999141036, -0.8273153806758138, -1.3247240659672528, -1.3580029613090359, 1.0078091831397777, -1.037765435378209]
#      ---------------------------------------------
#      Fidelity at Step 192: 0.9999798614554044
#      Parameters at Step 192: [0.5750348275406495, 0.06050912454155909, 1.3354045794993685, 1.7629382965909832, -0.009732835043158979, -0.8278312897232609, -1.3273137762418075, -1.3576734333506415, 1.0082177525611982, -1.0389580051847715]
#      ---------------------------------------------
#      Fidelity at Step 193: 0.9999867211457386
#      Parameters at Step 193: [0.575541937870243, 0.058638268388376595, 1.333829424344813, 1.7635015625270305, -0.00933054933450848, -0.8282103517661018, -1.330068048247978, -1.3573865132373175, 1.0086873691308977, -1.040078785952071]
#      ---------------------------------------------
#      Fidelity at Step 194: 0.9999885173075377
#      Parameters at Step 194: [0.5759330636780624, 0.05686170692563526, 1.332260716345552, 1.7640508985116936, -0.008758432038375216, -0.8287101056883912, -1.3328695949297107, -1.3571284961271388, 1.0091873394591213, -1.0409664786968056]
#      ---------------------------------------------
#      Fidelity at Step 195: 0.9999914414076407
#      Parameters at Step 195: [0.5761698087898832, 0.05520787437158809, 1.3306213537620428, 1.764541391429165, -0.007951988821247653, -0.8293609243539548, -1.3357305590793547, -1.3568807266091802, 1.0097224300509535, -1.041523503066072]
#      ---------------------------------------------
#      Fidelity at Step 196: 0.9999909358323312
#      Parameters at Step 196: [0.5764460927363217, 0.05365713280830934, 1.3297462583876818, 1.7650963298974642, -0.006946505580886105, -0.8303293111980891, -1.3385944472376565, -1.356607494314803, 1.0102670663121784, -1.041766147095695]
#      ---------------------------------------------
#      Fidelity at Step 197: 0.9999906039213291
#      Parameters at Step 197: [0.5767220976396684, 0.05221967959575094, 1.3291124851999803, 1.7656255362200648, -0.005933869704823914, -0.8313403443762636, -1.341316897342956, -1.3563194698910808, 1.0107906440580432, -1.0418902860638228]
#      ---------------------------------------------
#      Fidelity at Step 198: 0.999989638431839
#      Parameters at Step 198: [0.5770661079103651, 0.05082746036319119, 1.3287969456999924, 1.7661289073141504, -0.004936116431480546, -0.8324038971128354, -1.343923477449557, -1.355964787344698, 1.0112947775188423, -1.0419333188145385]
#      ---------------------------------------------
#      Fidelity at Step 199: 0.999991606617489
#      Parameters at Step 199: [0.5775094191382846, 0.04945618421796187, 1.328710317887879, 1.7666051097500668, -0.004059246822290077, -0.8334229380088882, -1.3463756706953927, -1.355537991382031, 1.0117698517998235, -1.042021085504792]
#      ---------------------------------------------
#      Fidelity at Step 200: 0.9999887382906469
#      Parameters at Step 200: [0.578030459572647, 0.048039145898378995, 1.328377894191988, 1.7669658843418832, -0.0034115315624715353, -0.8342053910707248, -1.3486503101922291, -1.3550075140083484, 1.0122104791186208, -1.0422898758521553]
#      ---------------------------------------------
#      Fidelity at Step 201: 0.9999925504526961
#      Parameters at Step 201: [0.5786045080319537, 0.046515376553358335, 1.3274020566780673, 1.7671434184496464, -0.0030863047854294773, -0.8345862731337464, -1.350754920349443, -1.3543502995139975, 1.012623699363371, -1.0428631724087638]
#      ---------------------------------------------
#      Fidelity at Step 202: 0.999987255051033
#      Parameters at Step 202: [0.5792017647238181, 0.04507292120345454, 1.3266199087241262, 1.7673322669051341, -0.002871042638470988, -0.8349088800492177, -1.3527100967128156, -1.3537096189628486, 1.013010919594112, -1.043485932366876]
#      ---------------------------------------------
#      Fidelity at Step 203: 0.9999888907573472
#      Parameters at Step 203: [0.5797387075603089, 0.04364080969232375, 1.325521237782578, 1.7674533054630355, -0.0027846252235358806, -0.8350404370721993, -1.3546345976760379, -1.3530614356430213, 1.0134144707666521, -1.044209093510324]
#      ---------------------------------------------
#      Fidelity at Step 204: 0.9999881322007121
#      Parameters at Step 204: [0.5802396206810376, 0.0422928753246885, 1.324706835881735, 1.7676251500038638, -0.0026872600106955867, -0.8352189681322226, -1.3565439873059322, -1.3524655967666366, 1.0138380231743556, -1.0448691491124078]
#      ---------------------------------------------
#      Fidelity at Step 205: 0.9999894671754493
#      Parameters at Step 205: [0.5806914874725001, 0.04104581140421217, 1.3244634196493874, 1.7679304385405523, -0.0024719827456277303, -0.8355911829340839, -1.358539534277575, -1.3519640678407858, 1.0143193814162696, -1.0453599992552314]
#      ---------------------------------------------
#      Fidelity at Step 206: 0.9999887550530641
#      Parameters at Step 206: [0.58112054461415, 0.03988085878168208, 1.324829539004447, 1.768372668192599, -0.0021429946527134463, -0.8361566271138334, -1.3605549445614127, -1.3515688560968233, 1.0148434248146052, -1.0456880730062577]
#      ---------------------------------------------
#      Fidelity at Step 207: 0.9999886259446119
#      Parameters at Step 207: [0.5814592119397126, 0.038760976413281766, 1.3251588665269407, 1.7688472494478302, -0.0017760094832740304, -0.8367119264841479, -1.3625314031351388, -1.3512812908547716, 1.0154026011795458, -1.0459490260414839]
#      ---------------------------------------------
#      Fidelity at Step 208: 0.9999881139367572
#      Parameters at Step 208: [0.5818449253018668, 0.03765135420895377, 1.325802181057237, 1.7694035248856217, -0.0014506017523894395, -0.837284168467825, -1.36437529011688, -1.351070052402854, 1.0159605514890004, -1.0462281768586885]
#      ---------------------------------------------
#      Fidelity at Step 209: 0.999987619645847
#      Parameters at Step 209: [0.5821125428709006, 0.036466933319086334, 1.325516216879589, 1.7698410345525648, -0.0012633434400915718, -0.8375101782469347, -1.3660476637189252, -1.350942916995247, 1.0165309547632595, -1.046658068686272]
#      ---------------------------------------------
#      Fidelity at Step 210: 0.99999025342367
#      Parameters at Step 210: [0.582340225734264, 0.03517995166803719, 1.3246358185219782, 1.7702078978309685, -0.0011893135878432212, -0.8374859194741336, -1.3675400627278114, -1.3508856113579193, 1.0171132484957168, -1.0472076010069815]
#      ---------------------------------------------
#      Fidelity at Step 211: 0.9999884765567478
#      Parameters at Step 211: [0.5827442110195769, 0.03389662460962347, 1.3246663160569445, 1.7707286210834383, -0.001116984343657353, -0.8376557949861562, -1.3689037468637544, -1.3508563250802832, 1.0176748261746098, -1.0477187392767007]
#      ---------------------------------------------
#      Fidelity at Step 212: 0.9999904561856239
#      Parameters at Step 212: [0.5832025352877948, 0.032631330812688694, 1.3250559757662133, 1.771294272313985, -0.0010245966853708556, -0.8379075762354566, -1.3701751164810776, -1.3508489587041526, 1.0182246556232561, -1.0481672409301965]
#      ---------------------------------------------
#      Fidelity at Step 213: 0.9999901858354218
#      Parameters at Step 213: [0.5836605704891774, 0.031376665348989655, 1.3255095674442012, 1.7718537512835941, -0.0009259331585511649, -0.8381631597635073, -1.3713892089714161, -1.3508567602125896, 1.0187724587122158, -1.0485744541707018]
#      ---------------------------------------------
#      Fidelity at Step 214: 0.999988435759673
#      Parameters at Step 214: [0.5841644644043431, 0.0301670059306356, 1.326257830920721, 1.7724447901025937, -0.0008609609656053234, -0.8384547421229761, -1.3725448268533085, -1.3508618728232376, 1.0192939089925728, -1.0489831592110408]
#      ---------------------------------------------
#      Fidelity at Step 215: 0.9999895762160621
#      Parameters at Step 215: [0.5846071363430934, 0.028952547931390647, 1.3266199248264239, 1.7729538086743766, -0.0008886357076636716, -0.8385738392256091, -1.3736919164157642, -1.3508571862340193, 1.019813802841107, -1.049476081419914]
#      ---------------------------------------------
#      Fidelity at Step 216: 0.9999865439963951
#      Parameters at Step 216: [0.5850557559413209, 0.027766459985979788, 1.327061556266125, 1.7734498368139349, -0.0009720306347003305, -0.8386553607704775, -1.3748193624390446, -1.3508389982062428, 1.020319208258268, -1.0500056998757614]
#      ---------------------------------------------
#      Fidelity at Step 217: 0.9999885827559089
#      Parameters at Step 217: [0.5854117281207973, 0.026620914277741707, 1.327155156248737, 1.7738595553203997, -0.0010939920041759354, -0.8386143736767395, -1.3759722960439218, -1.3508104401024605, 1.0208262718879615, -1.050558252967573]
#      ---------------------------------------------
#      Fidelity at Step 218: 0.999988349649449
#      Parameters at Step 218: [0.5858236667875478, 0.025519249618372042, 1.3277427224692333, 1.7743297264414288, -0.0011998850761463797, -0.8386915295261786, -1.3771535054686013, -1.350765659221492, 1.0213253207632875, -1.051067274653289]
#      ---------------------------------------------
#      Fidelity at Step 219: 0.9999927873362311
#      Parameters at Step 219: [0.5862885686180744, 0.02446858506824528, 1.328768317162256, 1.7748567588791937, -0.001287037408261083, -0.8388806807238663, -1.3783355403979574, -1.3507124604760232, 1.0218139081740487, -1.051532499293655]
#      ---------------------------------------------
#      Fidelity at Step 220: 0.9999904472208753
#      Parameters at Step 220: [0.5868062589518784, 0.023450048412156033, 1.330064725519595, 1.7754166728690355, -0.001396332986240341, -0.8391133113320853, -1.3794656098399214, -1.3506545039230118, 1.0222811084089933, -1.0520065794444358]
#      ---------------------------------------------
#      Fidelity at Step 221: 0.999989842546006
#      Parameters at Step 221: [0.5872844947166628, 0.022425904775236594, 1.3309321801533682, 1.7758978858197463, -0.0015924064284546574, -0.8391785376958127, -1.3805132302165977, -1.3505976173625995, 1.0227283344789568, -1.0525785601205644]
#      ---------------------------------------------
#      Fidelity at Step 222: 0.9999870275056041
#      Parameters at Step 222: [0.5877537904766912, 0.021392833563420723, 1.3315298501957273, 1.7763328917515049, -0.0018538439441622395, -0.8391290190289682, -1.3814592840869555, -1.3505514496180628, 1.0231558695945016, -1.0532272656902077]
#      ---------------------------------------------
#      Fidelity at Step 223: 0.9999884661453955
#      Parameters at Step 223: [0.5884048263614704, 0.020321388108284218, 1.332843939475398, 1.7769053981589935, -0.0021503852953904654, -0.8392136781452616, -1.3822951490721656, -1.3505225289686806, 1.0235581477181113, -1.0539266248019814]
#      ---------------------------------------------
#      Fidelity at Step 224: 0.9999884532070813
#      Parameters at Step 224: [0.5889900674029886, 0.01924770428584472, 1.3336934487269372, 1.7774144968686065, -0.002489047082506203, -0.8391561943730307, -1.3830499513386436, -1.3505289113292414, 1.0239430903605118, -1.0546918015277298]
#      ---------------------------------------------
#      Fidelity at Step 225: 0.9999898475733888
#      Parameters at Step 225: [0.5895506048504998, 0.018194378508048982, 1.3344672926785306, 1.7779352447027705, -0.0028199249136769576, -0.8390843005297155, -1.3837557963187326, -1.3505831368072339, 1.024316936801018, -1.0554693272819546]
#      ---------------------------------------------
#      Fidelity at Step 226: 0.9999907428905219
#      Parameters at Step 226: [0.5901977370756432, 0.01720730735481824, 1.3358847432853092, 1.7785804675311812, -0.0031111741874706956, -0.839195479627143, -1.3844284679651042, -1.3506652385280777, 1.0246693620289369, -1.0561995176813221]
#      ---------------------------------------------
#      Fidelity at Step 227: 0.9999906102497097
#      Parameters at Step 227: [0.590844867359659, 0.01627999061901328, 1.3374478668238698, 1.7792678723385187, -0.0033879803093705956, -0.8393568484949371, -1.385075003542551, -1.350777094926472, 1.0249992681104725, -1.0569197423061372]
#      ---------------------------------------------
#      Fidelity at Step 228: 0.9999910334499799
#      Parameters at Step 228: [0.591543823070571, 0.015365295171680707, 1.339289895838081, 1.7800321846247285, -0.00368734918107617, -0.8395639503160266, -1.3856726658278233, -1.3509285028095877, 1.0252961632984325, -1.057697135594466]
#      ---------------------------------------------
#      Fidelity at Step 229: 0.999988409038041
#      Parameters at Step 229: [0.5922323634726342, 0.01449344407434168, 1.341072365846745, 1.7807940042253934, -0.004020935168399662, -0.8397365382906985, -1.386217667855764, -1.351093930735195, 1.0255610966677864, -1.058518138916406]
#      ---------------------------------------------
#      Fidelity at Step 230: 0.9999901039286563
#      Parameters at Step 230: [0.5928132620427989, 0.013598646052713477, 1.3420267384272355, 1.7814540200052944, -0.004481787594141734, -0.8396210033567889, -1.3866913758321575, -1.3513050735492125, 1.0257866556443531, -1.059548334044546]
#      ---------------------------------------------
#      Fidelity at Step 231: 0.9999915196733226
#      Parameters at Step 231: [0.5932533362477989, 0.012721532992385855, 1.3421873232196984, 1.782005536688833, -0.004981545274549092, -0.8393000221813757, -1.3871439407418964, -1.3515519569762011, 1.0259807639210963, -1.0606729599407296]
#      ---------------------------------------------
#      Fidelity at Step 232: 0.9999884483776887
#      Parameters at Step 232: [0.5936264913568032, 0.011918917506283057, 1.3422479640294946, 1.7825284524150424, -0.005397205500562751, -0.8390286721940902, -1.387612788995174, -1.3518038286523277, 1.02615340247575, -1.0617009680045686]
#      ---------------------------------------------
#      Fidelity at Step 233: 0.9999893754195205
#      Parameters at Step 233: [0.5939147481832755, 0.011224914908710732, 1.342500451027886, 1.7830627156054486, -0.005544808905161805, -0.8390145839131689, -1.3881919560065117, -1.3520723348736075, 1.026317765175582, -1.0624413935125638]
#      ---------------------------------------------
#      Fidelity at Step 234: 0.9999869842920284
#      Parameters at Step 234: [0.5941250004028764, 0.010646979257375278, 1.343025322096422, 1.783601059347741, -0.0053556782844705235, -0.8393379560831443, -1.3889179381326244, -1.352327019736342, 1.0264676445133947, -1.0628130998079022]
#      ---------------------------------------------
#      Fidelity at Step 235: 0.9999893059352454
#      Parameters at Step 235: [0.5942117178606191, 0.010163933474947767, 1.3432180484802942, 1.784028868450274, -0.004968590479311849, -0.8397604447145343, -1.38973655486462, -1.3525316255797117, 1.026605482439269, -1.0629541374178728]
#      ---------------------------------------------
#      Fidelity at Step 236: 0.9999896434642385
#      Parameters at Step 236: [0.5943484019952318, 0.009694666764780353, 1.3434811896287295, 1.78436741956631, -0.004450894883286583, -0.8403327344086806, -1.390590184555642, -1.352592763454403, 1.0266775758984168, -1.0629430110938438]
#      ---------------------------------------------
#      Fidelity at Step 237: 0.9999901073736958
#      Parameters at Step 237: [0.5945756158662212, 0.009196184216358051, 1.3436384200091407, 1.7845877212992054, -0.003988396553925689, -0.8408677281742165, -1.3913982477537676, -1.3524894743047176, 1.0266746373408528, -1.0629792281037513]
#      ---------------------------------------------
#      Fidelity at Step 238: 0.9999927691264081
#      Parameters at Step 238: [0.594703642656405, 0.008692772398369507, 1.3427089405740413, 1.784543553512466, -0.0036730206376294873, -0.8410659516918019, -1.3921709162345033, -1.3522501730040704, 1.0266266948791514, -1.0631680015636504]
#      ---------------------------------------------
#      Fidelity at Step 239: 0.9999917315733722
#      Parameters at Step 239: [0.5948067580090348, 0.008218803138169877, 1.3414399845389282, 1.7843878034662675, -0.0034434592965103803, -0.8411340592696075, -1.3929314198469678, -1.3519230900200534, 1.0265522515142744, -1.0634263593280144]
#      ---------------------------------------------
#      Fidelity at Step 240: 0.9999905424814088
#      Parameters at Step 240: [0.5949800696134573, 0.007805234926347038, 1.3406572208595793, 1.784297829954337, -0.003224236022222644, -0.8413117231071137, -1.3936942408910464, -1.3515670053333104, 1.0264749143994978, -1.063658562179518]
#      ---------------------------------------------
#      Fidelity at Step 241: 0.9999885826045259
#      Parameters at Step 241: [0.5949962815780454, 0.007512344113460753, 1.3394914506452924, 1.7841052004053317, -0.002924095968009788, -0.8414742961671783, -1.3945856033212118, -1.351167893492728, 1.0264217099709094, -1.063775584532843]
#      ---------------------------------------------
#      Fidelity at Step 242: 0.9999883606888723
#      Parameters at Step 242: [0.5948245777087302, 0.0073659552955860266, 1.3380419916583655, 1.7838336882096821, -0.0024508502494072974, -0.8417114284919093, -1.3956439248697783, -1.3507514127294864, 1.026415173035054, -1.063672136033635]
#      ---------------------------------------------
#      Fidelity at Step 243: 0.9999861511179472
#      Parameters at Step 243: [0.5945540945025368, 0.00734998275974998, 1.3368001062312966, 1.783572564000042, -0.0017556868828075458, -0.8421690874043849, -1.3968405142949893, -1.3503339085157124, 1.0264641452937509, -1.0632846088543018]
#      ---------------------------------------------
#      Fidelity at Step 244: 0.9999911580594937
#      Parameters at Step 244: [0.5941391817848185, 0.0074079919451457265, 1.3352179770944113, 1.7832192624843257, -0.0008842166745967229, -0.8426835128816891, -1.3980958687205038, -1.3499101993315223, 1.0265699236460317, -1.0626600448263706]
#      ---------------------------------------------
#      Fidelity at Step 245: 0.9999898212508886
#      Parameters at Step 245: [0.593766426167929, 0.007461086818384863, 1.3337732245594254, 1.782883782463429, -3.520519593994305e-05, -0.843206839122646, -1.399266288825347, -1.3494930655712347, 1.0266859103435684, -1.0620202439793718]
#      ---------------------------------------------
#      Fidelity at Step 246: 0.9999856379788754
#      Parameters at Step 246: [0.5935274087548836, 0.007438726797372935, 1.3325679928963028, 1.782559960006288, 0.0007574560546551006, -0.8437377803837636, -1.4002772006975712, -1.3490425506377353, 1.0268113796108818, -1.061391656073096]
#      ---------------------------------------------
#      Fidelity at Step 247: 0.9999883239594803
#      Parameters at Step 247: [0.5933535967481981, 0.007343159535556303, 1.3311877546850395, 1.7821986098049338, 0.001414719058850634, -0.8441183829929937, -1.4011219020749626, -1.3485820724101096, 1.0269495375873934, -1.0608717265285883]
#      ---------------------------------------------
#      Fidelity at Step 248: 0.9999900698059424
#      Parameters at Step 248: [0.5934131491707247, 0.00715239199806403, 1.330460118591667, 1.7819551746946658, 0.0018896686076640776, -0.8444998490167378, -1.4017840398455415, -1.3481123911230533, 1.027081520629623, -1.0605114227579961]
#      ---------------------------------------------
#      Fidelity at Step 249: 0.9999885539374775
#      Parameters at Step 249: [0.5935062758562192, 0.006879797546502684, 1.3294961145063628, 1.781681104239747, 0.002145344361266706, -0.8446309996057094, -1.4022846754026792, -1.3476699145790654, 1.027214005311259, -1.0603643602112434]
#      ---------------------------------------------
#      Fidelity at Step 250: 0.9999895300027376
#      Parameters at Step 250: [0.5936777625599198, 0.006599806437940118, 1.3289779761618017, 1.7815249122821493, 0.002275726765265315, -0.8447460865153723, -1.402721063135856, -1.347291975458325, 1.0273548112675543, -1.060335920350216]
#      ---------------------------------------------
#      Fidelity at Step 251: 0.9999897825367936
#      Parameters at Step 251: [0.593862963547294, 0.006332944207955403, 1.3287970691222821, 1.781478031764532, 0.0023094209749266086, -0.8448313009439179, -1.4031274797944795, -1.347011456821376, 1.0275121600687303, -1.0604008336432516]
#      ---------------------------------------------
#      Fidelity at Step 252: 0.9999896387188143
#      Parameters at Step 252: [0.5939058088165182, 0.006097596632468126, 1.3282866949915018, 1.7814292255724553, 0.002266040513625809, -0.8447432600265276, -1.4035341399018701, -1.3468518581022193, 1.0276909463279056, -1.0605455629161815]
#      ---------------------------------------------
#      Fidelity at Step 253: 0.9999887068802289
#      Parameters at Step 253: [0.59387067843434, 0.005906120856296461, 1.32777561050331, 1.781422298638998, 0.0022220968178467077, -0.8446309645301731, -1.4039513992891786, -1.3467865416862939, 1.0278852875637112, -1.060682381012993]
#      ---------------------------------------------
#      Fidelity at Step 254: 0.9999912082647311
#      Parameters at Step 254: [0.5937380705822191, 0.005765497119177365, 1.3275153758191351, 1.7815164473039649, 0.0022604555415742266, -0.8445948794870505, -1.4044082361672299, -1.3468767773236676, 1.0281166962573705, -1.0607094184704775]
#      ---------------------------------------------
#      Fidelity at Step 255: 0.9999913261847386
#      Parameters at Step 255: [0.5936578158955296, 0.005640217289234351, 1.3278276470580708, 1.7817332040436074, 0.002350305575703594, -0.8447120575987057, -1.4048494338234274, -1.3470353050263304, 1.0283499935773348, -1.0606584350011485]
#      ---------------------------------------------
#      Fidelity at Step 256: 0.9999884429061434
#      Parameters at Step 256: [0.5937387657258045, 0.005466537421689071, 1.3289761481135578, 1.782106391856814, 0.0024479747093570704, -0.8450064258141284, -1.4051979002680197, -1.3472456516461127, 1.0285697676625802, -1.0605701923627995]
#      ---------------------------------------------
#      Fidelity at Step 257: 0.9999891831757101
#      Parameters at Step 257: [0.5937851969295652, 0.005196756021029536, 1.3294245827802798, 1.7823535410337858, 0.0024088866250306104, -0.8450219960253053, -1.4053849083966263, -1.3474686685044446, 1.0287637259980984, -1.0606193114296032]
#      ---------------------------------------------
#      Fidelity at Step 258: 0.9999883857699693
#      Parameters at Step 258: [0.5940953819353095, 0.004791092327584531, 1.3305076450871693, 1.7826956874110953, 0.0022010109558538555, -0.8450446134557336, -1.4053616165625418, -1.3476661487748742, 1.0289057839928066, -1.060837114189176]
#      ---------------------------------------------
#      Fidelity at Step 259: 0.9999904796568865
#      Parameters at Step 259: [0.5944964245890103, 0.004190556706255601, 1.3309402564751298, 1.7829042001339033, 0.0016739683970465424, -0.8446549383648442, -1.4050618203636571, -1.347832416441113, 1.0289875015471048, -1.0613979396555118]
#      ---------------------------------------------
#      Fidelity at Step 260: 0.9999895446462769
#      Parameters at Step 260: [0.5948290412691644, 0.003611728925779746, 1.3310376718637225, 1.7830523350642844, 0.0011174288273581834, -0.844163954881152, -1.4047616223944055, -1.348002212413368, 1.0290679993304543, -1.0619883701766766]
#      ---------------------------------------------
#      Fidelity at Step 261: 0.9999875391094621
#      Parameters at Step 261: [0.5951669123789316, 0.0031310396097699476, 1.331789961880425, 1.7833168983733616, 0.000681157510524022, -0.8439073978707629, -1.4045768433706556, -1.3482025491651533, 1.0291692895659659, -1.0624320524464426]
#      ---------------------------------------------
#      Fidelity at Step 262: 0.9999906306246412
#      Parameters at Step 262: [0.5952821276050037, 0.002775226252607156, 1.3320799556119916, 1.7834941541314508, 0.0003834410190620781, -0.8436445142949486, -1.404537104631228, -1.3484310545032105, 1.0292994237971727, -1.062717230258624]
#      ---------------------------------------------
#      Fidelity at Step 263: 0.9999918475423802
#      Parameters at Step 263: [0.5955068278307881, 0.002537066041621898, 1.333574274178206, 1.783890447565905, 0.00023900989459528507, -0.8437857783409928, -1.4046369850364349, -1.3486683103850958, 1.0294464368228422, -1.0628231123748586]
#      ---------------------------------------------
#      Fidelity at Step 264: 0.9999912431486708
#      Parameters at Step 264: [0.5955932148174262, 0.002386430386773306, 1.3347469769816265, 1.7842223336425767, 0.00016586603537737705, -0.8439075571128305, -1.4048322277067897, -1.3489091954001837, 1.0296144174254342, -1.0628521286989452]
#      ---------------------------------------------
#      Fidelity at Step 265: 0.9999872555933262
#      Parameters at Step 265: [0.5956772571815188, 0.002257828793900233, 1.33578997508499, 1.7845332412487767, 5.7699104043846676e-05, -0.843988917135868, -1.4050290899242444, -1.3491239263836743, 1.029776966527158, -1.0629292663204126]
#      ---------------------------------------------
#      Fidelity at Step 266: 0.9999902967524766
#      Parameters at Step 266: [0.5957507501519791, 0.002068783031136052, 1.3361309650648827, 1.7847050976583787, -0.00020519692950900033, -0.8438051992942112, -1.4051204013078076, -1.3492778991144794, 1.029910772991523, -1.063194799291699]
#      ---------------------------------------------
#      Fidelity at Step 267: 0.9999900150551528
#      Parameters at Step 267: [0.5960916717608045, 0.001792095457352826, 1.3372109116920048, 1.7849799192210827, -0.0005592198163297445, -0.8437340950967995, -1.405069258993, -1.349362115887662, 1.0299998148528058, -1.0635698848235604]
#      ---------------------------------------------
#      Fidelity at Step 268: 0.9999901632430063
#      Parameters at Step 268: [0.5964938257348139, 0.0014818649446677085, 1.3381974555803442, 1.7852420652228995, -0.0009791586947237767, -0.8436098082259162, -1.4049470696253095, -1.3494147180174358, 1.0300676204942645, -1.0640370491853084]
#      ---------------------------------------------
#      Fidelity at Step 269: 0.9999903697305814
#      Parameters at Step 269: [0.5970118277779763, 0.0011219116396615045, 1.3393961902403144, 1.7855517098796023, -0.0014572778116721133, -0.8435008592080591, -1.4047356296715312, -1.34945268329791, 1.030103626036143, -1.064599942972532]
#      ---------------------------------------------
#      Fidelity at Step 270: 0.9999899367559645
#      Parameters at Step 270: [0.5975015615937334, 0.0007479805980981278, 1.340286482112039, 1.785845478008108, -0.001971802490065613, -0.8433015328756424, -1.4044857384421614, -1.349519912456144, 1.030121254199464, -1.0652548452467254]
#      ---------------------------------------------
#      Fidelity at Step 271: 0.9999900678666692
#      Parameters at Step 271: [0.5977691910068161, 0.0004258803750859548, 1.3403369377028158, 1.7860300864203622, -0.00241999652252484, -0.8429562359241124, -1.4042866622671257, -1.3496564051307558, 1.0301453455775575, -1.0658908588260874]
#      ---------------------------------------------
#      Fidelity at Step 272: 0.9999874390289892
#      Parameters at Step 272: [0.5978043725676705, 0.00022655979324939457, 1.3399444553783821, 1.7861742417428905, -0.0026566307691544863, -0.8426747036087735, -1.4042431189145252, -1.3498703592182126, 1.030191903838473, -1.066332331123344]
#      ---------------------------------------------
#      Fidelity at Step 273: 0.9999852404955787
#      Parameters at Step 273: [0.5976018832312114, 0.00019732576339516204, 1.3394361789334397, 1.7863019114132654, -0.002520678451761485, -0.84265540315487, -1.404425981261165, -1.3501473692207824, 1.0302708567239884, -1.0663787728334129]
#      ---------------------------------------------
#      Fidelity at Step 274: 0.999991202216739
#      Parameters at Step 274: [0.5970193753976241, 0.00039122226824866557, 1.3383976223932375, 1.7862643127485072, -0.0018034291770251108, -0.8429631976270391, -1.404918488619218, -1.350438857017038, 1.0303917093774562, -1.0657749577970645]
#      ---------------------------------------------
#      Fidelity at Step 275: 0.9999868730098144
#      Parameters at Step 275: [0.5964002312016697, 0.0006386686899270823, 1.3372686174563626, 1.7861038705095236, -0.0008960319344269613, -0.843421216990226, -1.4054938764832425, -1.3506018283136876, 1.0304883342850093, -1.0649185421353626]
#      ---------------------------------------------
#      Fidelity at Step 276: 0.9999871021993375
#      Parameters at Step 276: [0.5959968024125873, 0.0008543240692973162, 1.3365676350952018, 1.78585275692967, 1.8806409563042127e-05, -0.8440244978214125, -1.406039485419777, -1.3505103596775951, 1.030531301231813, -1.0639819437643518]
#      ---------------------------------------------
#      Fidelity at Step 277: 0.9999881185344416
#      Parameters at Step 277: [0.5957871447329421, 0.0009511001755770216, 1.3354265535238314, 1.785356367154387, 0.0006376402592192541, -0.8443413411173908, -1.4064349034510109, -1.350115097279099, 1.0304975653635302, -1.0633064271646853]
#      ---------------------------------------------
#      Fidelity at Step 278: 0.9999906556766786
#      Parameters at Step 278: [0.5956367530266714, 0.0009847656361054722, 1.3336692706093733, 1.7846757019204984, 0.0009634945439259536, -0.8443164401766956, -1.4067434739871398, -1.3495484340103792, 1.0304263826065396, -1.062916102860791]
#      ---------------------------------------------
#      Fidelity at Step 279: 0.9999894442481566
#      Parameters at Step 279: [0.5955525976517577, 0.001028907117684992, 1.3321084157952712, 1.78401261978164, 0.001164747539451552, -0.8442449016140025, -1.4070533814616768, -1.3489321291968648, 1.0303594700642262, -1.0626365307718746]
#      ---------------------------------------------
#      Fidelity at Step 280: 0.999989762379587
#      Parameters at Step 280: [0.5955296466388412, 0.0011362950193639534, 1.331190926052615, 1.783475476175838, 0.001338438536987609, -0.8442954311854832, -1.407434080324662, -1.3483249178832246, 1.0303222541973678, -1.0623605557760991]
#      ---------------------------------------------
#      Fidelity at Step 281: 0.9999865518287322
#      Parameters at Step 281: [0.5955545184217964, 0.0013231465148655075, 1.3311625356047956, 1.78314061313324, 0.0015410837043886285, -0.8445508065681544, -1.4078983402622387, -1.3477988954190034, 1.0303360583279675, -1.0620348186940016]
#      ---------------------------------------------
#      Fidelity at Step 282: 0.9999885658815401
#      Parameters at Step 282: [0.59561242446879, 0.0015227890917823526, 1.3317033960996465, 1.7829985284014171, 0.0016868396520360817, -0.8448606401220743, -1.4083455925713972, -1.3474149162023767, 1.0303933476864877, -1.0617804808654006]
#      ---------------------------------------------
#      Fidelity at Step 283: 0.9999886200387803
#      Parameters at Step 283: [0.5957255375318595, 0.0016497018583908577, 1.3324425509618274, 1.783020973690081, 0.001659143486992421, -0.8450490961123054, -1.40864684396356, -1.347207439471713, 1.030483142356627, -1.0617566525637303]
#      ---------------------------------------------
#      Fidelity at Step 284: 0.999989658576399
#      Parameters at Step 284: [0.5959609043974802, 0.0015499456597282543, 1.3331254019793428, 1.7832049882435026, 0.0012922693222957348, -0.8449006011213077, -1.4085712839597364, -1.3472519572169899, 1.030580980547529, -1.0621844020557074]
#      ---------------------------------------------
#      Fidelity at Step 285: 0.9999894649394983
#      Parameters at Step 285: [0.5960238507252131, 0.0013189631337330305, 1.332539860511835, 1.783301927654444, 0.0007433866489563661, -0.8442854509445332, -1.4082711162144352, -1.347499484338802, 1.0306917016644268, -1.0628767596250766]
#      ---------------------------------------------
#      Fidelity at Step 286: 0.9999849933233228
#      Parameters at Step 286: [0.5960627205395083, 0.0010609043148659702, 1.3322430338627351, 1.7835243539681294, 0.00034286537921595847, -0.8438166939869268, -1.4079035520248064, -1.3479187962707286, 1.0308251683085796, -1.0634261212436664]
#      ---------------------------------------------
#      Fidelity at Step 287: 0.9999879639253013
#      Parameters at Step 287: [0.5956899682731177, 0.0009154963726379061, 1.3311635607559102, 1.7836488028761535, 0.0004225855939777025, -0.8434963287575707, -1.407661671454534, -1.3485458555802754, 1.0310382483211102, -1.0634353812611523]
#      ---------------------------------------------
#      Fidelity at Step 288: 0.9999861925231072
#      Parameters at Step 288: [0.5951550758551492, 0.0009283519295401207, 1.3304978498841862, 1.7837725159484372, 0.0010039543254848613, -0.8436443722100078, -1.4076374485189134, -1.3491935214571396, 1.0313008332555351, -1.0628211566014651]
#      ---------------------------------------------
#      Fidelity at Step 289: 0.9999899160547191
#      Parameters at Step 289: [0.5943319215746347, 0.0011024983291010803, 1.3290715959896324, 1.7835802504401053, 0.001939345423603396, -0.843900829201413, -1.407855162572034, -1.3496824841474095, 1.0315948437820224, -1.0616977278993973]
#      ---------------------------------------------
#      Fidelity at Step 290: 0.9999899406341382
#      Parameters at Step 290: [0.593879437917849, 0.0013024934116531413, 1.3292359330494528, 1.7834789968444222, 0.0028823150174223234, -0.8445658740971359, -1.408146953358147, -1.349884604304381, 1.0318369775129694, -1.060448351856046]
#      ---------------------------------------------
#      Fidelity at Step 291: 0.999987701120682
#      Parameters at Step 291: [0.5938227296665606, 0.001395519180852045, 1.3297453114468043, 1.7831820333246586, 0.0033488197044209502, -0.8449975130724344, -1.408343231895112, -1.3496242833250376, 1.0319721784111542, -1.0595916756870527]
#      ---------------------------------------------
#      Fidelity at Step 292: 0.9999895591448354
#      Parameters at Step 292: [0.594030472531836, 0.001275204988229672, 1.3289723092333423, 1.782426923070065, 0.002887692115341309, -0.8444637224643667, -1.4082817262878289, -1.3488816570547606, 1.031993478449861, -1.05965962408314]
#      ---------------------------------------------
#      Fidelity at Step 293: 0.9999892316742227
#      Parameters at Step 293: [0.5944107428465928, 0.0011561190186811987, 1.3287859463486562, 1.7817881428685853, 0.002200021375265551, -0.8439060424703603, -1.408222823638352, -1.3480851698187917, 1.0320067372168447, -1.0599710816436352]
#      ---------------------------------------------
#      Fidelity at Step 294: 0.9999899316350848
#      Parameters at Step 294: [0.5947992758125891, 0.0011229866430632221, 1.3293308466414726, 1.7813714966253529, 0.001521404070670901, -0.8435042668478548, -1.4082559739400768, -1.3474186267317105, 1.0320785139540976, -1.0602894796597422]
#      ---------------------------------------------
#      Fidelity at Step 295: 0.9999907505093709
#      Parameters at Step 295: [0.5951524351616371, 0.0011663624769124455, 1.3305982924157018, 1.7812371317727713, 0.000988489193023588, -0.8433565124020631, -1.4083561337808042, -1.3469875194740446, 1.0322186618691107, -1.0605020314513662]
#      ---------------------------------------------
#      Fidelity at Step 296: 0.9999881552104773
#      Parameters at Step 296: [0.5953267513068672, 0.0012511861842933924, 1.3319337927383572, 1.7813510571695326, 0.0006205158363415861, -0.8433069686202627, -1.4084495824717296, -1.3469315332264586, 1.032453125529733, -1.0606305908474725]
#      ---------------------------------------------
#      Fidelity at Step 297: 0.9999886313248761
#      Parameters at Step 297: [0.5952913194747004, 0.001267158715709055, 1.332815657793257, 1.7816823291053976, 0.00036993673278882546, -0.8431800027361622, -1.4083585518645125, -1.3473569073207656, 1.0327821979296068, -1.0607704079399491]
#      ---------------------------------------------
#      Fidelity at Step 298: 0.999989473771392
#      Parameters at Step 298: [0.595385647586571, 0.0011875072416154175, 1.3342622514353542, 1.7822505911816997, 0.00014488435321944392, -0.8431903850464094, -1.408104846362179, -1.34797102144586, 1.0331094484959207, -1.0609579558064077]
#      ---------------------------------------------
#      Fidelity at Step 299: 0.9999884925835485
#      Parameters at Step 299: [0.595518478464883, 0.0009458696659425904, 1.3352816336854778, 1.7828634413147242, -0.0001669195299291341, -0.8430304233164108, -1.4076043914415428, -1.34873729802631, 1.0334059085675407, -1.0613246722884933]
#      ---------------------------------------------
#

######################################################################
# With the learned parameters, we construct a visual representation
# of the Hamiltonian to which they correspond and compare it to the 
# target Hamiltonian, and the initial guessed Hamiltonian:
#

new_ham_matrix = create_hamiltonian_matrix(
    qubit_number, nx.complete_graph(qubit_number), [qgrnn_params[0:6], qgrnn_params[6:10]]
)

init_ham = create_hamiltonian_matrix(
    qubit_number, nx.complete_graph(qubit_number), [init[0:6], init[6:10]]
) 

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 6))

axes[0].matshow(ham_matrix, vmin=-7, vmax=7, cmap='hot')
axes[0].set_title("Target Hamiltonian", y=1.13)

axes[1].matshow(init_ham, vmin=-7, vmax=7, cmap='hot')
axes[1].set_title("Initial Guessed Hamiltonian", y=1.13)

axes[2].matshow(new_ham_matrix, vmin=-7, vmax=7, cmap='hot')
axes[2].set_title("Learned Hamiltonian", y=1.13)

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

qgrnn_params = list(qgrnn_params)

zero_weights = [qgrnn_params[1], qgrnn_params[4]]

del qgrnn_params[1]
del qgrnn_params[3]

######################################################################
# Then, we print all of the weights:
#

target_params = matrix_params[0] + matrix_params[1]

print(f"Target parameters: {target_params}")
print(f"Learned parameters: {qgrnn_params}")
print(f"Non-Existing Edge Parameters: {zero_weights}")


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
# 
#  .. code-block:: none
#
#      Target parameters: [0.56, 1.24, 1.67, -0.79, -1.44, -1.43, 1.18, -0.93]
#      Learned parameters: [0.5958460805811168, 1.33650946569539, 1.7835767870051, -0.8428389553658877, -1.4068803387384392, -1.3495612547209628, 1.03363436004853, -1.0618808945949776]
#      Non-Existing Edge Parameters: [0.000540998047389223, -0.0005928152653526596]
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
