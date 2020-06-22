"""
The Quantum Graph Recurrrent Neural Network
===========================================

.. meta::
    :property="og:description": Using a quantum graph recurrrent neural network to learn quantum dynamics.
    :property="og:image": https://pennylane.ai/qml/_images/qgrnn_thumbnail.png

*Author: Jack Ceroni*

"""

######################################################################
# In this demonstration, we investigate quantum graph
# recurrent neural networks (QGRNN), which are the quantum analogue of a
# classical graph recurrent neural network and a subclass of the more 
# general quantum graph
# neural network ansatz (`QGNN <https://arxiv.org/abs/1909.12264>`__).

######################################################################
# The Idea
# --------
#


###################################################################### 
# A graph is defined as a set of **nodes**, along with a set of 
# **edges**, which represent interactions or relationships between nodes. 
# Sometimes, we like to encode information into graphs by assigning numbers 
# to nodes and edges, which we call **weights**.
# It is usually convenient to think of a graph visually:
#
# .. image:: ../demonstrations/qgrnn/graph.png
#     :width: 70%
#     :align: center
# 
# In recent years, the concept of a 
# `graph neural network <https://arxiv.org/abs/1812.08434>`__ (GNN) has been
# receving a lot of attention from the machine learning community. 
# A GNN seeks
# to learn a representation (a mapping of data into a
# lower-dimensional vector space) of a given graph with features assigned
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
# A well known example of a quadratic Hamiltonian is the transverse 
# field Ising model, which is defined as
#
# .. math:: \hat{H}_{\text{Ising}}(\boldsymbol\theta) \ = \ \displaystyle\sum_{(i, j) \in E} \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \ \displaystyle\sum_{i} X_{i},
#
# where :math:`\boldsymbol\theta \ = \ \{\theta^{(1)}, \ \theta^{(2)}\}`.
# In this Hamiltonian, the set :math:`E` that determines which pairs of qubits 
# have :math:`ZZ` interactions is exactly the set of edges for some graph. With 
# the qubits as nodes we call this graph the *interaction graph*.
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
# Thus, we have a clean way of taking graph-theoretic Hamiltonians and turning 
# them into unitaries (quantum circuits) that preserve the same correspondance to a graph.
# In the case of the Ising Hamiltonian, we have:
#
# .. math:: U_{\text{Ising}} \ = \ e^{-it \hat{H}_{\text{Ising}} (\boldsymbol\theta)} \ = \ \exp \Big[ -it \Big( \displaystyle\sum_{(i, j) \in E} \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \ \displaystyle\sum_{i} X_{i} \Big) \Big]
#
# In general, this kind of unitary is very difficult to implement on a quantum computer, 
# but luckily, we can approximate it using the `Trotter-Suzuki decomposition 
# <https://en.wikipedia.org/wiki/Time-evolving_block_decimation#The_Suzuki-Trotter_expansion>`__:
#
# .. math:: \exp \Big[ -it \Big( \displaystyle\sum_{(i, j) \in E} \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \ \displaystyle\sum_{i} X_{i} \Big) \Big]
#            \ \approx \ \displaystyle\prod_{k \ = \ 1}^{t / \Delta} \Bigg[ \displaystyle\prod_{j \ = \ 1}^{Q} e^{-i \Delta \hat{H}_{\text{Ising}}^{j}(\boldsymbol\theta)} \Bigg] 
#
# where :math:`\hat{H}_{\text{Ising}}^{j}(\boldsymbol\theta)` is the :math:`j`-th term of the 
# Ising Hamiltonian and :math:`\Delta` is some small number.
#
# The circuit at which we have arrived is a specific instance of the **Quantum Graph 
# Recurrent Neural Network**, which in general is defined as a variational ansatz of
# the form
#
# .. math:: U_{H}(\boldsymbol\mu, \ \boldsymbol\gamma) \ = \ \displaystyle\prod_{i \ = \ 1}^{P} \Bigg[ \displaystyle\prod_{j \ = \ 1}^{Q} e^{-i \gamma_j H^{j}(\boldsymbol\mu)} \Bigg],
#
# for some parametrized Hamiltonian, :math:`H(\boldsymbol\mu)`, of 
# quadratic order.

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
# low-energy state with respect to the target Hamiltonian, :math:`|\psi_0\rangle`. In addition, 
# we have access to a collection of time-evolved states, 
# :math:`\{ |\psi(t_1)\rangle, \ |\psi(t_2)\rangle, \ ..., \ |\psi(t_N)\rangle \}`, defined by:
#
# .. math:: |\psi(t_k)\rangle \ = \ e^{-i t_k \hat{H}_{\text{Ising}}(\boldsymbol\alpha)} |\psi_0\rangle
#
# We call the low-energy states and the collection of time-evolved states *quantum data*.
# From here, we randomly pick a given number of time-evolved states
# from our collection. For some state that we choose, which is
# evolved to some time :math:`t_k`, we compare it
# to
#
# .. math:: U_{\hat{H}_{\text{Ising}}}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle \ \approx \ e^{-i t_k \hat{H}_{\text{Ising}}(\boldsymbol\mu)} |\psi_0\rangle
#
# by feeding one of the copies of :math:`|\psi_0\rangle` into a quantum circuit 
# with the QGRNN ansatz, with some "guessed" set of parameters :math:`\boldsymbol\mu`
# and a guessed interaction graph, :math:`G'`.
# We then use a classical optimizer to maximize the average 
# "similarity" between the time-evolved states and the states prepared
# with the QGRNN.
#
# As the QGRNN states becomes more "similar" to 
# each time-evolved state for each sampled time, it follows that 
# :math:`\boldsymbol\mu \ \rightarrow \ \boldsymbol\alpha`
# and we are able to learn the unknow parameters of the Hamiltonian.
#
# .. figure:: ../demonstrations/qgrnn/qgrnn3.png
#     :width: 90%
#     :align: center
#
#     A visual representation of one execution of the QGRNN for one piece of quantum data
#


######################################################################
# Learning an Ising Model with the QGRNN
# ---------------------------------------
#


######################################################################
# We now attempt to use the QGRNN to learn the parameters corresponding
# to some arbitrary transverse field Ising model Hamiltonian.
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


######################################################################
# We can then define some fixed values that are used throughout
# the simulation.
#


qubit_number = 4
qubits = range(qubit_number)


######################################################################
# In order to use the QGRNN, we need access to quantum data. In this 
# simulation, we don't have quantum data readily available to pass into 
# the QGRNN, so we have to generate it ourselves. To do this, we must 
# have knowledge of the target interaction graph and the target Hamiltonian.
#
# We first define the target interaction graph 
# of the Ising Hamiltonian to be the cycle graph:
#


ising_graph = nx.cycle_graph(qubit_number)

nx.draw(ising_graph)
print("Edges: {}" .format(ising_graph.edges))


######################################################################
# We can then initialize the “unknown” target parameters that describe the
# target Hamiltonian, :math:`\boldsymbol\alpha \ = \ \{\alpha^{(1)}, \ \alpha^{(2)}\}`.
# Recall from the introduction that we have defined our parametrized 
# Ising Hamiltonian to be of the form:
#
# .. math:: \hat{H}_{\text{Ising}}(\boldsymbol\theta) \ = \ \displaystyle\sum_{(i, j) \in E} \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \ \displaystyle\sum_{i} X_{i},
#
# where :math:`E` is the set of edges in the interaction graph, and
# :math:`X_i` and :math:`Z_i` are the Pauli-X and Pauli-Z on the 
# :math:`i`-th qubit. 
#
# For this tutorial, we choose the target parameters manually.
#


matrix_params = [[-0.3, 0.58, -0.77, 0.83], [0.7, 0.82, 0.17, 0.14]]


######################################################################
# In theory, these parameters can
# be any value we want, provided they are reasonbly small enough that the QGRNN can reach them 
# in a tractable number of optimization steps.
# The first list represents the :math:`ZZ` interaction parameters and 
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
        matrix = np.add(matrix, params[0][count] * m)

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
        matrix = np.add(matrix, np.add(params[1][i] * m1, m2))

    return matrix

# Prints a visual representation of the Hamiltonian matrix
ham_matrix = create_hamiltonian_matrix(qubit_number, ising_graph, matrix_params)
plt.matshow(ham_matrix)
plt.show()


######################################################################
# Preparing Quantum Data
# ^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# The collection of quantum data needed to run the QGRNN has two componenets: 
# copies of a low-energy state, and a collection of time-evolved states, each of which are
# simply the low-energy state evolved to different times. 
# For the target Hamiltonian we 
# defined, the following is a low-energy state 
#

low_energy_state = np.array([-0.02086666+0.00920016j, -0.00379192-0.00859852j,  0.06594626-0.02907913j,
  0.0119852 +0.02717445j,  0.07633593-0.03366391j,  0.01387486+0.03145572j,
 -0.24124938+0.10640219j, -0.04385454-0.09941154j,  0.06397641-0.02820407j,
  0.01162454+0.02636273j, -0.20218878+0.08914518j, -0.03674192-0.08331586j,
 -0.23404312+0.10320031j, -0.04253486-0.09644206j,  0.73966164-0.32618732j,
  0.13444079+0.30479209j])


######################################################################
# We can verify that this is a low-energy
# state by numerically finding the lowest eigenvalue of the Hamiltonian
# matrix and comparing it to the energy expectation of the low-energy state:
#

# Finds the energy expectation

def expectation_value(vector, matrix):
    return np.inner(np.conj(vector), (matrix @ vector))

energy_exp = np.inner(np.conj(low_energy_state), (ham_matrix @ low_energy_state))
print("Energy Expectation: {}".format(energy_exp))

def ground_state_energy(matrix):

    # Finds the eigenstates of the matrix
    val = np.linalg.eig(matrix)[0]

    # Returns the minimum eigenvalue
    return min(val)

ground_state = ground_state_energy(ham_matrix)
print("Ground State Energy: {}".format(ground_state))


######################################################################
# This shows us that we have in fact found a low-energy, non-ground state, 
# as the energy expectation is slightly greater than the energy of the true ground
# state. This, however, is only half of the information we need. We also require 
# a collection of time-evolved states. 
# Evolving a state forward in time is fairly straightforward: all we 
# have to do is multiply the initial state by a time-evolution unitary. We define 
# this operation as a custom gate in PennyLane:
#


def state_evolve(hamiltonian, qubits, time):

    U = scipy.linalg.expm(-1j* hamiltonian * time)
    qml.QubitUnitary(U, wires=qubits)


######################################################################
# We don't actually generate this time-evolved quantum data quite yet, 
# but we now have all the pieces required for its preparation.
#


######################################################################
# Learning the Hamiltonian
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# With the quantum data defined, we are able to construct the QGRNN and
# learn the target Hamiltonian. As was discussed above, we wish to use the
# QGRNN to approximate time-evolution of the target Hamiltonian, with
# guessed parameters. It follows that we let each of the three
# :math:`\hat{H}_{q}(\boldsymbol\mu)` used in the QGRNN ansatz be the
# collections of :math:`ZZ`, :math:`Z`, and :math:`X` terms from the Ising
# Hamiltonian. This gives us:
#


def qgrnn_layer(param1, param2, qubits, graph, trotter):

    # Applies a layer of coupling gates (based on a graph)
    for count, i in enumerate(graph.edges):
        qml.MultiRZ(2 * param1[count] * trotter, wires=[i[0], i[1]])

    # Applies a layer of RZ gates
    for count, i in enumerate(qubits):
        qml.RZ(2 * param2[count] * trotter, wires=i)

    # Applies a layer of RX gates
    for i in qubits:
        qml.RX(2 * trotter, wires=i)


######################################################################
# As was mentioned in the first section, the QGRNN has two qubit
# registers. In one register, some piece of quantum data
# :math:`|\psi(t)\rangle` is prepared and in the other we have
# :math:`U_{H}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle`. We need a
# way to measure the similarity between these states. 
# One way that we can do this is by using the fidelity, which is
# simply the modulus squared of the inner product between the states,
# :math:`| \langle \psi(t) | U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle |^2`.
# To calculate this value, we utilize a `SWAP
# test <https://en.wikipedia.org/wiki/Swap_test>`__ between the registers:
#


def swap_test(control, register1, register2):

    qml.Hadamard(wires=control)
    for i in range(0, len(register1)):
        qml.CSWAP(wires=[int(control), register1[i], register2[i]])
    qml.Hadamard(wires=control)


######################################################################
# Before creating the full QGRNN and the cost function, we
# define a few more fixed values. Among these fixed values is a "guessed"
# interaction graph, which we define to be the complete graph. This choice 
# is motivated by the fact that any target interaction graph will be a subgraph 
# of this initial guess. 
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

print("Edges: " + str(new_ising_graph.edges))
nx.draw(new_ising_graph)


######################################################################
# Part of the idea behind the QGRNN is that
# we don’t know the interaction graph, and it has to be learned. In this case, the graph  
# is learned **automatically** as the target parameters are learned. The
# :math:`\boldsymbol\mu` parameters that correspond to edges that don't exist in 
# the target graph will simply approach :math:`0`.
#
# With this done, we implement the QGRNN circuit for some given time value:
#


def qgrnn(params1, params2, time=None):

    # Prepares the low energy state in the two qubit registers
    qml.QubitStateVector(np.kron(low_energy_state, low_energy_state), wires=reg1+reg2)

    # Evolves the first qubit register with the time-evolution circuit, to prepare a piece of quantum data
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
# Notice how the value returned from the circuit is
# :math:`\langle Z \rangle`, with respect to the ancilla qubit. This is
# because after performing a SWAP test between two states 
# :math:`|\psi\rangle` and :math:`|\phi\rangle`, it is tue that
# :math:`\langle Z \rangle \ = \ |\langle \psi | \phi \rangle|^2`.
#
# We have the full QGRNN circuit, but we still need to define a cost
# function to minimize. We know that
# :math:`| \langle \psi(t) | U_{H}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle |^2`
# approaches :math:`1` as the states become more similar, thus we choose
# to minimize the quantity
# :math:`1 \ - \ | \langle \psi(t) | U_{H}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle |^2`.
# Since we are interested in calculating this value for many different
# pieces of quantum data, the final cost function is the **average
# infidelity** between registers:
#
# .. math:: \mathcal{L}(\boldsymbol\mu, \ \Delta) \ = \ 1 \ - \ \frac{1}{N} \displaystyle\sum_{t \ = \ 1}^{N} | \langle \psi(t) | \ U_{H}(\boldsymbol\mu, \ \Delta) \ |\psi_0\rangle |^2,
#
# where we use :math:`N` pieces of quantum data.
#
# Before creating the cost function, we must define a few more fixed
# variables, the device, and the QNode:
#

N = 15  # The number of pieces of quantum data that are used for each step
max_time = 0.1  # The maximum value of time that can be used for quantum data

# Defines the new device

qgrnn_dev = qml.device("default.qubit", wires=2 * qubit_number + 1)

# Defines the new QNode

qnode = qml.QNode(qgrnn, qgrnn_dev)


######################################################################
# We then define the infidelity cost function:
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
        total_cost += 1 - result

    # Prints the value of the cost function
    if iterations % 5 == 0:
        print("Cost at Step " + str(iterations) + ": " + str((1 - total_cost / N)._value))
        print("Parameters at Step " + str(iterations) + ": " + str(params._value))
        print("---------------------------------------------")

    iterations += 1

    return total_cost / N


######################################################################
# The last step is to define and execute the optimizer. We use Adam,
# with a step-size of :math:`0.3`:
#


iterations = 0

optimizer = qml.AdamOptimizer(stepsize=0.3)
steps = 100
qgrnn_params = list([np.random.randint(-20, 20) / 50 for i in range(0, 10)])

# Executes the optimization method

for i in range(0, steps):
    qgrnn_params = optimizer.step(cost_function, qgrnn_params)

print("Final Parameters: {}".format(qgrnn_params))


######################################################################
# With the learned parameters, we construct a visual representation 
# of the Hamiltonian:
#

new_ham_matrix = create_hamiltonian_matrix(
    qubit_number, nx.complete_graph(qubit_number), [qgrnn_params[0:6], qgrnn_params[6:10]]
)
plt.matshow(new_ham_matrix)
plt.show()


######################################################################
# In addition, we can construct a visual representation of the target 
# and learned parameters, to assess their similarity.
#


# Creates the colour plot function

def create_colour_plot(data):

    array = np.array(data)
    plt.matshow(array)
    plt.colorbar()
    plt.show()

# Inserts 0s where there is no edge present in target parameters

target_params = matrix_params[0] + matrix_params[1]
target_params.insert(1, 0)
target_params.insert(4, 0)

# Prints the colour plot of the parameters

create_colour_plot([target_params, qgrnn_params])


######################################################################
# The top row is the target values, while the bottom row is the learned 
# ones.
# The similarity of colours indicates that the parameters are very
# similar, which we can verify by looking at their exact values:
#

print("Target parameters: {}".format(target_params))
print("Learned parameters: {}".format(qgrnn_params))


######################################################################
# References
# ===========
#
# 1. Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., &
#    Hidary, J. (2019). Quantum Graph Neural Networks. arXiv preprint
#    `arXiv:1909.12264 <https://arxiv.org/abs/1909.12264>`__.
#
