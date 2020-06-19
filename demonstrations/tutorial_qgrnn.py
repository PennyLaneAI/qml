"""
The Quantum Graph Recurrrent Neural Network
===========================================

.. meta::
    :property="og:description": Using a quantum graph recurrrent neural network to learn quantum dynamics.
    :property="og:image": https://pennylane.ai/qml/_images/qgrnn_thumbnail.png

*Author: Jack Ceroni*

"""

######################################################################
# In this Notebook, we investigate the idea of a quantum graph
# recurrent neural network (QGRNN), which is the quantum analogue of a
# classical graph RNN and a subclass of the more general quantum graph
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
# `graph neural network <https://arxiv.org/abs/1812.08434>`__ has been
# receving a lot of attention from the machine learning community. 
# A **graph neural network** seeks
# to learn a representation (a mapping of data into a
# lower-dimensional vector space) of a given graph with features assigned
# to nodes and edges.  Each of the vectors in the learned
# representation preserves not only the features, but also the overall 
# topology of the graph, i.e., which nodes are connected by edges. The 
# quantum graph neural network attempts to do something similar, but for
# features that are quantum-mechanical; for instance, a
# collection of quantum states.
#


######################################################################
# Consider the class of qubit Hamiltonians that are **quadratic**, meaning that 
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
# have :math:`ZZ` interactions is exactly the set of edges for some graph.
# The :math:`\theta^{(1)}` parameters correspond to the edge weights and
# the :math:`\theta^{(2)}` 
# parameters correspond to weights on the nodes.
#


######################################################################
# This result implies that we can think about **quantum circuits** with 
# graph-theoretic properties. Recall that the time-evolution operator 
# with respect to some Hamiltonian :math:`H` is defined as:
#
# .. math:: U \ = \ e^{-it H}.
#
# Thus, we have a clean way of taking our graph-theoretic Hamiltonians and turning 
# them into unitaries (quantum circuits) that possess the same correspondance to a graph.
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
# Ising Hamiltonian. :math:`\Delta` is some small number.
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
# Continuing on with the Ising model example, let's imagine we have some system
# governed by :math:`\hat{H}_{\text{Ising}}(\boldsymbol\alpha)` for an unknown set of
# target parameters, 
# :math:`\boldsymbol\alpha`. Let's also suppose we have access to a bunch of copies of some 
# low-energy state with respect to the target Hamiltonian, :math:`|\psi_0\rangle`. In addition, 
# we have access to a collection of time evolved states, 
# :math:`\{ |\psi(t_1)\rangle, \ |\psi(t_2)\rangle, \ ..., \ |\psi(t_N)\rangle \}`, defined by:
#
# .. math:: |\psi(t_k)\rangle \ = \ e^{-i t_k \hat{H}_{\text{Ising}}(\boldsymbol\alpha)} |\psi_0\rangle
#
# From here, we randomly pick a given number of time-evolved states
# from our collection. For some state that we choose, which is
# evolved to some arbitrary time :math:`t_k`, we compare it
# to
#
# .. math:: U_{\hat{H}_{\text{Ising}}}(\boldsymbol\mu, \ \Delta) |\psi_0\rangle \ \approx \ e^{-i t_k \hat{H}_{\text{Ising}}(\boldsymbol\mu)} |\psi_0\rangle
#
# by feeding one of the copies of :math:`|\psi_0\rangle` into a quantum circuit 
# with the QGRNN ansatz, with some "guessed" set of parameters :math:`\boldsymbol\mu`.
# We then use a classical optimizer to maximize the average 
# "similarity" between the time-evolved states and the states outputted
# from the QGRNN.
#
# As the QGRNN states becomes more "similar" to 
# each time-evolved state for each sampled time, it follows that 
# :math:`\boldsymbol\mu \ \rightarrow \ \boldsymbol\alpha`
# and we are able to learn the unknow parameters of the Hamiltonian.
#
# .. image:: ../demonstrations/qgrnn/qgrnn2.png
#     :width: 90%
#     :align: center
#


######################################################################
# Learning an Ising Model with the QGRNN
# ---------------------------------------
#


######################################################################
# In this demonstration, we attempt to learn the parameters corresponding
# to some randomly-initialized transverse field Ising model Hamiltonian,
# using the QGRNN.
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
# have *a priori* knowledge of the target Hamiltonian and the interaction 
# graph. We choose the Hamiltonian to be a transerve field Ising model 
# of the form:
#
# .. math:: \hat{H}(\boldsymbol\theta) \ = \ \displaystyle\sum_{(i, j) \in E} \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \ \displaystyle\sum_{i} X_{i},
#
# where :math:`E` is the set of edges in the interaction graph.
# :math:`X_i` and :math:`Z_i` are the Pauli-X and Pauli-Z on the 
# :math:`i`-th qubit. We then define the target interaction graph 
# of the Hamiltonian to be the cycle graph:
#


ising_graph = nx.cycle_graph(qubit_number)

nx.draw(ising_graph)
print("Edges: {}" .format(ising_graph.edges))


######################################################################
# We then can initialize the “unknown” target parameters that describe the
# Hamiltonian, :math:`\alpha \ = \ \{\alpha^{(1)}, \ \alpha^{(2)}\}`,
# randomly
#


def create_params(graph):

    # Creates the interaction parameters
    interaction = [np.random.randint(-100, 100) / 100 for i in range(0, len(graph.edges))]

    # Creates the bias parameters
    bias = [np.random.randint(-100, 100) / 100 for i in range(0, qubit_number)]

    return [interaction, bias]

# Creates and prints the parameters for our simulation
matrix_params = create_params(ising_graph)
print("Target parameters: {}".format(matrix_params))


######################################################################
# Finally, we use this information to generate the matrix form of the
# Ising model Hamiltonian (in the computational basis):
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
# Preparing Quantum Data with VQE
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# To generate the quantum data, we prepare a low-energy state, and 
# then evolve it forward in time with the time-evolution unitary, 
# under the target Hamiltonian.
#
# To prepare the initial low-energy state, we use the :doc:`Variational 
# Quantum Eigensolver </demos/tutorial_vqe>` (VQE). The VQE is
# usually used to prepare an approximate ground state of a given
# Hamiltonian. However, in this particular scenario, we **don’t** want the
# VQE to prepare the ground state, as it will have trivial time-dependence
# equating to a global phase, and we will effectively
# have a bunch of copies of the same state (which is useless).
# In order to ensure that the VQE doesn’t converge, we pick an ansatz such
# that the exact ground state cannot possibly be prepared.
#
# In the case of the Ising model Hamiltonian, we hypothesize (based off of
# the interaction terms present in the Hamiltonian) that there are
# correlations between qubits in the ground state. Thus, we pick an ansatz
# that cannot learn correlations, and instead just rotates the individual
# qubits. We use the single-qubit rotations in the ``AngleEmbedding``
# template for each layer of the ansatz, giving us:
#


def ansatz_circuit(params, vqe_depth, qubits):

    # Splits the parameters
    updated_params = np.split(np.array(params), 4)

    # Applies single-qubit rotations
    for i in range(0, vqe_depth):
        rotations = ["X", "Z", "X", "Z"]
        for j in range(0, len(rotations)):
            qml.templates.AngleEmbedding(
                features=updated_params[j], wires=qubits, rotation=rotations[j]
            )


######################################################################
# We then create the VQE circuit, 
# which returns the expected value of the target Hamiltonian with
# respect to the prepared state.
#


def vqe_circuit(params):

    # Adds the ansatz
    ansatz_circuit(params, vqe_depth, qubits)

    # Measures the expectation value of the Hamiltonian
    return qml.expval(qml.Hermitian(ham_matrix, wires=qubits))


######################################################################
# Notice how we defined two separate methods, ``ansatz_circuit`` and
# ``vqe_circuit``. This is because we eventually have to
# use ``ansatz_circuit`` as a sub-component of the QGRNN circuit (to
# prepare the quantum data). Now, we are able to optimize the VQE circuit
# to arrive at the low-energy state:
#

# Defines the device on which we perform the simulations

vqe_dev = qml.device("default.qubit", wires=qubit_number)
vqe_depth = 1

# Defines the QNode

vqe_qnode = qml.QNode(vqe_circuit, vqe_dev)

# Creates the optimizer for VQE

optimizer = qml.AdamOptimizer(stepsize=0.8)

steps = 200
vqe_params = list([np.random.randint(-100, 100) / 10 for i in range(0, 4 * qubit_number)])

for i in range(0, steps):
    vqe_params = optimizer.step(vqe_qnode, vqe_params)
    if i % 50 == 0:
        print("Cost Step {}: {}".format(i, vqe_qnode(vqe_params)))

print("Parameters: {}".format(vqe_params))


######################################################################
# We can verify that we have prepared a low-energy
# state by numerically finding the lowest eigenvalue of the Hamiltonian
# matrix:
#


def ground_state_energy(matrix):

    # Finds the eigenstates of the matrix
    val = np.linalg.eig(matrix)[0]

    # Returns the minimum eigenvalue
    return min(val)

ground_state = ground_state_energy(ham_matrix)
print("Ground State Energy: {}".format(ground_state))


######################################################################
# This shows us that we have in fact found a low-energy, non-ground state, 
# as the energy we arrived at is slightly greater than that of the true ground
# state. The last step in preparing the quantum-data is to time-evolve
# these low-energy states to arbitrary times, with the 
# time-evolution operator (implemented as a custom unitary in PennyLane).
#


def state_evolve(hamiltonian, qubits, time):

    U = scipy.linalg.expm(complex(0, -1) * hamiltonian * time)
    qml.QubitUnitary(U, wires=qubits)


######################################################################
# We don't actually generate the quantum data quite yet, but we now have 
# all the pieces we need for its preparation.
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
# :math:`U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle`. We need a
# way to measure the similarity between the states contained in the
# registers. One way that we can do this is by using the fidelity, which is
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
# is motivated by the fact that any target interactions graph will be a subgraph 
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
    ansatz_circuit(vqe_params, vqe_depth, reg1)
    ansatz_circuit(vqe_params, vqe_depth, reg2)

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
# :math:`| \langle \psi(t) | U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle |^2`
# approaches :math:`1` as the states become more similar, thus we choose
# to minimize the quantity
# :math:`1 \ - \ | \langle \psi(t) | U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle |^2`.
# Since we are interested in calculating this value for many different
# pieces of quantum data, the final cost function is the **average
# infidelity** between registers:
#
# .. math:: \mathcal{L}(\Delta, \ \boldsymbol\mu) \ = \ 1 \ - \ \frac{1}{N} \displaystyle\sum_{t \ = \ 1}^{N} | \langle \psi(t) | \ U_{H}(\Delta, \ \boldsymbol\mu) \ |\psi_0\rangle |^2,
#
# where we use :math:`N` pieces of quantum data.
#
# Before creating the cost function, we must define a few more fixed
# variables, the device, and the QNode:
#

N = 15  # The number of pieces of quantum data that are used
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
# of the Hamiltonian to which they correspond:
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
# The similarity of colours indicates that the parameters are very
# similar, which we can verify by looking at their numerical values:
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
