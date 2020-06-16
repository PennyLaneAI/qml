"""
The Quantum Graph Recurrrent Neural Network
-------------------------------------------

"""

# Starts by importing all of the necessary dependencies

import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize
import networkx as nx


######################################################################
# In this Notebook, we investigate the idea of a quantum graph
# recurrent neural network (QGRNN), which is the quantum analogue of a
# classical graph RNN, and a subclass of the more general quantum graph
# neural network ansatz (`QGNN <https://arxiv.org/abs/1909.12264>`__).
#


######################################################################
# The Idea
# ~~~~~~~~
#


######################################################################
# In recent years, the idea of a graph neural network has been
# receving a lot of attention from the machine learning research community,
# for its ability to learn from data that is inherently
# graph-theoretic. More specifically, graph neural networks seek
# to learn a **representation** (a mapping of the data into a
# lower-dimensional vector space) of a given graph, with features assigned
# to nodes and edges, such that the each of the vectors in the learned
# representation preserves not only the features, but also the overall 
# topology of the graph.
#
# The quantum graph neural network attempts to do the same thing, but for
# features that are inherently quantum-mechanical (for instance, a
# collection of quantum states).
#


######################################################################
# We define the QGRNN to be a variational ansatz of the form:
#
# .. math:: U_{H}(\boldsymbol\gamma, \ \boldsymbol\theta) \ = \ \displaystyle\prod_{i \ = \ 1}^{P} \Bigg[ \displaystyle\prod_{j \ = \ 1}^{Q} e^{-i \gamma_{j} H_{j}(\boldsymbol\theta) }\Bigg]
#


######################################################################
# Where we have:
#


######################################################################
# .. math:: \hat{H}_{j}(\boldsymbol\theta) \ = \ \displaystyle\sum_{(a,b) \in E} \displaystyle\sum_{c} V_{jabc} \hat{A}_{a}^{jc} \otimes \hat{B}_{b}^{jc} \ + \ \displaystyle\sum_{v \in V} \displaystyle\sum_{d} W_{jvd} \hat{C}_{v}^{jd}
#


######################################################################
# This is a very general class of Hamiltonians that posses a direct
# mapping between interaction and bias terms, and the edges and vertices
# (repsectively) of some graph :math:`G \ = \ (V, \ E)`, which we call 
# the interaction graph. As a result, the
# QGRNN encompasses a very general class of unitaries. As it turns out,
# one of the unitaries that falls under the umbrella of QGRNN ansatze is
# the Trotterized simulation of a Hamiltonian. Let us fix a time
# :math:`T`. Let us also fix a parameter that controls the size of the
# Trotterization steps (essentially the :math:`1/P` in the above formula),
# which we call :math:`\Delta`. This allows us to keep the precision of
# our approximate time-evolution for different values of :math:`T` the
# same. If we define:
#
# .. math:: \hat{H}(\boldsymbol\theta) \ = \ \displaystyle\sum_{q} \hat{H}_{q}(\boldsymbol\theta)
#
# then by using the Trotter-Suzuki decomposition defined above, we can see
# that the time-evolution operator for this particular Hamiltonian can be
# approximated as:
#
# .. math:: e^{-i T \hat{H}(\boldsymbol\theta)} \ \approx \ \displaystyle\prod_{i \ = \ 1}^{T / \Delta} \Bigg[ \displaystyle\prod_{j \ = \ 1}^{Q} e^{-i \Delta H_{j}(\boldsymbol\theta)} \Bigg] \ = \ U_{H}(\Delta, \ \boldsymbol\theta)
#
# This suggests to us that the QGRNN ansatz can be used to learn the
# quantum dynamics of a quantum system. Let’s say that we have some
# Hamiltonian :math:`\hat{H}(\theta)`, with unkown target parameters
# :math:`\boldsymbol\theta \ = \ \boldsymbol\alpha`. The interaction graph
# of the Hamiltonian is also unknown. If given copies of low-energy
# quantum state, which we call :math:`|\psi_0\rangle`, as well as a
# collection of
# :math:`|\psi(t)\rangle \ = \ e^{-itH(\boldsymbol\alpha)} |\psi_0\rangle`
# for a range of times :math:`t`, we can use the QGRNN that we just
# defined to learn the unknown parameters of the target Hamiltonian, and
# thus the interaction graph. More concretely, we have just shown that in
# this case, the QGRNN is equivalent to Trotterized time-evolution, so if
# we take a bunch of pieces of quantum data, :math:`|\psi(t)\rangle`, and
# look at how “similar” they are to
# :math:`U_{H}(\Delta, \boldsymbol\mu) |\psi_0\rangle` (each with
# corresponding :math:`P \ = \ t/ \Delta`), for guessed parameters
# :math:`\boldsymbol\mu`, then optimizing this “similarity” leads to
# :math:`\boldsymbol\mu \ = \ \boldsymbol\alpha`, and we have thus learned
# the target parameters.
#


######################################################################
# You may now be wondering: “how is this problem of Hamiltonian learning a
# **graph-theoretic** problem, fit for a **graph** neural network?”. Well,
# firstly, the :math:`ZZ` terms of the
# Hamiltonian encode exactly where the edges of our graph
# interactions are. In addition to this, if we consider the
# collection of quantum data, :math:`\{ |\psi(t)\rangle \}`, to be the features
# associated with the graph, then we are able to reconstruct these
# features using the Hamiltonian (all we have to do is evolve our fixed
# initial state forward in time to :math:`t`). Thus, the
# Hamiltonian is in fact a representation that describes the interaction
# graph, along with its features.
#


######################################################################
# Learning an Ising Model with the QGRNN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# In this tutorial, we attempt to learn the parameters corresponding
# to some randomly-initialized transverse field Ising model Hamiltonian,
# using the QGRNN.
#


######################################################################
# Getting Started
# ~~~~~~~~~~~~~~~
#


######################################################################
# We begin by defining some fixed values that are used throughout
# the simulation.
#

# Initializes fixed values

qubit_number = 4
qubits = range(qubit_number)


######################################################################
# In order to use the QGRNN, we need access to quantum data. In the real
# world, this wouldn’t be a problem, but in this simulation, we have to
# generate the quantum data ourself. To do this, we must have *a priori*
# knowledge of the target Hamiltonian and the interaction graph. We choose
# the Hamiltonian to be a transerve field Ising model of the form:
#
# .. math:: \hat{H}(\boldsymbol\theta) \ = \ \displaystyle\sum_{(i, j) \in E} \theta_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \theta_{i}^{(2)} Z_{i} \ + \ \displaystyle\sum_{i} X_{i}
#
# where :math:`E` is the set of edges in the interaction graph. We define
# the target interaction graph of the Hamiltonian to be the cycle graph:
#

# Creates the graph structure of the quantum system

ising_graph = nx.Graph()
ising_graph.add_nodes_from(range(0, qubit_number))
ising_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# Plots the graph

nx.draw(ising_graph)
print("Edges: " + str(ising_graph.edges))


######################################################################
# We then can initialize the “unknown” target parameters that describe the
# Hamiltonian, :math:`\alpha \ = \ \{\alpha^{(1)}, \ \alpha^{(2)}\}`,
# randomly
#


def create_params(graph):

    # Creates the interaction parameters
    interaction = [np.random.randint(-150, 150) / 100 for i in range(0, len(graph.edges))]

    # Creates the bias parameters
    bias = [np.random.randint(-150, 150) / 100 for i in range(0, qubit_number)]

    return [interaction, bias]


# Creates and prints the parameters for our simulation
matrix_params = create_params(ising_graph)
print("Target parameters: " + str(matrix_params))


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

    # Creates the "bias" components of the matrix
    for i in range(0, n):
        m1 = 1
        m2 = 1
        for j in range(0, n):
            if j == i:
                m1 = np.kron(m1, qml.PauliZ.matrix)
                m2 = np.kron(m2, qml.PauliX.matrix)
            else:
                m1 = np.kron(m1, np.identity(2))
                m2 = np.kron(m2, np.identity(2))
        matrix = np.add(matrix, np.add(params[1][i] * m1, m2))

    return matrix


# Defines and prints the matrix for the target interaction graph and parameters
ham_matrix = create_hamiltonian_matrix(qubit_number, ising_graph, matrix_params)
print(ham_matrix)


######################################################################
# Preparing Quantum Data with VQE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# To generate the quantum data (a collection of time-evolved, low-energy quantum
# states), the strategy we use is preparing a low-energy state, and then evolving
# it forward in time with the time-evolution unitary, under the target
# Hamiltonian.
#
# To prepare the initial, low-energy state, we use VQE. The VQE is
# usually used to prepare an approximate ground state of a given
# Hamiltonian. However, in this particular scenario, we **don’t** want the
# VQE to prepare the ground state, as it will have trivial time-dependence
# equating to the addition of a global phase, and we will effectively
# have a bunch of copies of the same state (which is useless).
# In order to ensure that the VQE doesn’t converge, we pick an ansatz such
# that the exact ground state cannot possibly be prepared.
#
# In the case of the Ising model Hamiltonian, we hypothesize (based off of
# the interaction terms present in the Hamiltonian) that there are
# correlations between qubits in the ground state. Thus, we pick an ansatz
# that cannot learn correlations, and instead just rotates the individual
# qubits. Our ansatz circuit is defined as:
#

# Defines the ansatz circuit

def ansatz_circuit(params, vqe_depth, qubits):

    # Splits the parameters
    updated_params = np.split(np.array(params), 4)

    # Applies single-qubit rotations
    for i in range(0, vqe_depth):
        rotations = ["X", "Z", "X", "Z"]
        for j in range(0, len(rotations)):
            qml.templates.embeddings.AngleEmbedding(
                features=updated_params[j], wires=qubits, rotation=rotations[j]
            )


######################################################################
# Where we have used single-qubit rotations in the ``AngleEmbedding``
# template for each layer. We then create the VQE circuit, 
# which returns the expected value of the target Hamiltonian with
# respect to the prepared state.
#

# Defines the circuit that is used for VQE

def vqe_circuit(params):

    # Adds the ansatz
    ansatz_circuit(params, vqe_depth, qubits)

    # Measures the expectation value of the Hamiltonian
    return qml.expval(qml.Hermitian(ham_matrix, wires=qubits))


######################################################################
# Notice how we defined two separate methods, ``ansatz_circuit`` and
# ``vqe_circuit``. This is because we eventually have to
# use ``ansatz_circuit`` as a sub-component of the QGRNN circuit, to
# prepare the quantum data. Now, we are able to optimize the VQE circuit
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
        print("Cost Step " + str(i) + ": " + str(vqe_qnode(vqe_params)))

print("Parameters: {}".format(vqe_params))


######################################################################
# We can verify that we have prepared a low-energy
# state by numerically finding the lowest eigenvalue of the Hamiltonian
# matrix:
#

# Finds the ground state energy of a Hamiltonian

def ground_state_energy(matrix):

    # Finds the eigenstates of the matrix
    val = np.linalg.eig(matrix)[0]

    # Returns the minimum eigenvalue
    return min(val)

ground_state = ground_state_energy(ham_matrix)
print("Ground State Energy: " + str(ground_state))


######################################################################
# This shows us that we have found a low-energy, non-ground state, as the
# energy we arrived at is slightly greater than that of the true ground
# state. The last step in preparing the quantum-data is to time-evolve
# these low-energy states to arbitrary times, which we can do using the
# time-evolution operator (implemented as a custom unitary in PennyLane):
#

# Creates an exact time-evolution unitary

def state_evolve(hamiltonian, qubits, time):

    U = scipy.linalg.expm(complex(0, -1) * hamiltonian * time)
    qml.QubitUnitary(U, wires=qubits)


######################################################################
# We won’t generate the quantum data quite yet, but we have all the pieces
# we need for its preparation.
#


######################################################################
# Learning the Hamiltonian with a QGRNN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

# Method that prepares a time-evolution layer


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
# registers. In one register, some piece of quantum data,
# :math:`|\psi(t)\rangle`, is prepared, and in the other, we have
# :math:`U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle`. We need a
# way to measure the similarity between the states contained in the
# registers. One way that we can do this is using the fidelity, which is
# simply the modulus squared of the inner product between the states,
# :math:`| \langle \psi(t) | U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle |^2`.
# To calculate this value, we utilize a `SWAP
# test <https://en.wikipedia.org/wiki/Swap_test>`__ between the registers:
#

# Implements the SWAP test between two qubit registers

def swap_test(control, register1, register2):

    qml.Hadamard(wires=control)
    for i in range(0, len(register1)):
        qml.CSWAP(wires=[int(control), register1[i], register2[i]])
    qml.Hadamard(wires=control)


######################################################################
# Before proceeding with defining the full QGRNN and the cost function, we
# define a few more fixed values. Among these fixed values is a "guessed"
# interaction graph. Recall that part of the idea behind the QGRNN is that
# we don’t know the interaction graph, and it has to be learned by setting
# certain :math:`\boldsymbol\mu` parameters to :math:`0`, for corresponding
# edges that don’t exist in the target interaction graph. We define the
# initial “guessed” graph to simply be the complete graph:
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
# Finally, we implement the QGRNN circuit:
#

# Implements the quantum graph neural network for a given time step

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
# :math:`\langle Z \rangle`, with respect to the ancilla qubit, as after
# performing a SWAP test between two states :math:`|\psi\rangle` and
# :math:`|\phi\rangle`, it is tue that
# :math:`\langle Z \rangle \ = \ |\langle \psi | \phi \rangle|^2`.
#
# We have the full QGRNN circuit, but we still need to define a cost
# function to minimize. We know that
# :math:`| \langle \psi(t) | U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle |^2`
# approaches :math:`1` as the states become more similar, thus we choose
# to minimize the quantity
# :math:`1 \ - \ | \langle \psi(t) | U_{H}(\Delta, \ \boldsymbol\mu) |\psi_0\rangle |^2`.
# Since we are interested in calculating this value for many different
# pieces of quantum data, and final cost function is the **average
# infidelity** between registers:
#
# .. math:: \mathcal{L}(\Delta, \ \boldsymbol\mu) \ = \ 1 \ - \ \frac{1}{N} \displaystyle\sum_{t \ = \ 1}^{N} | \langle \psi(t) | \ U_{H}(\Delta, \ \boldsymbol\mu) \ |\psi_0\rangle |^2
#
# where we use :math:`N` pieces of quantum data.
#
# Before creating the cost function, we must define a few more fixed
# variables, the device, and the QNode:
#

N = 15  # The number of different times that are used
max_time = 0.1  # The maximum value of time that can be used for quantum data

# Defines the new device

qgrnn_dev = qml.device("default.qubit", wires=2 * qubit_number + 1)

# Defines the new QNode

qnode = qml.QNode(qgrnn, qgrnn_dev)


######################################################################
# We then define the infidelity cost function:
#

# Defines the cost function


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
# with a step-size of :math:`0.2`:
#

# Defines the optimization method

iterations = 0

optimizer = qml.AdamOptimizer(stepsize=0.2)
steps = 150
qgrnn_params = list([np.random.randint(-20, 20) / 50 for i in range(0, 10)])

# Executes the optimization method

for i in range(0, steps):
    qgrnn_params = optimizer.step(cost_function, qgrnn_params)

print(qgrnn_params)


######################################################################
# With the learned parameters, we construct the Hamiltonian to which they
# correspond:
#

new_ham_matrix = create_hamiltonian_matrix(
    qubit_number, nx.complete_graph(qubit_number), [qgrnn_params[0:6], qgrnn_params[6:10]]
)
print(new_ham_matrix)


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

target_params = list(np.array(matrix_params).flatten())
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
# ~~~~~~~~~~
#
# 1. Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., &
#    Hidary, J. (2019). Quantum Graph Neural Networks. arXiv preprint
#    `arXiv:1909.12264 <https://arxiv.org/abs/1909.12264>`__.
#
