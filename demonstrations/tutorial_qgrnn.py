"""
The Quantum Graph Reccurent Neural Network
---------------------------------------------------

"""

# Starts by importing all of the necessary dependencies

import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize
import networkx as nx
import seaborn


######################################################################
# Introduction
# ~~~~~~~~~~~~
#


######################################################################
# In this tutorial, we investigate the idea of a quantum graph
# neural network (`QGNN <https://arxiv.org/abs/1909.12264>`__), 
# which is the quantum analogue of a classical graph neural network. 
# In particular, the QGNN we simulate is a **recurrent** 
# quantum graph neural network (QGRNN), which is a subclass of the general
# QGNN.
#


######################################################################
# The Quantum Graph Neural Network
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# In recent years, the idea of a graph neural network has been
# receving a lot of attention from the machine learning research community
# for its ability to learn from data that is inherently
# graph-theoretic. More specifically, graph neural networks seek
# to learn a **representation** (a mapping of the data into a
# lower-dimensional vector space) of a given graph with features assigned
# to nodes and edges, such that each of the vectors in the learned
# representation preserves not only the features, but also the overall
# topology of the graph.
#


######################################################################
# We attempt to perform the same task, but from a quantum
# computational perspective. Specifically, we want to define an ansatz
# that we can use for quantum machine learning tasks that are inherently
# graph-theoretic. From the original QGNN paper, the general QGNN ansatz
# is defined as:
#
# .. math:: U(\boldsymbol\gamma, \ \boldsymbol\theta) \ = \ \displaystyle\prod_{i \ = \ 1}^{P} \Bigg[ \displaystyle\prod_{j \ = \ 1}^{Q} e^{-i \gamma_{ij} H_{j}(\boldsymbol\theta) }\Bigg]
#
# With a collection of parametrized Hamiltonians of the form:
#
# .. math:: \hat{H}_{j}(\boldsymbol\theta) \ = \ \displaystyle\sum_{(a,b) \in E} \displaystyle\sum_{c} \theta_{jabc}^{(1)} \hat{A}_{a}^{jc} \otimes \hat{B}_{b}^{jc} \ + \ \displaystyle\sum_{v \in V} \displaystyle\sum_{d} \theta_{jvd}^{(2)} \hat{C}_{v}^{jd}
#
# where :math:`\boldsymbol\theta \ = \ \{\boldsymbol F, \ \boldsymbol G\}`
# is the set of coefficients associated with the interaction and bias terms
# As you can see, this is the class of Hamiltonians that posses a direct
# mapping between interaction and bias terms, and the edges and vertices
# (repsectively) of some graph :math:`G \ = \ (V, \ E)`. This form of the
# Hamiltonian is very general, and as a result, has a lot of indices. The
# subscripts on operators represent action upon the
# :math:`a`-th, :math:`b`-th or :math:`v`-th qubit, and the other
# parameters (:math:`c` and :math:`d`) simply allow us to label each
# parameter/type of operator in each term. 
# This makes the QGRNN ansatz encompass a very broad class of unitaries.
# In fact, all we can really learn from looking at the QGNN ansatz is that 
# it involves many repeated layers of parametrized rotation and coupling 
# gates, acting on qubits according to the structure of some graph.
#


######################################################################
# The Quantum Graph Recurrent Neural Network
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Now that we have discussed what a general quantum graph neural network 
# looks like, let’s discuss what distinguishes a quantum graph 
# recurrent neural network (QGRNN) from the more general QGNN. 
#
# With the QGRNN, we tie the :math:`\boldysmbol\gamma` parameters
# over temporal layers, meaning that each :math:`\gamma_{ij}`
# is only determined by the index :math:`j`, and is the same across all
# values of :math:`i` (with some fixed :math:`j`). As the paper explains,
# this is equivalent to how a classical recurrent neural network behaves,
# where parameters remain the same over layers of the neural network.
# Thus, we will have:
#
# .. math:: U_{\text{RNN}}(\boldsymbol\gamma, \ \boldsymbol\theta) \ = \ \displaystyle\prod_{i \ = \ 1}^{P} \Bigg[ \displaystyle\prod_{j \ = \ 1}^{Q} e^{-i \gamma_{j} H_{j}(\boldsymbol\theta) }\Bigg]
#
# We are now in a position to understand what the QGRNN ansatz can be 
# used for. The Trotter-Suzuki decomposition says that:
#
# .. math:: \exp \Bigg[ \displaystyle\sum_{n} A_n \Bigg] \ = \ \lim_{P \rightarrow \infty} \displaystyle\prod_{j \ = \ 1}^{P} \Bigg[ \displaystyle\prod_{n} e^{A_n / P} \Bigg],
#


######################################################################
# In practice, we can't implement a quantum circuit with 
# :math:`P \ \rightarrow \ \infty`, but we can perform an approximate 
# Trotter-Suzuki  decomposition, with finite :math:`P \ \gg \ 1`. It 
# isn’t too difficult to see that the quantum graph RNN resembles the 
# Trotter-Suzuki decomposition. In fact, there is a specific case of 
# the quantum graph RNN can be thought of as the Trotterization of the 
# **time-evolution operator**, for some Hamiltonian. Let us fix a time 
# :math:`t`. Let us also fix a parameter that controls the 
# size of the Trotterization steps (essentially :math:`1/P` in exponents, in 
# the above formula), which we call :math:`\Delta`. This allows us to keep 
# the precision of our approximate time-evolution for different values of 
# :math:`t` the same. We define:
#
# .. math:: \hat{H} \ = \ \displaystyle\sum_{q} \hat{H}_{q}(\boldsymbol\theta)
#
# By using the Trotter-Suzuki decomposition defined above, we can see that
# the time-evolution operator for this particular Hamiltonian can be
# approximated as:
#
# .. math:: e^{-i t H} \ \approx \ \displaystyle\prod_{i \ = \ 1}^{t / \Delta} \Bigg[ \displaystyle\prod_{q \ = \ 1}^{Q} e^{-i \Delta H_{q}(\boldsymbol\theta)} \Bigg] \ = \ U_{\text{RNN}}(\Delta, \ \boldsymbol\theta)
#


######################################################################
# Thus, time-evolution is just a specific case of the QGRNN ansatz, for
# some arbitrary collection of parametrized Hamiltonians. This suggests
# to us that this ansatz may be particularly useful for learning the
# dynamics of quantum systems. 
#
# Let’s say that we are given a set
# :math:`\{|\Psi(t)\rangle\}_{t}` of low-energy states that have
# been evolved under some unknown Hamiltonian :math:`H`, for a collection of
# different times, :math:`t`, along with copies of  
# the low-energy gate at :math:`t \ = \ 0`. Our goal is to 
# determine the Hamiltonian of this system, given only these collections of states, and
# the type of Hamiltonian by which the system evolves (Ising, Heisenberg,
# etc.). In theory, one doesn’t have to know the type of Hamiltonian,
# but the algorithm would then require an absurd amount of gates, and a lot of
# quantum data, so we will stick to the simpler case.
#


######################################################################
# The Hamiltonian as a Feature Representation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You may now be wondering: “how is this problem of Hamiltonian learning a
# **graph-theoretic** problem, fit for a **graph** neural network?”.
# Well, this process of learning the
# Hamiltonian is a form of graph representation learning! 
# The interaction terms of the Hamiltonian will encode
# exactly where the edges of our graph interactions are, and 
# if we consider the collection of states
# :math:`\{ |\Psi(t)\rangle \}` to be the features associated with the
# graph, then we are able to determine all of these from the
# Hamiltonian (all we have to do is evolve our fixed initial state forward
# in time by some time :math:`t`). Thus, the Hamiltonian is exactly the
# node representation that describes the graph along with its features. In
# this paradigm, however, we are dealing with inherently quantum
# mechanical features, so our quantum graph neural
# network is particularly well-suited to this task.
#


######################################################################
# The QGRNN Applied to an Ising Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this Notebook we apply the QGRNN
# to learning the dynamics of a 4-qubit Ising model.
#


######################################################################
# Initializing the Graph and Preparing the Target Hamiltonian
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# We can begin by defining some fixed values that will be used throughout
# the simulation, as well as the device on which we will run our
# simulations:
#

# Initializes fixed values

qubit_number = 4
qubits = range(qubit_number)

# Initializes the device on which the simulation is run

vqe_dev = qml.device("default.qubit", wires=qubit_number)


######################################################################
# We continue by defining the graph on which our Ising model is defined, using
# `networkx`. For this tutorial, we pick a simple :math:`4`-node cycle graph.
#

# Creates the graph structure of the quantum system

ising_graph = nx.Graph()
ising_graph.add_nodes_from(range(0, qubit_number))
ising_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# Plots the graph

nx.draw(ising_graph)

print("Edges: " + str(ising_graph.edges))


######################################################################
# Now, we define the Hamiltonian we eventually learn with the QGRNN.
# There are two reasons for doing this. Firstly, we need to compare the
# prepared Hamiltonian to the target Hamiltonian, to make sure that our
# toy example of the QGRNN actually works. Secondly, we have to
# prepare the quantum data that will be used in training. In a real-world
# application of this algorithm, we would have access to a bunch of
# quantum data that we could feed into the neural network, but in this
# scenario, we have to generate the quantum data ourselves. The only way to
# do this is to know the Hamiltonian (and thus, the dynamics) of our
# system *a priori*. The Ising model Hamiltonian will be of the form:
#
# .. math:: \hat{H} \ = \ \displaystyle\sum_{(i, j) \in E} \alpha_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \alpha_{i}^(2) Z_{i} \ + \ \displaystyle\sum_{i} X_{i},
#
# where :math:`Z_i` and :math:`X_i` are the Pauli-Z and Pauli-X 
# acting on the :math:`i`-th qubit, and :math:`E` is the set of edges of 
# the cyclic interaction graph that we defined. In addition, 
# :math:`\boldsymbol\alpha` is the collection of target parameters, 
# composed of subsets :math:`\boldsymbol\alpha^{(1)}` and 
# :math:`\boldsymbol\alpha^{(2)}` of interaction and bias parameters, 
# respectively. These are the collections of values that we are attempting 
# to learn with the QGRNN. 
#
# We initialize the target parameters randomly:
#


def create_params(graph):

    # Creates the interaction parameters
    interaction = [np.random.randint(-200, 200) / 100 for i in range(0, len(graph.edges))]

    # Creates the bias parameters
    bias = [np.random.randint(-200, 200) / 100 for i in range(0, qubit_number)]

    return [interaction, bias]


# Creates and prints the target parameters
matrix_params = create_params(ising_graph)
print("Target parameters: " + str(matrix_params))


######################################################################
# With this knowledge, we construct the matrix representation of
# the Hamiltonian in the computational basis:
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

    # Creates the "bias" component of the matrix
    for i in range(0, n):
        m = 1
        for j in range(0, n):
            if j == i:
                m = np.kron(m, qml.PauliZ.matrix)
            else:
                m = np.kron(m, np.identity(2))
        matrix = np.add(matrix, params[1][i] * m)

    # Creates the X component of the matrix
    for i in range(0, n):
        m = 1
        for j in range(0, n):
            if j == i:
                m = np.kron(m, qml.PauliX.matrix)
            else:
                m = np.kron(m, np.identity(2))
        matrix = np.add(matrix, m)

    return matrix


# Defines and prints the Ising Hamiltonian for the given interaction graph and parameters
ham_matrix = create_hamiltonian_matrix(qubit_number, ising_graph, matrix_params)
print(ham_matrix)


######################################################################
# The Hamiltonian is the main object of interest in the quantum
# graph neural network, so let’s visualize it with a colour-plot.
# Since the trainable parameters in the Ising model correspond only to 
# :math:`Z` gates or :math:`ZZ` interactions, the diagonal elements of 
# the Hamiltonian change over steps of the QGRNN while the off-diagonal 
# components remain fixed, so we only really have to look at the diagonal:
#

# Matrix colour-plot function

def create_density_plot(data):

    array = np.array(data)
    plt.matshow(array)
    plt.colorbar()
    plt.axis('off')
    plt.show()

# Plots the diagonal of the Hamiltonian

create_density_plot([np.diag(ham_matrix)])


######################################################################
# Preparing the Quantum Data with VQE
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# Now that we have constructed the target Hamiltonian, we can use it to
# prepare all of the quantum data needed to train the
# QGRNN. To generate the data, we first prepare copies of a 
# low-energy state, which we then evolve to different times, to prepare a
# collection of different states from which the QGRNN can learn.
#
# To prepare the low-energy states, we use the Variational Quantum Eigensolver
# (`VQE <https://pennylane.ai/qml/demos/tutorial_vqe.html>`__). 
# The purpose of the VQE is to find the ground energy state 
# of a given Hamiltonian, however, for the purposes of the QGRNN, we don’t 
# want to find the **exact** ground state of the Hamiltonian, as it’s time 
# evolution will just equate to the addition of a global phase:
#
# .. math:: U(t)|E_k\rangle \ = \ e^{-i H t / \hbar} |E_k\rangle \ = \ \displaystyle\sum_{n \ = \ 0}^{\infty} \Big( -\frac{i t}{\hbar} \Big)^n \frac{H^n |E_k\rangle}{n!} \ = \ \displaystyle\sum_{n \ = \ 0}^{\infty} \frac{1}{n!} \Big( \frac{- i E_k t}{\hbar} \Big)^n |E_k\rangle \ = \ e^{i E_k t / \hbar} |E_k\rangle
#
# It follows that this quantum data will be practically useless, as we are
# essentially just giving ourselves a bunch of copies of the exact same
# quantum state. So, instead of prearing the exact ground state, we
# attempt to prepare a state that is close to, but not quite the ground
# state, giving us non-trivial time 
# evolution. To do this, we will 
# use a VQE ansatz that we know **can’t converge exactly**
# to the ground state of the Hamiltonian.
#
# For the specific example of the Ising model Hamiltonian we are considering in
# this tutorial, we have :math:`ZZ` interaction terms between qubits that share
# an edge on the graph. This allows us to hypothesize that there will be 
# correlations between qubits in the ground state. We
# then choose our ansatz such that the qubits **do not** interact with any
# multi-qubit gates, and as a result, our output state will  always be separable.
#
# The initial layer of the VQE ansatz is an even
# superposition over all basis states:
#

# Defines a method that creates an even superposition of basis states


def even_superposition(qubits):

    for i in qubits:
        qml.Hadamard(wires=i)


######################################################################
# The following layers is alternating applications of 
# single-qubit :math:`RZ` and :math:`RX` gates:
#

# Defines the decoupled rotational ansatz


def decoupled_layer(param1, param2, qubits):

    # Applies a layer of RZ and RX gates
    for count, i in enumerate(qubits):

        qml.RZ(param1[count], wires=i)
        qml.RX(param2[count], wires=i)


######################################################################
# We then define the VQE ansatz as:
#

# Method that creates the decoupled VQE ansatz


def vqe_circuit(parameters, qubits, depth):

    even_superposition(qubits)

    for i in range(0, depth):
        decoupled_layer(parameters[0], parameters[1], qubits)


######################################################################
# Next, we create a function and a QNode that allow us to run our
# VQE circuit:
#

# Defines the depth of our variational circuit
vqe_depth = 2

# Defines the circuit that we use to perform VQE for the Hamiltonian
def create_circuit(params1, params2):

    vqe_circuit([params1, params2], qubits, vqe_depth)

    return qml.expval(qml.Hermitian(ham_matrix, wires=range(qubit_number)))


# Creates the corresponding QNode
qnode = qml.QNode(create_circuit, vqe_dev)

# Prints out the circuit
resulting_circuit = qnode([1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
print(resulting_circuit)
print(qnode.draw())


######################################################################
# Finally, we define the cost function, along with an optimizer and run 
# it:
#

# Creates the cost function


def cost_function(params):

    return qnode(params[0:qubit_number], params[qubit_number : 2 * qubit_number])


# Creates the optimizer for VQE

optimizer = qml.AdamOptimizer(stepsize=0.8)

steps = 200
vqe_params = list([np.random.randint(-100, 100) / 10 for i in range(0, 2 * qubit_number)])

for i in range(0, steps):
    vqe_params = optimizer.step(cost_function, vqe_params)
    if i % 50 == 0:
        print("Cost Step " + str(i) + ": " + str(cost_function(vqe_params)))

print(vqe_params)


######################################################################
# Let's verify our results by numerically calculating the actual ground 
# state energy of the Hamiltonian, and compare it to the energy of the 
# VQE-generated state:
#

# Finds the ground state energy of a Hamiltonian

def ground_state_energy(matrix):

    # Finds the eigenstates of the matrix
    val = np.linalg.eig(matrix)[0]

    # Returns the minimum eigenvalue
    return min(val)


ground_state = ground_state_energy(ham_matrix)
print("Ground State Energy: "+str(ground_state))


######################################################################
# The ground state energy is slightly lower than the energy value we 
# found above with the decoupled VQE, so we can conclude that we have 
# generated a low-energy, non-ground state. With this state, we can now prepare the
# quantum data, by evolving it forward in time under the Ising Hamiltonian. 
# 
# We can implement a custom time-evolution unitary gate in PennyLane:
#

# Creates an exact time-evolution unitary

def state_evolve(hamiltonian, qubits, time):

    U = scipy.linalg.expm(complex(0, -1) * hamiltonian * time)
    qml.QubitUnitary(U, wires=qubits)


######################################################################
# Learning the Hamiltonian
# ^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# Now that have all the pieces needed to generate the quantum data,
# let’s turn our attention to the QGRNN itself. The motivation for the 
# following QGRNN ansatz comes from the first part of the 
# tutorial, where we showed that the QGRNN can be used
# to approximate time-evolution. If we construct the QGRNN such that it
# is exactly a Trotterized version of an Ising model Hamiltonian, with tuneable
# coefficients, then the neural network can learn the optimal parameters through training 
# on the quantum data.
#
# The process begins by preparing a piece of quantum data, at an arbitrary 
# time :math:`t` from :math:`0` to :math:`T` in the first register of the quantum circuit. 
# In the second register, we initialize the qubits in the low-energy state
# we found in the previous section, which we call :math:`|\psi_0\rangle` for the remainder
# of this tutorial. We then pass the second register through the QGRNN ansatz,
# :math:`U_{\text{RNN}}(\Delta, \ \boldsymbol\theta')`, for fixed :math:`\Delta`,  
# where :math:`\boldsymbol\theta'` is the list of trainable parameters. We define 
# the depth of the outer sequence index to be :math:`P \ = \ \frac{t}{\Delta}`.
# In addition, since we know we are dealing with an Ising Hamiltonian, we pick 
# the parametrized Hamiltonians in the ansatz to be:
#
# .. math:: H_1(\boldsymbol\theta) \ = \ \displaystyle\sum_{(i, j) \in E'} \boldsymbol\theta_{ij}^{(1)} Z_{i} Z_{j} \ + \ \displaystyle\sum_{i} \boldsymbol\theta_{i}^{(2)} Z_{i}
#
# .. math:: H_2(\boldsymbol\theta) \ = \ \displaystyle\sum_{i} X_{i} 
#
# Where :math:`E'` is a guessed interaction graph (remember, the 
# whole point of the QGRNN is that we have no initial knowledge of the target set of 
# edges, :math:`E`). After passing the second register through the QGRNN, we perform a 
# `SWAP test <https://en.wikipedia.org/wiki/Swap_test>`__ between the 
# two registers, which allows us to compute the inner product between the quantum 
# data and the state prepared with the QGRNN, telling us how similar the two states 
# are. This process is repeated :math:`N` times, for :math:`N` pieces of quantum data, 
# evolved to randomly chosen times from :math:`0` to :math:`T`, after which the results 
# of each execution are combined in the following cost function:
#
# .. math:: \mathcal{L}(\Delta, \ \boldsymbol\theta) \ = \ 1 \ - \ \frac{1}{N} \displaystyle\sum_{i \ = \ 0}^{N} | \langle \Psi(t_i) | \ U_{\text{RNN}}(\Delta, \ \boldsymbol\mu)^{i} \ |\psi_0\rangle |^2
#
# where :math:`t_i` denotes the :math:`i`-th chosen time. This entire 
# process is repeatedly executed, with new parameters, 
# :math:`\boldsymbol\theta'`, chosen by a classical optimizer until the 
# cost function is minimized and the QGRNN terminates.
#
# As the cost function decreases, we know that the inner product between the quantum
# data and the parametrized state is approaching one, indicating the state are becoming 
# more "similar". Since both registers were initialized as :math:`|\psi_0\rangle`, this implies
# that the QGRNN ansatz acts similarly to the time-evolution operator under the target Hamiltonian.
# We chose the parametrized Hamiltonians used in the QGRNN to 
# contain identical operators to that of the target Ising model, and (as we demonstrated 
# in the first part of the Notebook) this particular QGRNN ansatz is essentially 
# equivalent to time-evolution under the sum of its parametrized Hamiltonians. This means that 
# the only way the QGRNN ansatz could be equivalent to the time-evolution operator
# is if the learned parameters :math:`\boldsymbol\theta'` are approaching 
# the target parameters, :math:`\boldsymbol\theta`, therefore, minimizing the cost 
# function allows us to learn the target Hamiltonian!
#
# Here is a diagram depicting the entire quantum circuit:
#
# .. image:: qgrnn/qgrnn.png
#
# Without further delay, let's build the QGRNN! 
# Th Ising Hamiltonian involves :math:`ZZ` gates, so we write a
# method that allows us to construct :math:`RZZ` gates out of the standard
# gate set in Pennylane:
#

# Defines the RZZ gate, in terms of gates in the standard basis set

def RZZ(param, qubit1, qubit2):

    qml.CNOT(wires=[qubit1, qubit2])
    qml.RZ(param, wires=qubit2)
    qml.CNOT(wires=[qubit1, qubit2])


####################################################################
# With this, we can now write a function that implements one layer 
# of the QGRNN:
#


def qgrnn_layer(param1, param2, qubits, graph, trotter):

    # Applies a layer of coupling gates (based on the guessed graph)
    for count, i in enumerate(graph.edges):
        RZZ(2 * param1[count] * trotter, i[0], i[1])

    # Applies a layer of RZ gates
    for count, i in enumerate(qubits):
        qml.RZ(2 * param2[count] * trotter, wires=i)

    # Applies a layer of RX gates
    for i in qubits:
        qml.RX(2 * trotter, wires=i)


######################################################################
# Finally, we will need to make use of the :math:`SWAP` test, to calculate
# the fidelity between the prepared quantum state and our quantum data. We can
# write a function that performs this process:
#

# Implements the SWAP test between two qubit registers

def swap_test(control, register1, register2):

    qml.Hadamard(wires=control)
    for i in range(0, len(register1)):
        qml.CSWAP(wires=[int(control), register1[i], register2[i]])
    qml.Hadamard(wires=control)


######################################################################
# From here, we can build our quantum circuit that corresponds to one
# iteration of the QGRNN, for a given time step. First, let’s define a new
# quantum device with :math:`9` qubits rather than :math:`4`, as we will
# need registers for the neural network and data-preparation, plus an
# extra ancilla qubit for the :math:`SWAP` test.
#

# Defines the new
qgrnn_dev = qml.device("default.qubit", wires=2 * qubit_number + 1)


######################################################################
# We then define a few more fixed variables needed for our simulation.
# One of these variables is the new, guessed graph of interations. 
# As we previously noted, we do not know the
# interaction structure (which qubits are “connected” by :math:`ZZ` 
# interactions) of our model before we begin the simulation. Thus, 
# we initialize our system in the complete graph of :math:`4` nodes 
# and hope that as our algorithm converges, it learn the structure of
# the target cycle graph by setting the parameters 
# corresponding to edge :math:`(0, \ 2)` and :math:`(1, \ 3)` to :math:`0`.
#

# Defines some fixed values

reg1 = list(range(qubit_number))
reg2 = list(range(qubit_number, 2 * qubit_number))

control = 2 * qubit_number
trotter_step = 0.01

# Defines the interaction graph for the new qubit system

new_ising_graph = nx.Graph()
new_ising_graph.add_nodes_from(range(qubit_number, 2 * qubit_number))
new_ising_graph.add_edges_from([(4, 5), (5, 6), (6, 7), (4, 6), (7, 4), (5, 7)])

print("Edges: " + str(new_ising_graph.edges))
nx.draw(new_ising_graph)


######################################################################
# Finally, we can implement the QGRNN ansatz, as we outlined above:
#

# Implements the quantum graph neural network for a given time

def qgrnn(params1, params2, time):

    # Prepares the low energy state in the two qubit registers
    vqe_circuit(
        [vqe_params[0:qubit_number], vqe_params[qubit_number : 2 * qubit_number]], reg1, vqe_depth
    )
    vqe_circuit(
        [vqe_params[0:qubit_number], vqe_params[qubit_number : 2 * qubit_number]], reg2, vqe_depth
    )

    # Evolves the first qubit register with the time-evolution circuit
    state_evolve(ham_matrix, reg1, time.val)

    # Applies the QGRNN layers to the second qubit register
    depth = time.val / trotter_step
    for i in range(0, int(depth)):
        qgrnn_layer(params1, params2, reg2, new_ising_graph, trotter_step)

    # Applies the SWAP test between the registers
    swap_test(control, reg1, reg2)

    # Returns the results of the SWAP test
    return qml.expval(qml.PauliZ(control))


######################################################################
# The reason we set the function to return the expectation value of the 
# Pauli-Z with respect to the ancilla qubit is due to the SWAP test.
# After performing the SWAP test, it is true that:
#
# .. math:: \text{P}(\text{Ancilla} \ = \ 0) \ = \ \frac{1}{2} \ + \ \frac{1}{2} | \langle \psi | \phi \rangle |^2
#
# Where :math:`|\psi\rangle` and :math:`|\phi\rangle` are the states
# contained in the two registers. We also have:
#
# .. math:: \langle Z_A \rangle \ = \ (1) \text{P}(\text{Ancilla} \ = \ 0) \ + \ (-1) \text{P}(\text{Ancilla} \ = \ 1) 
#                                 = \ \text{P}(\text{Ancilla} \ = \ 0) \ - \ \big(1 \ - \ \text{P}(\text{Ancilla} \ = \ 0)\big)
#                                 = \ 2\text{P}(\text{Ancilla} \ = \ 0) \ - \ 1
#
# where :math:`Z_A` is the Pauli-Z observable corresponding to the 
# ancilla qubit. From the definition of the expectation value, we 
# also have
#
# .. math:: \text{P}(\text{Ancilla} \ = \ 0) \ = \ \frac{1}{2} \ + \ \frac{1}{2} \langle Z_A \rangle,
#
# which, from the first and third equations, gives us:
# :math:`\langle Z_A \rangle \ = \ |\langle \psi | \phi \rangle|^2`. 
# This happens to be exectly the quantity we need to calculate the 
# cost function.

######################################################################
# Now, we can construct the cost function of the model. To
# evaluate the cost function, we simply have to choose a bunch of time
# steps at which we will execute of QGRNN. We begin by defining a few more
# values, along with the new QNode:
#

batch = 15  # The number of different times that will be used
max_time = 0.1  # The maximum value of time that can be used to prepare quantum data

# Defines the new QNode

qnode = qml.QNode(qgrnn, qgrnn_dev)


######################################################################
# Then, we define the cost function as it was described earlier in the 
# tutorial:
#

# Defines the cost function

def cost_function(params):

    global iterations

    # Separates the parameter list
    weight_params = params[0:6]
    bias_params = params[6:10]

    # Samples times at which the QGRNN will be run
    times_sampled = [np.random.uniform() * max_time for i in range(0, batch)]

    # Cycles through each of the sampled times and calculates the cost
    total_cost = 0
    for i in times_sampled:
        result = qnode(weight_params, bias_params, i)
        total_cost += 1 - result

    # Prints the value of the fidelity
    print(
        "Fidelity at Step "
        + str(iterations)
        + ": "
        + str((1 - total_cost / batch)._value)
        + " - "
        + str(params._value)
    )
    iterations += 1

    return total_cost / batch


######################################################################
# Finally, we run the neural network using the Adam optimizer.
#

# Defines the optimizer

iterations = 0

optimizer = qml.AdamOptimizer(stepsize=0.5)
steps = 100
qgrnn_params = list([np.random.randint(-20, 20) / 10 for i in range(0, 10)])

# Optimizes the cost function

for i in range(0, steps):
    qgrnn_params = optimizer.step(cost_function, qgrnn_params)


######################################################################
# Now, let’s prepare the Hamiltonian corresponding to the learned
# parameters:
#

new_ham_matrix = create_hamiltonian_matrix(
    qubit_number, nx.complete_graph(qubit_number), [qgrnn_params[0:6], qgrnn_params[6:10]]
)
print(new_ham_matrix)


######################################################################
# Finally, we can comapre the diagonal elements of the learned Hamiltonian
# (bottom row) to the those of the target Hamiltonian (top row):
#

create_density_plot([np.diag(ham_matrix), np.diag(new_ham_matrix)])


######################################################################
# The similarity of colours indicates that the QGRNN was able to learn the
# target parameters to a very high degree accuracy, which we can verify by 
# looking at the numerical values of the parameters:
#

# Inserts 0s into the target parameters (the target values of edges not contained in the target graph)

matrix_params.insert(1, 0)
matrix_params.insert(5, 0)

# Pints the target and learned parameters

print("Target Parameters: "+str(matrix_params))
print("Learned Parameters: "+str(qgrnn_params))


#######################################################################
#
# ..note:: To increase the accuracy of the model, try downloading this tutorial
#          as a Python file or a Notebook, then decrease the learning 
#          rate, and increase the number of steps the optimizer takes.
#          For the purposes of this minimizing runtime, we kept the number 
#          of steps lower than it should be for optimal performance, so we 
#          strongly reccomend that the motivated reader play around with the  
#          code on their own!
# 

######################################################################
# References
# ~~~~~~~~~~
#
# 1. Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., &
#    Hidary, J. (2019). Quantum Graph Neural Networks. arXiv preprint
#    `arXiv:1909.12264 <https://arxiv.org/abs/1909.12264>`__.
#
