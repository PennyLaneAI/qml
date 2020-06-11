r"""
The Variational Quantum Thermalizer
===================================

*Author: Jack Ceroni (jackceroni@gmail.com)*


This tutorial discusses theory and experiments relating to 
a recently proposed quantum algorithm called the 
`Variational Quantum Thermalizer <https://arxiv.org/abs/1910.02071>`__ (VQT): 
a generalization of the well-know Variational Quantum Eigensolver
(`VQE <https://pennylane.ai/qml/demos/tutorial_vqe.html>`__) to systems 
with non-zero temperatures.

The Idea
--------

Before diving into the simulations, this tutorial first investigates
the mathematical ideas that make the VQT algorithm possible. 
This tutorial assumes some knowledge of variational quantum
algorithms, so for readers unfamiliar with this concept, 
we reccomend taking a look at the VQE tutorials in the QML gallery (like `this
one <https://pennylane.ai/qml/demos/tutorial_vqe.html>`__).

The goal of the VQT is to construct the **thermal state** of
some arbitrary Hamiltonian :math:`H` at a given temperature :math:`T`, 
which is defined as:

.. math::
    \rho_\text{thermal} \ = \ \frac{e^{- H \beta / k_B}}{\text{Tr}(e^{- H \beta / k_B})} \ =
    \ \frac{e^{- H \beta / k_B}}{Z_{\beta}},

where :math:`\beta \ = \ 1/T` and :math:`k_B` is Boltzman's 
constant, which is set to :math:`1` for the remainder of this 
tutorial.

The thermal state is the state of a quantum system
such that the system is in thermal equilibrium with an environment. Knowing
the thermal state allows us to extract information about the system. This is
particularly useful in understanding quantum many-body systems, such as 
Bose-Hubbard models.

The input into the algorithm is an arbitrary Hamiltonian :math:`H`, and
we wish to find :math:`\rho_\text{thermal}`, or more specifically,
the variational parameters of a circuit that prepare a state
that is very close to :math:`\rho_\text{thermal}`.

We begin the VQT processs by picking a "simple" mixed state. 
This initial density matrix is described by a
collection of parameters :math:`\theta`, which determine
the probabilities corresponding to the different computational basis 
states. In this implementation of the algorithm, we use the idea of 
a **factorized latent space** where the initial density matrix is 
completely un-correlated: it is simply a tensor product of 
multiple, :math:`2 \times 2` density matrices that are diagonal 
in the computational basis. If we assign each qubit its own 
diagonal density matrix, we only require one probability, 
:math:`p_i(\theta_i)`, to completely describe the state of the 
:math:`i`-th qubit, which we call :math:`\rho_i`:

.. math:: \rho_{i} \ = \ p_i(\theta_i) |0\rangle \langle 0| \ + \ (1 \ - \ p_i(\theta_i))|1\rangle \langle1|

As a result, for :math:`N` qubits, we only require :math:`N` parameters, 
so the number of parameters scales linearly with qubits, rather 
than exponentially. 

Once we have determined the probability distributions that describe the 
factorized subsystems, we sample from each, which gives us 
a basis state. This basis state is passed through a parametrized ansatz, which we call
:math:`U(\phi)`, and the expectation value of the Hamiltonian with respect to this new 
state is calculated. This process is repeated many times, 
and we take the average of all the calculated expectation values. This gives us
the expectation value of the Hamiltonian with respect to the transformed density matrix, 
:math:`U(\phi)\rho_{\theta}U(\phi)^{\dagger}`.

By combining the energy expectation with the von Neumann
entropy of the transformed state, we define a **free energy cost function**, which is
given by:

.. math::
    \mathcal{L}(\theta, \ \phi) \ = \ \beta \langle \hat{H} \rangle \ - \ S_\theta \ = \
    \beta \ \text{Tr} (\hat{H} \ \rho_{\theta \phi}) \ - \ S_\theta \ = \ \beta \ \text{Tr}( \hat{H} \ \hat{U}(\phi)
    \rho_{\theta} \hat{U}(\phi)^{\dagger} ) \ - \ S_\theta,

where :math:`\rho_\theta` is the initial density matrix, 
and :math:`S_\theta` is the von Neumann entropy of 
:math:`\rho_{\theta \phi}`. It is important to note that the
von Neumann entropy of :math:`\rho_{\theta \phi}` is the same as the von
Neumann entropy of :math:`\rho_{\theta}`, since entropy is invariant
under unitary transformations. This means that we only have to calculate 
the entropy of the simple initial state.

This entire process is then repeated with new :math:`\phi` and 
:math:`\theta` parameters, chosen after each step of the algorithm 
by a classical optimizer, until free energy is minimized. Upon 
minimizing the cost function, we have arrived at the thermal state.
This comes from the fact that the free energy cost function is equivalent
to the relative entropy between :math:`\rho_{\theta \phi}` and the
target thermal state. Relative entropy between two arbitrary states
:math:`\rho_1` and :math:`\rho_2` is defined as

.. math:: D(\rho_1 || \rho_2) \ = \ \text{Tr} (\rho_1 \log \rho_1) \ - \ \text{Tr}(\rho_1 \log \rho_2),

and is minimized (equal to zero) when :math:`\rho_1 \ = \ \rho_2`.
Thus, when the cost function is minimized, it is true that 
:math:`\rho_{\theta \phi} \ = \ \rho_{\text{thermal}}` which means that we have 
found the thermal state.

For a diagramatic representation of how the VQT works, check out Figure 3
from the `original VQT paper <https://arxiv.org/abs/1910.02071>`__.

The 3-Qubit Ising Model on a Line
---------------------------------

We begin by consdering the Ising model on a linear graph, for three
qubits. This is a fairly simple model, and acts as a good test to
see if the VQT is working as it is supposed to.

Numerical Calculation of Target State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we calculate the target state numerically, so that it can be
compared to the state the VQT prepares. 

We can define a few fixed values that are used throughout 
this example:
"""

# Starts by importing all of the necessary dependencies

import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize
import networkx as nx
import seaborn
import itertools

# Defines all necessary variables

beta = 0.5  # Note that beta = 1/T
num_qubits = 3  # Number of qubits being used
qubits = range(num_qubits)

# Defines the device on which the simulation is run

dev = qml.device("default.qubit", wires=len(qubits))


######################################################################
# The model that we are investigating is defined on a linear graph, which we
# construct using ``networkx`` for the purpose of eventually
# using it to construct the Hamiltonian.
#

# Creates the graph of interactions for the Heisenberg grid, then draws it

interaction_graph = nx.Graph()
interaction_graph.add_nodes_from(range(0, num_qubits))
interaction_graph.add_edges_from([(0, 1), (1, 2)])

nx.draw(interaction_graph)


######################################################################
# Next, we implement a method that allows us to calculate
# the matrix form of the Hamiltonian (in the computational basis), for 
# the case of :math:`n` qubits. The Ising model Hamiltonian can be 
# written as:
#
# .. math:: \hat{H} \ = \ \displaystyle\sum_{j} X_{j} X_{j + 1} \ + \ \displaystyle\sum_{i} Z_{i},
#
# where :math:`X_i` and :math:`Z_i` are the Pauli-X and Pauli-Z operations acting 
# on the :math:`i`-th qubit.
#


def create_hamiltonian_matrix(n, graph):

    matrix = np.zeros((2 ** n, 2 ** n))

    for i in graph.edges:
        m = 1
        for j in range(0, n):
            if j == i[0] or j == i[1]:
                m = np.kron(m, qml.PauliX.matrix)
            else:
                m = np.kron(m, np.identity(2))
        matrix = np.add(matrix, m)

    for i in range(0, n):
        m = 1
        for j in range(0, n):
            if j == i:
                m = np.kron(m, qml.PauliZ.matrix)
            else:
                m = np.kron(m, np.identity(2))
        matrix = np.add(matrix, m)

    return matrix


# Constructs the Hamiltonian

ham_matrix = create_hamiltonian_matrix(num_qubits, interaction_graph)
print(ham_matrix)


######################################################################
# We now construct the target thermal state, which is of the form:
#
# .. math:: \rho_{\text{thermal}} \ = \ \frac{e^{-\beta \hat{H}}}{Z_{\beta}}.
#
# All we have to do is take the matrix exponential of the
# Hamiltonian, and then calculate its trace to find the partition function. In
# addition to finding the thermal state, we go one step further and
# also calculate the value of the cost function associated with this
# target state. Below, we write a general method for finding a
# thermal state of some arbitrary Hamiltonian:
#


def create_target(qubit, beta, ham, graph):

    # Calculates the matrix form of the density matrix, by taking the exponential of the Hamiltonian

    h = ham(num_qubits, graph)
    y = -1 * beta * h
    new_matrix = scipy.linalg.expm(np.array(y))
    norm = np.trace(new_matrix)
    final_target = (1 / norm) * new_matrix

    # Calculates the entropy, the expectation value, and the final cost

    entropy = -1 * np.trace(final_target @ scipy.linalg.logm(final_target))
    ev = np.trace(final_target @ h)
    real_cost = beta * np.trace(final_target @ h) - entropy

    # Prints the calculated values

    print("Expectation Value: " + str(ev))
    print("Entropy: " + str(entropy))
    print("Final Cost: " + str(real_cost))

    return final_target


######################################################################
# Finally, the thermal state corresponding to the pre-defined
# Ising Hamiltonian and inverse temperature can be calculated. 
# We also visualize it using the ``seaborn`` library:
#

# Plots the entry-wise absolute value of the final density matrix

final_density_matrix = create_target(num_qubits, beta, create_hamiltonian_matrix, interaction_graph)
seaborn.heatmap(abs(final_density_matrix))


######################################################################
# Variational Quantum Thermalization of the Ising Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Now that we know exactly what the thermal state should look like, we
# attempt to construct it with the VQT. We begin by calculating the
# classical probability distribution, which gives the probabilities
# corresponding to each basis state in the initial, factorized density  
# matrix. We let the probability associated with the :math:`i`-th one-qubit 
# system be given by the sigmoid function:
#
# .. math:: p_{i}(\theta_{i}) \ = \ \frac{e^{\theta_i}}{e^{\theta_i} \ + \ 1}
#
# The motivation behind this choice is that the sigmoid has a
# range of :math:`0` to :math:`1`, meaning we don't need to constrain 
# the parameter values to some subset of :math:`\mathbb{R}`. We can implement 
# this function as:
#

# Creates the probability distribution according to the theta parameters

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)


######################################################################
# From this, we construct a function that calculates the entire 
# initial probability distribution, which is a list of pairs of 
# probabilities that correspond to each one-qubit system
# in the factorized latent space:
#

# Creates the probability distributions for each of the one-qubit systems

def prob_dist(params):

    dist = []
    for i in params:
        dist.append([sigmoid(i), 1 - sigmoid(i)])

    return dist


######################################################################
# Now, we can define the quantum components of the VQT.
# The first step of the circuit is to prepare a given
# basis state, so we write a function that takes a list 
# of bits and returns a quantum circuit that prepares
# the corresponding computational basis state:
#

# Creates the initialization gate

def create_v_gate(prep_state):

    for i in range(0, len(prep_state)):
        if prep_state[i].val == 1:
            qml.PauliX(wires=i)


######################################################################
# Now, we build the parametrized circuit, through which we pass the initial
# states. We use a multi-layered ansatz, where each layer is composed
# of :math:`RX`, :math:`RZ`, and :math:`RY` gates on each qubit, followed
# by exponentiated :math:`CNOT` gates placed between qubits that share an
# edge in the interaction graph of the Ising model. 
# We first define the single-qubit rotations:
#

# Creates the single rotational ansatz

def single_rotation(phi_params, q):

    qml.RZ(phi_params[0], wires=q)
    qml.RY(phi_params[1], wires=q)
    qml.RX(phi_params[2], wires=q)


######################################################################
# Putting this together with the :math:`CNOT` gates, the general
# ansatz is of the form:
#

# Creates the ansatz circuit

def ansatz_circuit(params, qubits, layers, graph, param_number):

    param_number = int(param_number.val)
    number = param_number * num_qubits + len(graph.edges)

    # Partitions the parameters into param lists
    partition = []
    for i in range(0, int((len(params) / number))):
        partition.append(params[number * i : number * (i + 1)])

    for j in range(0, depth):

        # Implements the single qubit rotations
        sq = partition[j][0 : (number - len(graph.edges))]
        for i in qubits:
            single_rotation(sq[int(i.val) * param_number : (int(i.val) + 1) * param_number], int(i.val))

        # Implements the coupling layer of gates
        for count, i in enumerate(graph.edges):
            p = partition[j][(number - len(graph.edges)) : number]
            qml.CRX(p[count], wires=[i[0], i[1]])


######################################################################
# There are a lot of variables floating around in this function. The
# ``param_number`` variable tells us how many unique parameters we
# assign to each qubit, for each application of the single-qubit 
# rotation layer. We multiply this by the number of qubits, to get 
# the total number of single-rotation parameters, and then add the 
# number of edges in the interaction graph, which is the number 
# of unique parameters needed for a layer of the :math:`CNOT` 
# gates, to get the total number of parameters used in one full
# layer of the ansatz. We then repeatedly apply the same parameters 
# over layers of the ansatz, for a given depth.
#
# With all of these components, we define a function 
# that acts as the final quantum circuit, and pass it into a
# QNode:
#

# Defines the depth/parameter number  of the variational circuit

depth = 3
param_number = 3

# Creates the quantum circuit

def quantum_circuit(params, qubits, sample, param_number):

    # Prepares the initial basis state corresponding to the sample

    create_v_gate(sample)

    # Prepares the variational ansatz for the circuit

    ansatz_circuit(params, qubits, depth, interaction_graph, param_number)

    # Calculates the expectation value of the Hamiltonian, with respect to the prepared states

    return qml.expval(qml.Hermitian(ham_matrix, wires=range(num_qubits)))

qnode = qml.QNode(quantum_circuit, dev)

# Tests and draws the QNode

results = qnode(np.ones([12 * depth]), qubits, [1, 0, 1], param_number)
print(qnode.draw())


######################################################################
# There is one more thing we must do before implementing the cost
# function: write a method that calculates the entropy of a 
# state. We take a probability distribution as 
# an input, each entry of which corresponds to the digonal elements of 
# the density matrix of one of the factorized subsystem. We output a list
# of entropies for each subsystem, which we eventually sum together, as
# the entropy of a collection of subsystems is the same as the sum of 
# the entropies of the individual systems.
#

# Calculates the von Neumann entropy of the initial density matrices

def calculate_entropy(distribution):

    total_entropy = []
    for i in distribution:
        total_entropy.append(-1 * i[0] * np.log(i[0]) + -1 * i[1] * np.log(i[1]))

    # Returns an array of the entropy values of the different initial density matrices

    return total_entropy


######################################################################
# Finally, we define the cost function. More specifically, this is an
# **exact** version of the VQT cost function. Instead of sampling from the
# classical probability distribution, we simply calculate the probability
# corresponding to every basis state in the initial density matrix. We then pass 
# each possible basis state through the ansatz, calculating the expectation value. 
# Finally, each probability is multiplied by its corresponding expectation value, 
# and the energy expectation with respect to :math:`\rho_{\theta \phi}` is 
# determined. This is not quite how the VQT would work in the practice. 
# For large systems, when the number of basis states (and thus 
# the size of the probability distribution) scales exponentially, sampling
# is necessary for the algorithm to scale sub-exponentially. For small
# toy-models such as this, the exact form runs faster.
#

def exact_cost(params):

    global iterations

    # Separates the list of parameters

    dist_params = params[0:num_qubits]
    params = params[num_qubits:]

    # Creates the probability distribution

    distribution = prob_dist(dist_params)

    # Generates a list of all computational basis states

    combos = itertools.product([0, 1], repeat=num_qubits)
    s = [list(i) for i in combos]

    # Passes each basis state through the variational circuit and multiplies the calculated energy EV
    # with the associated probability from the distribution

    final_cost = 0
    for i in s:
        result = qnode(params, qubits, i, param_number)
        for j in range(0, len(i)):
            result = result * distribution[j][i[j]]
        final_cost += result

    # Calculates the entropy and the final cost

    entropy = calculate_entropy(distribution)
    cost = beta * final_cost - sum(entropy)

    # Prints the value of the cost function every 100 steps

    if iterations % 100 == 0:
        print("Cost at Step " + str(iterations) + ": " + str(cost))

    iterations += 1

    return cost


######################################################################
# Finally, we optimize the cost function, using the gradient-free COBYLA:
#

# Creates and runs the optimizer

iterations = 0
params = [np.random.randint(-100, 100) / 100 for i in range(0, (12 * depth) + num_qubits)]

out = minimize(exact_cost, x0=params, method="COBYLA", options={"maxiter": 2000})
params = out["x"]
print(out)


######################################################################
# With the optimal parameters we now prepare the state to which
# they correspond, to see how close the prepared state is to the target
# state:
#

def prepare_state(params, device):

    # Initializes the density matrix

    final_density_matrix = np.zeros((2 ** num_qubits, 2 ** num_qubits))

    # Prepares the optimal parameters, creates the distribution and the bitstrings

    dist_params = params[0:num_qubits]
    unitary_params = params[num_qubits:]

    distribution = prob_dist(dist_params)

    combos = itertools.product([0, 1], repeat=num_qubits)
    s = [list(i) for i in combos]

    # Runs the circuit in the case of the optimal parameters, for each bitstring,
    # and adds the result to the final density matrix.

    for i in s:
        qnode(unitary_params, qubits, i, param_number)
        state = device.state
        for j in range(0, len(i)):
            state = np.sqrt(distribution[j][i[j]]) * state
        final_density_matrix = np.add(final_density_matrix, np.outer(state, np.conj(state)))

    # Returns the prepared density matrix

    return final_density_matrix


prepared_density_matrix = prepare_state(params, dev)


######################################################################
# To asess how "close together" the prepared and target states
# are, we use the trace distance, which is a metric (a
# "distance function" with certain properties) on the space of density
# matrices defined by:
#
# .. math:: T(\rho, \ \sigma) \ = \ \frac{1}{2} \text{Tr} \sqrt{(\rho \ - \ \sigma)^{\dagger} (\rho \ - \ \sigma)}
#
# We can implement this as a function, and compute the trace distance
# between the target and prepared states:
#

# Finds the trace distance between two density matrices

def trace_distance(one, two):

    return 0.5 * np.trace(np.absolute(np.add(one, -1 * two)))

# Calculates the trace distance between the prepared and target states

print("Final Trace Distance: " + str(trace_distance(prepared_density_matrix, final_density_matrix)))


######################################################################
# These results are good! A trace distance close to :math:`0` means that the
# states are close together, thus we prepared a decent
# approximation of the thermal state. If you prefer a visual
# representation, the absolute value of the prepared state can 
# be plotted as a heatmap:
#

seaborn.heatmap(abs(prepared_density_matrix))


######################################################################
# Then, we compare it to the absolute value of the target state:
#

seaborn.heatmap(abs(final_density_matrix))


######################################################################
# As you can see, the two images are not completely the same, but there is
# definitely some resemblance between them!
#


######################################################################
# The 4-Qubit Heisenberg Model on a Cycle Graph
# ---------------------------------------------
#
# Let's look at one more example of the VQT in action, this time, for a
# slightly more complicated system: the Heisenberg model on a 4-qubit cycle
# graph.
#


######################################################################
# Numerical Calculation of Target State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We start by defining fixed values:
#

# Defines all necessary variables

beta = 1  # Note that beta = 1/T
num_qubits = 4
qubits = range(num_qubits)
depth = 2

# Defines the device on which the simulation is run

dev2 = qml.device("default.qubit", wires=len(qubits))


######################################################################
# This model is defined on a cycle graph, which we initialize:
#

# Creates the graph of interactions for the Heisenberg grid, then draws it

interaction_graph = nx.Graph()
interaction_graph.add_nodes_from(range(0, num_qubits))
interaction_graph.add_edges_from([(0, 1), (2, 3), (0, 2), (1, 3)])

nx.draw(interaction_graph)


######################################################################
# Recall that the two-dimensional Heiseberg model Hamiltonian can be
# written as:
#
# .. math:: \hat{H} \ = \ \displaystyle\sum_{(i, j) \in E} X_i X_{j} \ + \ Z_i Z_{j} \ + \ Y_i Y_{j},
#
# with :math:`X_i`, :math:`Y_i`, and :math:`Z_i` being the Pauli-X, 
# Pauli-Y and Pauli-Z on the :math:`i`-th qubit.

######################################################################
# With this knowledge, we can define the new Hamiltonian matrix:
#

# Creates the target Hamiltonian matrix

def create_hamiltonian_matrix(n, graph):

    matrix = np.zeros((2 ** n, 2 ** n))

    for i in graph.edges:
        m = 1
        for j in range(0, n):
            if j == i[0] or j == i[1]:
                m = np.kron(m, qml.PauliX.matrix)
            else:
                m = np.kron(m, np.identity(2))
        matrix = np.add(matrix, m)

    for i in graph.edges:
        m = 1
        for j in range(0, n):
            if j == i[0] or j == i[1]:
                m = np.kron(m, qml.PauliY.matrix)
            else:
                m = np.kron(m, np.identity(2))
        matrix = np.add(matrix, m)

    for i in graph.edges:
        m = 1
        for j in range(0, n):
            if j == i[0] or j == i[1]:
                m = np.kron(m, qml.PauliZ.matrix)
            else:
                m = np.kron(m, np.identity(2))
        matrix = np.add(matrix, m)

    return matrix


ham_matrix = create_hamiltonian_matrix(num_qubits, interaction_graph)
print(ham_matrix)


######################################################################
# We then calculate and plot the thermal state at the inverse temperature 
# defined above:
#

# Plots the absolute value of the target density matrix

final_density_matrix = create_target(num_qubits, beta, create_hamiltonian_matrix, interaction_graph)
seaborn.heatmap(abs(final_density_matrix))


######################################################################
# Variational Quantum Thermalization of the Ising Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# To find the thermal state using the VQT, we use the same form of 
# the initial state, ansatz, and cost function as above. All we have 
# to do is re-define the qnode, since we are now using a device with 
# :math:`4` qubits rather than :math:`3`:
#

# Defines the new QNode

qnode = qml.QNode(quantum_circuit, dev2)


######################################################################
# We then run the optimizer (using COBYLA again):
#

# Creates the optimizer

iterations = 0

params = [np.random.randint(-100, 100) / 100 for i in range(0, (16 * depth) + num_qubits)]
out = minimize(exact_cost, x0=params, method="COBYLA", options={"maxiter": 2000})
params = out["x"]
print(out)


######################################################################
# With the optimal parameters, we can post-process the data. We start by
# calculating the matrix form of the prepared density matrix:
#

# Prepares the density matrix

prepared_density_matrix = prepare_state(params, dev2)


######################################################################
# We then calculate the trace distance between the prepared and 
# target states:
#

print("Final Trace Distance: " + str(trace_distance(prepared_density_matrix, final_density_matrix)))


######################################################################
# This is pretty good, but it could be better (most likely with a deeper
# ansatz and a more sophisticated optimizer, but to keep execution time
# relatively short, we don't go down those avenues in this tutorial).
# To end off, we visualize the absolute values of the target and 
# prepared density matrices. We begin with the prepared matrix:
#

seaborn.heatmap(abs(prepared_density_matrix))


######################################################################
# Then, we print the target:
#

seaborn.heatmap(abs(final_density_matrix))


######################################################################
# References
# ----------
#
# 1. Verdon, G., Marks, J., Nanda, S., Leichenauer, S., & Hidary, J.
#    (2019). Quantum Hamiltonian-Based Models and the Variational Quantum
#    Thermalizer Algorithm. arXiv preprint
#    `arXiv:1910.02071 <https://arxiv.org/abs/1910.02071>`__.
#
