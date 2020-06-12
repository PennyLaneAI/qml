"""
The Variational Quantum Thermalizer
-----------------------------------

"""

# Starts by importing all of the necessary dependencies

import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
from numpy import array
import scipy
from scipy.optimize import minimize
import networkx as nx
import seaborn
import itertools
from pennylane.templates.state_preparations import BasisStatePreparation
from pennylane.templates.layers import BasicEntanglerLayers


######################################################################
# The Idea
# ~~~~~~~~
#


######################################################################
# This tutorial discusses theory and experiments relating to a recently
# proposed quantum algorithm called the `Variational Quantum
# Thermalizer <https://arxiv.org/abs/1910.02071>`__ (VQT): a
# generalization of the well-know `Variational Quantum
# Eigensolver <https://pennylane.ai/qml/demos/tutorial_vqe.html>`__ (VQE)
# to systems with non-zero temperatures. The goal of the VQT is to prepare
# the **thermal state** of a given Hamiltonian :math:`\hat{H}` at
# temperature :math:`T`, which is defined as:
#
# .. math:: \rho_\text{thermal} \ = \ \frac{e^{- H \beta}}{\text{Tr}(e^{- H \beta})} \ = \ \frac{e^{- H \beta}}{Z_{\beta}}
#
# where :math:`\beta \ = \ 1/T`. The thermal state is a **mixed state**,
# which means that it is described by an ensemble of pure states, each
# corresponding to some probability in a classical probability
# distribution. Since we are attempting to learn a mixed state, we must
# deviate from the standard variational method of passing a pure state
# through an ansatz, and minimizing the energy expectation.
#
# The VQT begins with an initial density matrix, described by a
# probability distribution parametrized by some collection of parameters
# :math:`\theta`, and an ensemble of pure states,
# :math:`\{|\psi_i\rangle\}`. We let :math:`p_i(\theta_i)` be the
# probability corresponding to the :math:`i`-th pure state. We sample from
# this probability distribution to get some pure state
# :math:`|\psi_k\rangle`, which we pass through a parametrized circuit,
# :math:`U(\phi)`. From the results of this circuit, we then calculate
# :math:`\langle \psi_k | U^{\dagger}(\phi) \hat{H} U(\phi) |\psi_k\rangle`.
# Repeating this process multiple times and taking the average of these
# expectation values gives us the the expectation value of :math:`\hat{H}`
# with respect to :math:`U \rho_{\theta} U^{\dagger}`. This new density
# matrix is analogous to the resulting pure state after passage through an
# ansatz in VQE.
#
# Arguably, the most important part of a variational circuit is its cost
# function, which we atempt to minimize. In VQE, we generally try to
# minimize :math:`\langle \psi(\theta) | \hat{H} | \psi(\theta) \rangle`
# which, upon minimization, gives us a parametrized circuit that prepares
# a good approximation to the ground state of :math:`\hat{H}`. In the VQT,
# the goal is to arrive at a parametrized probability distribution, and a
# parametrized ansatz, that generate a good approximation to the thermal
# state, which in general will involve more than calculating the energy
# expectation value. Luckily, we know that the thermal state of
# :math:`\hat{H}` minimizes the following free-energy cost function:
#
# .. math:: \mathcal{L}(\theta, \ \phi) \ = \ \beta \ \text{Tr}( \hat{H} \ \hat{U}(\phi) \rho_{\theta} \hat{U}(\phi)^{\dagger} ) \ - \ S_\theta
#
# where :math:`S_{\theta}` is the von Neumann entropy of
# :math:`U \rho_{\theta} U^{\dagger}`, which is the same as the von
# Neumann entropy of :math:`\rho_{\theta}` due to invariance of entropy
# under unitary transformations. This cost function is minimized when
# :math:`\hat{U}(\phi) \rho_{\theta} \hat{U}(\phi)^{\dagger} \ = \ \rho_{\text{thermal}}`,
# so similarly to VQE, we minimize it with a classical optimizer to obtain
# the target parameters, and thus the target state.
#
# All together, the outlined processes give us a general protocal to
# generate thermal states. Throughout the Notebook, more nuances in
# relation to the particular classical and quantum components will be
# mentioned as they are implemented.
#


######################################################################
# Simulating the VQT for a 4-Qubit Heisenberg Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# In this tutorial, we will be simulating the 4-qubit Heisenberg model,
# which is defined as:
#
# .. math:: \hat{H} \ = \ \displaystyle\sum_{(i, j) \in E} X_i X_{j} \ + \ Z_i Z_{j} \ + \ Y_i Y_{j}
#
# where :math:`X_i`, :math:`Y_i` and :math:`Z_i` are the Pauli gates
# acting on the :math:`i`-th qubit. In addition, :math:`E` is the set of
# edges in the graph :math:`G \ = \ (V, \ E)` describing the interactions
# between the qubits. In this tutorial, we define the interaction graph to
# be the cycle graph:
#

# Creates the graph of interactions for the Heisenberg grid, then draws it

interaction_graph = nx.Graph()
interaction_graph.add_nodes_from(range(0, N))
interaction_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

nx.draw(interaction_graph)


######################################################################
# With this, we can calculate the matrix representation of the Heisenberg
# Hamiltonian, in the computational basis:
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


ham_matrix = create_hamiltonian_matrix(N, interaction_graph)
print(ham_matrix)


######################################################################
# With this done, we can construct the VQT. We begin by defining some
# fixed variables that are used throughout the simulation:
#

# Defines all necessary variables

beta = 1  # beta = 1/T
N = 4  # Number of qubits


######################################################################
# The first step of the VQT is to create the initial density matrix,
# :math:`\rho_\theta`. In this tutorial, we let :math:`\rho_\theta` be
# **factorized**, meaning that it can be written as an uncorrelated tensor
# product of :math:`4` one-qubit, density matrices that are diagonal in
# the computational basis. The motivation for this choice is due to the
# fact that in this factorized model, the number of :math:`\theta_i`
# parameters needed to describe :math:`\rho_\theta` versus the number of
# qubits scales linearly rather than exponentially, as for each one-qubit
# system described by :math:`\rho_\theta^i`, we will have:
#
# .. math:: \rho_{\theta}^{i} \ = \ p_i(\theta_i) |0\rangle \langle 0| \ + \ (1 \ - \ p_i(\theta_i))|1\rangle \langle1|
#
# From here, all we have to do is define :math:`p_i(\theta_i)`, which we
# choose to be the sigmoid:
#
# .. math:: p_{i}(\theta_{i}) \ = \ \frac{e^{\theta_i}}{e^{\theta_i} \ + \ 1}
#

# Creates the probability distribution according to the theta parameters


def sigmoid(x):

    return np.exp(x) / (np.exp(x) + 1)


######################################################################
# This is a natural choice for probability function, as it has a range of
# :math:`[0, \ 1]`, meaning that we don’t need to restrict the domain of
# :math:`\theta_i` to some subset of the real number. With the probability
# function defined, we can write a method that gives us the diagonal
# elements of each one-qubit density matrix, for some collection
# :math:`\theta`:
#

# Creates the probability distributions for each of the one-qubit systems


def prob_dist(params):

    dist = []
    for i in params:
        dist.append([sigmoid(i), 1 - sigmoid(i)])

    return dist


######################################################################
# With this done, we can move on to defining the ansatz circuit,
# :math:`U(\phi)`. The ansatz must begin by preparing some arbitrary
# computational basis state sampled from the initial density matrix. This
# is easily implemented in PennyLane with the ``BasisStatePreparation``
# template. The next step is to build the rotational and coupling layers
# used in the ansatz. The rotation layer will simply be :math:`RX`,
# :math:`RY`, and :math:`RZ` gates applied to each qubit.
#

# Creates the single rotational ansatz


def single_rotation(phi_params, qubits):

    rotations = ["Z", "Y", "X"]
    for i in range(0, len(rotations)):
        qml.templates.embeddings.AngleEmbedding(phi_params[i], wires=qubits, rotation=rotations[i])


######################################################################
# Notice the use of the ``AngleEmbeddings`` function, which allows us to
# easily pass parameters into rotational layers. To construct the general
# ansatz, we combine the method we have just defined with a collection of
# parametrized coupling gates, placed between qubits that share an edge in
# the interaction graph. In addition, we define the depth of the ansatz,
# and the device on which the simulations are run;
#

# Defines the depth of the variational circuit and the device on which

depth = 4
dev = qml.device("default.qubit", wires=N)

# Creates the quantum circuit


def quantum_circuit(rotation_params, coupling_params, sample=None):

    # Prepares the initial basis state corresponding to the sample
    BasisStatePreparation(sample, wires=range(N))

    # Prepares the variational ansatz for the circuit
    for i in range(0, depth):
        single_rotation(rotation_params[i], range(N))
        qml.broadcast(
            unitary=qml.CRX, pattern="ring", wires=range(N), parameters=coupling_params[i]
        )

    # Calculates the expectation value of the Hamiltonian, with respect to the prepared states
    return qml.expval(qml.Hermitian(ham_matrix, wires=range(N)))


# Constructs the QNode
qnode = qml.QNode(quantum_circuit, dev)


######################################################################
# We can get an idea of what this circuit looks like by printing out a
# test circuit:
#

# Draws the QNode

results = qnode(
    [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]] for i in range(0, depth)],
    [[1, 1, 1, 1] for i in range(0, depth)],
    sample=[1, 0, 1, 0],
)
print(qnode.draw())


######################################################################
# Recall that the final cost function depends not only on the expectation
# value of the Hamiltonian, but also the von Neumann entropy of our state,
# which is determined by the collection of :math:`p_i(\theta_i)`\ s. Since
# the entropy of a collection of multiple uncorrelated susbsystems is the
# same as the sum of the individual values of entropy for each subsystem,
# we can sum the entropy of each one-qubit system in the factorized space
# to get the total:
#

# Calculate the Von Neumann entropy of the initial density matrices


def calculate_entropy(distribution):

    total_entropy = 0
    for i in distribution:
        total_entropy += -1 * i[0] * np.log(i[0]) + -1 * i[1] * np.log(i[1])

    # Returns an array of the entropy values of the different initial density matrices

    return total_entropy


######################################################################
# Finally, we combine the ansatz and the entropy function to get the cost
# function. In this tutorial, we deviate slightly from how VQT would be
# performed in practice. Instead of sampling from the probability
# distribution describing the initial mixed state, we use the ansatz to
# calculate
# :math:`\langle x_i | U^{\dagger}(\phi) \hat{H} U(\phi) |x_i\rangle` for
# each basis state :math:`|x_i\rangle`. We then multiply each of these
# expectation values by their corresponding :math:`(\rho_\theta)_{ii}`,
# which is exactly the probability of sampling :math:`|x_i\rangle` from
# the distribution. Summing each of these terms together gives us the
# expected value of the Hamiltonian with respect to the transformed
# density matrices. In the case of this small, simple model, exact
# calculations reduce the number of circuit executions, and thus total
# execution time.
#
# You may have noticed previously that the “structure” of the list of
# parameters passed into the ansatz is very complicated. We write a
# general function that takes a one-dimensional list, and converts it into
# the nestled list structure that can be inputted into the ansatz:
#


def convert_list(params):

    # Separates the list of parameters
    dist_params = params[0:N]
    ansatz_params_1 = params[N : (depth + 1) * N]
    ansatz_params_2 = params[(depth + 1) * N :]

    coupling = np.split(ansatz_params_1, depth)

    # Partitions the parameters into multiple lists
    split = np.split(ansatz_params_2, depth)
    rotation = []
    for i in split:
        rotation.append(np.split(i, 3))

    ansatz_params = [rotation, coupling]

    return [dist_params, ansatz_params]


######################################################################
# We then pass this function, along with the ansatz and the entropy
# function into the final cost function:
#


def exact_cost(params):

    global iterations

    # Transforms the parameter list
    parameters = convert_list(params)
    dist_params = parameters[0]
    ansatz_params = parameters[1]

    # Creates the probability distribution
    distribution = prob_dist(dist_params)

    # Generates a list of all computational basis states, of our qubit system
    combos = itertools.product([0, 1], repeat=N)
    s = [list(i) for i in combos]

    # Passes each basis state through the variational circuit and multiplies the calculated energy EV with the associated probability from the distribution
    final_cost = 0
    for i in s:
        result = qnode(ansatz_params[0], ansatz_params[1], sample=i)
        for j in range(0, len(i)):
            result = result * distribution[j][i[j]]
        final_cost += result

    # Calculates the entropy and the final cost function
    entropy = calculate_entropy(distribution)
    final_final_cost = beta * final_cost - entropy

    if iterations % 20 == 0:
        print("Cost at Step " + str(iterations) + ": " + str(final_final_cost))

    iterations += 1

    return final_final_cost


######################################################################
# The last step is to define the optimizer, and execute the optimization
# method:
#

# Creates the optimizer

iterations = 0

params = [np.random.randint(-300, 300) / 100 for i in range(0, (N * (1 + depth * 4)))]
out = minimize(exact_cost, x0=params, method="COBYLA", options={"maxiter": 1000})
out_params = out["x"]
print(out)


######################################################################
# We can now check to see how well our optimization method performed. We
# write a function that re-constructs the transformed density density
# matrix of some initial state, with respect to some lists of
# :math:`\theta` and :math:`\phi` parameters:
#


def prepare_state(params, device):

    # Initializes the density matrix

    final_density_matrix = np.zeros((2 ** N, 2 ** N))

    # Prepares the optimal parameters, creates the distribution and the bitstrings
    parameters = convert_list(params)
    dist_params = parameters[0]
    unitary_params = parameters[1]

    distribution = prob_dist(dist_params)

    combos = itertools.product([0, 1], repeat=N)
    s = [list(i) for i in combos]

    # Runs the circuit in the case of the optimal parameters, for each bitstring, and adds the result to the final density matrix

    for i in s:
        qnode(unitary_params[0], unitary_params[1], sample=i)
        state = device.state
        for j in range(0, len(i)):
            state = np.sqrt(distribution[j][i[j]]) * state
        final_density_matrix = np.add(final_density_matrix, np.outer(state, np.conj(state)))

    return final_density_matrix


######################################################################
# We then re-construct the state prepared by the VQT:
#

prep_density_matrix = prepare_state(out_params, dev)
print(prep_density_matrix)


######################################################################
# If you prefer a visual rperesentation, we can plot a heatmap of the
# absolute value of the density matrix as well:
#

seaborn.heatmap(abs(prep_density_matrix))


######################################################################
# To verify that we have in fact prepared a good approximation of the
# thermal state, let’s calculate it numerically by taking the matrix
# exponential of the Heisenberg Hamiltonian, as was outlined in the first
# part of the tutorial.
#

# Creates the target density matrix


def create_target(qubit, beta, ham, graph):

    # Calculates the matrix form of the density matrix, by taking the exponential of the Hamiltonian

    h = ham(qubit, graph)
    y = -1 * float(beta) * h
    new_matrix = scipy.linalg.expm(np.array(y))
    norm = np.trace(new_matrix)
    final_target = (1 / norm) * new_matrix

    return final_target


target_density_matrix = create_target(N, beta, create_hamiltonian_matrix, interaction_graph)


######################################################################
# Finally, we can plot a heatmap of the target Hamiltonian:
#

# Plots the final density matrix

seaborn.heatmap(abs(final_density_matrix))


######################################################################
# The two images look very similar, which suggests that we have
# constructed a good approximation of the thermal state! Alternatively, if
# you prefer a more quantitative measure of similarity, we can calculate
# the trace distance between the two density matrices, which is defined
# as:
#
# .. math:: T(\rho, \ \sigma) \ = \ \frac{1}{2} \text{Tr} \sqrt{(\rho \ - \ \sigma)^{\dagger} (\rho \ - \ \sigma)}
#
# and is a metric (a “distance function”) on the space of density
# matrices:
#

# Finds the trace distance between two density matrices


def trace_distance(one, two):

    return 0.5 * np.trace(np.absolute(np.add(one, -1 * two)))


print("Trace Distance: " + str(trace_distance(final_density_matrix_2, final_density_matrix)))


######################################################################
# The closer to zero, the more similar the two states are. Thus, we have
# numerical proof that we have found an approximation of the thermal state
# of :math:`H` with the VQT!
#


######################################################################
# References
# ----------
#
# 1. Verdon, G., Marks, J., Nanda, S., Leichenauer, S., & Hidary, J.
#    (2019). Quantum Hamiltonian-Based Models and the Variational Quantum
#    Thermalizer Algorithm. arXiv preprint
#    `arXiv:1910.02071 <https://arxiv.org/abs/1910.02071>`__.
#
