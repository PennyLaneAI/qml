r"""
The Variational Quantum Thermalizer
===================================

*Author: Jack Ceroni*


This tutorial will discuss theory and
experiments relating to a recently proposed quantum algorithm called the
`Variational Quantum Thermalizer <https://arxiv.org/abs/1910.02071>`__ (VQT). 
The VQT algorithm is able to use a variational approach to 
reconstruct the thermal state of a Hamiltonian, at a given temperature. 
Interestingly enough, the original VQT paper demonstrates that 
the VQT is actually a generalization of the well-know Variational Quantum Eigensolver 
(`VQE <https://pennylane.ai/qml/demos/tutorial_vqe.html>`__), and as the effective 
"temperature" approaches zero, the VQT algorithm converges to the standard VQE.

The Idea
--------

Before constructing the simulations, this tutorial first investigates
the mathematical and physical theory that makes
the VQT algorithm possible. For more background on variational quantum
algorithms and why VQE works, check out the other tutorials in
the QML gallery (like `this
one <https://pennylane.ai/qml/demos/tutorial_vqe.html>`__).

The goal of the VQT is to construct a **thermal state**, which is
defined as:

.. math::
    \rho_\text{thermal} \ = \ \frac{e^{- H \beta / k_B}}{\text{Tr}(e^{- H \beta / k_B})} \ =
    \ \frac{e^{- H \beta / k_B}}{Z_{\beta}},

where :math:`H` is the Hamiltonian of the system,
:math:`\beta \ = \ 1/T`, where :math:`T` is the temperature of our
system, and :math:`k_B` is Boltzman's constant, which we will set to
:math:`1` for the remainder of this demonstration.

The thermal state is the state of a quantum system
such that the system is in thermal equilibrium with an environment. Knowing
the thermal state allows us to extract information about the system. This is
particularly useful in understanding quantum many-body systems, such as Bose-Hubbard models.

The input into our algorithm is an arbitrary Hamiltonian :math:`H`, and
our goal is to find :math:`\rho_\text{thermal}`, or more specifically
the variational parameters of a circuit that prepare a state
that is very close to :math:`\rho_\text{thermal}`.

In order to do this, we pick a "simple" mixed state to begin the
process. This initial density matrix is described by a
collection of parameters :math:`\boldsymbol\theta`, which describe
the probabilities corresponding to different pure states. In this
implementation of the algorithm, we use the idea of a **factorized
latent space** where the initial density matrix describing the quantum
system in completely un-correlated. It is simply a tensor product of
multiple :math:`2 \times 2`, density matrices that are diagonal 
in the computational basis. This makes the algorithm scale linearly, rather than exponentially.
If we describe each qubit by its own, diagonal density matrix, we only require
one probability, :math:`p_i(\theta_i)', to completely describe the state of the :math:`i`-th qubit, so for $N$
qubits, we only require :math:`N` parameters. More concretely, the state of the :math:`i`-th 
qubit is described by:

.. math:: \rho_{i} \ = \ p_i(\theta_i) |0\rangle \langle 0| \ + \ (1 \ - \ p_i(\theta_i))|1\rangle \langle1|

We then sample from each one-qubit system, which gives us 
a basis state. This basis state is passed through a parametrized ansatz, which we call
:math:`U(\phi)`, and the expectation value of the Hamiltonian with respect to this new 
state is subsequently calculated. This process is repeated many times, 
and we take the average of all the calculated expectation values, which gives us
the expectation value of the Hamiltonian, with respect to the transformed density matrix, 
:math:`U(\phi)\rho_{\theta}U(\phi)^{\dagger}`.

Combining the energy expectation with the Von Neumann
entropy of the state, we define **free energy cost function**, which is
given by:

.. math::
    \mathcal{L}(\theta, \ \phi) \ = \ \beta \langle \hat{H} \rangle \ - \ S_\theta \ = \
    \beta \ \text{Tr} (\hat{H} \ \rho_{\theta \phi}) \ - \ S_\theta \ = \ \beta \ \text{Tr}( \hat{H} \ \hat{U}(\phi)
    \rho_{\theta} \hat{U}(\phi)^{\dagger} ) \ - \ S_\theta,

where :math:`\rho_\theta` is the initial density matrix, :math:`U(\phi)`
is the paramterized ansatz, and :math:`S_\theta` is the von Neumann
entropy of :math:`\rho_{\theta \phi}`. It is important to note that the
von Neumann entropy of :math:`\rho_{\theta \phi}` is the same as the von
Neumann entropy of :math:`\rho_{\theta}`, since entropy is invariant
under unitary transformations.

The algorithm is repeated with new parameters until we minimize free
energy, with new parameters being chosen by a classical optimizer after each step of the algorithm.
Once we have done this, we have arrived at the thermal state.
This comes from the fact that the free energy cost function is equivalent
to the relative entropy between :math:`\rho_{\theta \phi}` and the
target thermal state. Relative entropy between two arbitrary states
:math:`\rho_1` and :math:`\rho_2` is defined as:

.. math:: D(\rho_1 || \rho_2) \ = \ \text{Tr} (\rho_1 \log \rho_1) \ - \ \text{Tr}(\rho_1 \log \rho_2)

Relative entropy is minimized (equal to zero) when :math:`\rho_1 \ = \ \rho_2`.
Thus, when the cost function is minimized, it is true that 
:math:`\rho_{\theta \phi} \ = \ \rho_{\text{Thermal}}` which means that the 
thermal state has been learned.

For a diagramatic representation of how this whole process works, check out Figure 3
from the `original VQT paper <https://arxiv.org/abs/1910.02071>`__.

The 3-Qubit Ising Model on a Line
---------------------------------

We begin by consdering the Ising model on a linear graph, for three
qubits. This is a fairly simple model, and will act as a good test to
see if the VQT is working as it is supposed to.

Numerical Calculation of Target State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we calculate the target state numerically, so that it can be
compared to the state our circuit prepares. We define a few
fixed values that we will use throughout this example:
"""

# Start by importing all of the necessary dependencies

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
# will construct using ``networkx`` for the purpose of eventually
# constructing the Hamiltonian.
#

# Creates the graph of interactions for the Heisenberg grid, then draws it

interaction_graph = nx.Graph()
interaction_graph.add_nodes_from(range(0, num_qubits))
interaction_graph.add_edges_from([(0, 1), (1, 2)])

nx.draw(interaction_graph)


######################################################################
# Next, we implement a method that allows us to calculate
# the matrix form of the Hamiltonian (in the computational-basis). The Ising
# model Hamiltonian can be written as:
#
# .. math:: \hat{H} \ = \ \displaystyle\sum_{j} X_{j} X_{j + 1} \ + \ \displaystyle\sum_{i} Z_{i}
#
# Where :math:`X_i` and :math:`Z_i` are the Pauli-X and Pauli-Z gates acting on the :math:`i`-th 
# qubit. We can write this as a function that returns the :math:`n`-qubit matrix
# form of the Ising model Hamiltonian:
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
# All that is left to do is construct the target
# thermal state, which is of the form:
#
# .. math:: \rho_{\text{thermal}} \ = \ \frac{e^{-\beta \hat{H}}}{Z_{\beta}}.
#
# Thus, we simply have to take the matrix exponential of the
# Hamiltonian. The partition function can be found by simply taking the
# trace of the numerator (as it simply acts as a normalization factor). In
# addition to finding the thermal state, let's go one step further and
# also calculate the value of the cost function associated with this
# target state.
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
# Finally, we can calculate the thermal state corresponding to the
# Hamiltonian and inverse temperature, and visualize it using the
# ``seaborn`` library:
#

# Plots the absolute value of the final density matrix

final_density_matrix = create_target(num_qubits, beta, create_hamiltonian_matrix, interaction_graph)
seaborn.heatmap(abs(final_density_matrix))


######################################################################
# Variational Quantum Thermalization of the Ising Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Now that we know exactly what our thermal state should look like, let's
# attempt to construct it with the VQT. Let's begin by constructing the
# classical probability distribution, which gives us the probabilities
# corresponding to each basis state in the expansion of our density
# matrix. As was discussed earlier, we will be using the
# factorized latent space model. We will let the probability
# associated with the :math:`i`-th one-qubit system be:
#
# .. math:: p_{i}(\theta_{i}) \ = \ \frac{e^{\theta_i}}{e^{\theta_i} \ + \ 1}
#
# The motivation behind this choice comes from the fact that this function has a
# range of :math:`0` to :math:`1`. This means that we don't need to constrain 
# the values of our parameters to some finite domain. In addition, this
# function is called a sigmoid, and is a common choice as an activation
# function in neural networks. We can implement the sigmoid as:
#

# Creates the probability distribution according to the theta parameters

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)


######################################################################
# From this, we can construct a function that calculates the entire 
# probability distribution, which will be a
# list of pairs of probabilities that correspond to each one-qubit system
# in the factorized latent space:
#

# Creates the probability distributions for each of the one-qubit systems

def prob_dist(params):

    dist = []
    for i in params:
        dist.append([sigmoid(i), 1 - sigmoid(i)])

    return dist


######################################################################
# Now, we can define the quantum parts of our circuit.
# Befor any qubit register is passed through the variational circuit, we
# must prepare it in a given basis state. Thus, we can write a function
# that takes a list of bits, and returns a quantum circuit that prepares
# the corresponding computational basis state:
#

# Creates the initialization unitary for each of the computational basis states

def create_v_gate(prep_state):

    for i in range(0, len(prep_state)):
        if prep_state[i].val == 1:
            qml.PauliX(wires=i)


######################################################################
# All that is left to do before we construct the cost function is to
# construct the parametrized circuit, through which we pass our initial
# states. We will use a multi-layered ansatz, where each layer is composed
# of :math:`RX`, :math:`RZ`, and. :math:`RY` gates on each qubit, followed
# by exponentiated :math:`CNOT` gates placed between qubits that share an
# edge in the interaction graph. Our general single-qubit rotations can be
# implemented as:
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
# ``param_number`` variable simply tells us how many unique parameters we
# assign to each application of the single-qubit rotation layer. We
# multiply this by the number of qubits, to get the total number of
# single-rotation parameters, and then add the number of edges in the
# interaction graph, which is the number of unique parameters
# needed for the :math:`CNOT` gates. 
#
# With all of these components, we can define a function 
# that acts as our final quantum circuit, and pass it into a
# QNode:
#

# Defines the depth of the variational circuit

depth = 3

# Creates the quantum circuit

def quantum_circuit(params, qubits, sample, param_number):

    # Prepares the initial basis state corresponding to the sample

    create_v_gate(sample)

    # Prepares the variational ansatz for the circuit

    ansatz_circuit(params, qubits, depth, interaction_graph, param_number)

    # Calculates the expectation value of the Hamiltonian, with respect to the preparred states

    return qml.expval(qml.Hermitian(ham_matrix, wires=range(num_qubits)))

qnode = qml.QNode(quantum_circuit, dev)

# Tests and draws the QNode

results = qnode(np.ones([12 * depth]), qubits, [1, 0, 1, 0], 3)
print(qnode.draw())


######################################################################
# There is one more thing we must do before implementing the cost
# function: write a method that calculates the entropy of a
# state. We will take a probability distribution as
# our argument, each entry of which corresponds to the digonal elements of
# the density matrix of a one-qubit subsystems. The entropy of a collection of
# subsystems is the same as the sum of the entropies of the individual
# systems, so we get:
#

# Calculate the Von Neumann entropy of the initial density matrices

def calculate_entropy(distribution):

    total_entropy = []
    for i in distribution:
        total_entropy.append(-1 * i[0] * np.log(i[0]) + -1 * i[1] * np.log(i[1]))

    # Returns an array of the entropy values of the different initial density matrices

    return total_entropy


######################################################################
# Finally, we define the cost function. More specifically, this is an
# **exact** version of the VQT cost function. Instead of sampling from our
# classical probability distribution, we simply calculate the probability
# corresponding to every basis state, and thus calculate the energy
# expectation exactly for each iteration of the algorithm. 
# This is not exactly how the VQT would work in the practice, 
# for large systems. When the number of basis states (and thus 
# the size of the probability distribution) scales exponentially, sampling
# is necessary to scale the algorithm sub-exponentially. For small
# toy-models such as this, the exact form runs faster:
#

def exact_cost(params):

    global iterations

    # Separates the list of parameters

    dist_params = params[0:num_qubits]
    params = params[num_qubits:]

    # Creates the probability distribution

    distribution = prob_dist(dist_params)

    # Generates a list of all computational basis states, of our qubit system

    combos = itertools.product([0, 1], repeat=num_qubits)
    s = [list(i) for i in combos]

    # Passes each basis state through the variational circuit and multiplis the calculated energy EV
    # with the associated probability from the distribution

    final_cost = 0
    for i in s:
        result = qnode(params, qubits, i, 3)
        for j in range(0, len(i)):
            result = result * distribution[j][i[j]]
        final_cost += result

    # Calculates the entropy and the final cost function

    entropy = calculate_entropy(distribution)
    final_final_cost = beta * final_cost - sum(entropy)

    # Prints the value of the cost function every 100 steps

    if iterations % 100 == 0:
        print("Cost at Step " + str(iterations) + ": " + str(final_final_cost))

    iterations += 1

    return final_final_cost


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
# With our optimal parameters we now prepare the state to which
# they correspond, to see how close our prepared state is to the target
# state:
#

def prepare_state(params, device):

    # Initializes the density matrix

    final_density_matrix_2 = np.zeros((2 ** num_qubits, 2 ** num_qubits))

    # Prepares the optimal parameters, creates the distribution and the bitstrings

    dist_params = params[0:num_qubits]
    unitary_params = params[num_qubits:]

    distribution = prob_dist(dist_params)

    combos = itertools.product([0, 1], repeat=num_qubits)
    s = [list(i) for i in combos]

    # Runs the circuit in the case of the optimal parameters, for each bitstring,
    # and adds the result to the final density matrix.

    for i in s:
        qnode(unitary_params, qubits, i, 3)
        state = device.state
        for j in range(0, len(i)):
            state = np.sqrt(distribution[j][i[j]]) * state
        final_density_matrix_2 = np.add(final_density_matrix_2, np.outer(state, np.conj(state)))

    # Returns the prepared density matrix

    return final_density_matrix_2


final_density_matrix_2 = prepare_state(params, dev)


######################################################################
# To asess how "close together" the prepared and target state
# are, we will use the trace distance, which is a metric (a
# "distance function" with certain properties) on the space on density
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


print("Final Trace Distance: " + str(trace_distance(final_density_matrix_2, final_density_matrix)))


######################################################################
# These results are good! A trace distance close to :math:`0` means that the
# states are close together, thus we prepared a good
# approximation of the thermal state. If you prefer a visual
# representation, the prepared state can be plotted as a heatmap:
#

seaborn.heatmap(abs(final_density_matrix_2))


######################################################################
# Then, we compare it to the target:
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
# slightly more complicated model: the Heisenberg model on a 4-qubit cycle
# graph.
#


######################################################################
# Numerical Calculation of Target State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We start by defining our fixed values:
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
# .. math:: \hat{H} \ = \ \displaystyle\sum_{(i, j) \in E} X_i X_{j} \ + \ Z_i Z_{j} \ + \ Y_i Y_{j}
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

# Plots the final density matrix

final_density_matrix = create_target(num_qubits, beta, create_hamiltonian_matrix, interaction_graph)
seaborn.heatmap(abs(final_density_matrix))


######################################################################
# We will use the same form of the latent space, ansatz, and cost function
# as above, thus only minor modifications need to be made. We
# re-define our qnode, since we are now using a device with :math:`4`
# qubits rather than :math:`3`:
#

# Defines the new QNode

qnode = qml.QNode(quantum_circuit, dev2)


######################################################################
# We then run our optimizer (we use COBYLA again):
#

# Creates the optimizer

iterations = 0

params = [np.random.randint(-100, 100) / 100 for i in range(0, (16 * depth) + num_qubits)]
out = minimize(exact_cost, x0=params, method="COBYLA", options={"maxiter": 1000})
params = out["x"]
print(out)


######################################################################
# With our optimal parameters, we can post-process our data. We start by
# calculating the matrix form of the prepared density matrix:
#

# Prepares the density matrix

final_density_matrix_2 = prepare_state(params, dev2)


######################################################################
# We then calculate the trace distance:
#

print("Final Trace Distance: " + str(trace_distance(final_density_matrix_2, final_density_matrix)))


######################################################################
# This is pretty good, but it could be better (most likely with a deeper
# ansatz and a more sophisticated optimizer, but to keep execution time
# relatively short, we don't go down those avenues in this tutorial).
# To end off, we visualize the target and prepared density matrices. 
# We begin with the prepared matrix:
#

seaborn.heatmap(abs(final_density_matrix_2))


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