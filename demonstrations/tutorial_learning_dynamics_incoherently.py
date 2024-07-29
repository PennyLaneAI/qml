r"""Learning quantum dynamics incoherently: Variational learning using classical shadows
====================================================================================

How can we recreate and simulate an unknown quantum process with a quantum circuit? One approach is to learn the
dynamics of this process incoherently, as done by Jerbi et al. [#Jerbi]_. Here, we'll reproduce the numerical
simulations of [#Jerbi]_ using the authors' data, as provided in the
`Learning Dynamics Incoherently PennyLane Dataset <https://pennylane.ai/datasets/other/learning-dynamics-incoherently>`__. 

In simple terms, learning dynamics incoherently consists of two steps. First, we measure the output
of the unknown process for many different inputs. Then, we adjust a variational quantum circuit
until it produces the same input-output combinations as the unknown process. For step 1, we measure
:doc:`classical shadows <tutorial_classical_shadows>` of the target process output.
For step 2, we simulate the model circuit to get its final state. To know how similar the simulated
state is to the target process output, we estimate the overlap of the states using the classical
shadows from step 1.

This approach is different to learning the quantum process *coherently* [#Huang]_ because it does not require the model
circuit to be connected to the target quantum process. That is, the model circuit does not receive
quantum information from the target process directly. Instead, we train the model circuit using
classical information obtained from the classical shadow measurements. This is useful because
it's not always possible to port the quantum output of a system directly to hardware without
first measuring it. However, depending on the quantum process, an exponential number of classical shadow measurements
may be required [#Jerbi]_.

In this tutorial, we will use PennyLane to create an unknown target quantum process and the initial
states to go into it, simulate shallow classical shadow measurements of the target process, and create and
train a model variational circuit that approximates the target process.
Finally, we will use the `Learning Dynamics Incoherently PennyLane Dataset <https://pennylane.ai/datasets/other/learning-dynamics-incoherently>`__
to replicate part of the original investigation.
"""

######################################################################
# 1. Creating an unknown target quantum process
# ----------------------------------------------
#
# For our unknown quantum process, we will use a well-known quantum process,
# the :doc:`time evolution of a Hamiltonian </demos/tutorial_qaoa_intro/#circuits-and-hamiltonians>`:
#
# .. math:: U(H, t) = e^{-i H t / \hbar} .
#
# Specifically, we will use an approximation of :math:`U` via `Trotterization <https://en.wikipedia.org/wiki/Hamiltonian_simulation#Product_formulas>`_.
# For the Hamiltonian, :math:`H`, we choose a transverse-field Ising Hamiltonian (as in
# [#Jerbi]_):
#
# .. math:: H = \sum_{i=0}^{n-1} Z_iZ_{i+1} + \sum_{i=0}^{n}\alpha_iX_i,
#
# where :math:`n` is the number of qubits and :math:`\alpha` are randomly generated weights.
#
# We first create the Hamiltonian and Trotterize later in a quantum circuit via
# :class:`~pennylane.TrotterProduct`.

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt

# number of qubits for the Hamiltonian
n_qubits = 2

# set random seeds for reproducibility
pnp.random.seed(0)
np.random.seed(0)

# generate random weights
alphas = pnp.random.normal(0, 0.5, size=n_qubits)

# create the Hamiltonian
hamiltonian = qml.sum(
    *[qml.PauliZ(wires=i) @ qml.PauliZ(wires=i + 1) for i in range(n_qubits - 1)]
) + qml.dot(alphas, [qml.PauliX(wires=i) for i in range(n_qubits)])

######################################################################
# 2. Creating random initial states
# -----------------------------------
#
# The next step in our procedure is to prepare several initial states. We will then apply the
# unknown quantum process to each of these states to create input-output pairs. That is, for each
# input state, we will be able to measure the output state after applying the unknown quantum
# process.
#
# Ideally, our input states should be uniformly distributed over the state space. If they are all
# clustered together, our model circuit will not learn to approximate the unknown quantum process
# behavior for states that are very different from our training set.
#
# For quantum systems, this means we want to sample :doc:`Haar random states <tutorial_haar_measure>`.
#
# .. note ::
#
#    On a personal computer, this method becomes slow (>1 second) around 10 qubits.
#

from scipy.stats import unitary_group

n_random_states = 100

# Generate several random unitaries
random_unitaries = unitary_group.rvs(2**n_qubits, n_random_states)
# Take the first column of each unitary
random_states = [random_unitary[:, 0] for random_unitary in random_unitaries]

######################################################################
# .. note ::
#
#    On a personal computer, this method becomes slow (>1 second) around 10 qubits.
#


######################################################################
# 3. Time evolution and classical shadow measurements
# ----------------------------------------------------
#
# Now we can evolve the initial states using a Trotterized version of the `Hamiltonian above <#creating-an-unknown-target-quantum-process>`_. This
# will approximate the time evolution of the corresponding transverse-field Ising system.
#

dev = qml.device("default.qubit")


@qml.qnode(dev)
def target_circuit(input_state):
    # prepare training state
    qml.StatePrep(input_state, wires=range(n_qubits))

    # evolve according to desired hamiltonian
    qml.TrotterProduct(hamiltonian, 2, 1, 1)
    return qml.classical_shadow(wires=range(n_qubits))


qml.draw_mpl(target_circuit)(random_states[0])
plt.show()

######################################################################
#
# Since the circuit returns :func:`~pennylane.classical_shadow` as a measurement,
# we can run the circuit with a ``shot`` value to obtain the desired number of classical shadow
# measurements. We create a set of measurements for each initial state:
#

n_measurements = 10000

shadows = []
for random_state in random_states:
    bits, recipes = target_circuit(random_state, shots=n_measurements)
    shadow = qml.ClassicalShadow(bits, recipes)
    shadows.append(shadow)


######################################################################
# 4. Creating a model circuit that will learn the target process
# ----------------------------------------------------------------
#
# Now that we have the classical shadow measurements, we need to create a ``model_circuit`` that
# learns to produce the same output as the target circuit. We will then use the classical shadow
# As done in [#Jerbi]_, we create a ``model_circuit`` with the same gate structure as the target
#
# As done in [#Jerbi]_, we create a ``model_circuit`` with the same gate structure as the target
# structure. If the target quantum process were truly unknown, then we could choose a general
# variational quantum circuit like in the :doc:`Variational classifier demo </demos/tutorial_variational_classifier>`.
#
# .. note ::
#
#    We will be performing *local* measurements to keep the computational complexity lower and
#    because classical shadows are well-suited to estimating local observables [#Jerbi]_.
#    For this reason, the following circuit returns local density matrices for each qubit. In
#    hardware, the density matrix is obtained via state tomography using Pauli measurements or classical shadows.


@qml.qnode(dev)
def model_circuit(params, random_state):
    qml.StatePrep(random_state, wires=range(n_qubits))
    # this is a parameterized quantum circuit with the same gate structure as the target Trotterized unitary
    for i in range(n_qubits):
        qml.RX(params[i], wires=i)

    for i in reversed(range(n_qubits - 1)):
        qml.IsingZZ(params[n_qubits + i], wires=[i, i + 1])
    return [qml.density_matrix(i) for i in range(n_qubits)]


initial_params = pnp.random.random(size=7, requires_grad=True)

qml.draw_mpl(model_circuit)(initial_params, random_states[0])
plt.show()

######################################################################
# 5. Training a model circuit using classical shadows in a cost function
# ------------------------------------------------------------------------
#
# We now have to find the optimal parameters for ``model_circuit`` to mirror the ``target_circuit``.
# We can estimate the similarity between the circuits according to this cost function (see
# appendix B of [#Jerbi]_):
#
# .. math:: C^l_N(\theta) = 1 - \frac{1}{nN}\sum^N_{j=1}\sum^n_{i=1}Tr[U|\psi^{(j)}\rangle\langle\psi^{(j)}|U^\dagger O^{(j)}_i(\theta)],
#
# where :math:`n` is the number of qubits, :math:`N` is the number of initial states, :math:`\psi^{(j)}`
# are random states, :math:`U` is our target unitary operation, and :math:`O_i` is the local density
# matrix for qubit :math:`i` after applying the ``model_circuit``. That is, the local states
# :math:`\rho_{i}^{(j)}` are used as the observables:
#
# .. math:: O_{i}^{(j)}(\theta) := \rho_{i}^{(j)}.
#
# We can calculate this cost for our system by using the
# `shadow measurements <#time-evolution-and-classical-shadow-measurements>`_ to estimate
# the expectation value of :math:`O_i`. Roughly, this cost function measures the fidelity between
# the model circuit and the target circuit, by proxy of the single-qubit reduced states
# :math:`\rho_{i}^{(j)}` of the model over a variety of input-output pairs.


def cost(params):
    cost = 0.0
    for idx, random_state in enumerate(random_states):
        # Obtain the density matrices for each qubit
        observable_mats = model_circuit(params, random_state)
        # Convert to a PauliSentence
        observable_pauli = [
            qml.pauli_decompose(observable_mat, wire_order=[qubit])
            for qubit, observable_mat in enumerate(observable_mats)
        ]
        # Estimate the overlap for each qubit
        cost = cost + qml.math.sum(shadows[idx].expval(observable_pauli))
    cost = 1 - cost / n_qubits / n_random_states
    return cost


params = initial_params

optimizer = qml.GradientDescentOptimizer(stepsize=5)
steps = 50
costs = [None] * (steps + 1)
costs[0] = cost(initial_params)

for i in range(steps):
    params, costs[i + 1] = optimizer.step_and_cost(cost, params)

print("Initial cost:", costs[0])
print("Final cost:", costs[-1])

######################################################################
#
# We can also plot the cost over the iterations and compare to the ideal cost.

# Find the ideal parameters from the original Trotterized Hamiltonian
ideal_parameters = [
    op.decomposition()[0].parameters[0]
    for op in qml.TrotterProduct(hamiltonian, 2, 1, 1).decomposition()
]
ideal_parameters = ideal_parameters[:n_qubits][::-1] + ideal_parameters[n_qubits:]

ideal_cost = cost(ideal_parameters)

plt.plot(costs, label="Training")
plt.plot([0, steps], [ideal_cost, ideal_cost], "r--", label="Ideal Parameters")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.legend()
plt.show()

######################################################################
# In this case, we see
# that the ideal cost is greater than 0. This is because for the ideal parameters, the model outputs
# and target outputs are equal:
#
# .. math:: \rho_{i}^{(j)} := O_i = U|\psi^{(j)}\rangle\langle\psi^{(j)}|U^\dagger.
#
# Since the single-qubit reduced states used in the
# cost function are mixed states, they have trace less than one:
#
# .. math:: Tr[(\rho_{i}^{(j)})^2] < 1,
#
# the cost :math:`C^l_N(\theta)` is therefore greater than 0.
#
# Using the learning dynamics incoherently dataset
# ----------------------------------------------------------------
#
# In Jerbi et al. [#Jerbi]_, the authors perform this procedure to learn dynamics incoherently on a larger, 16-qubit transverse-field Ising
# Hamiltonian, and use classical shadow samples from quantum hardware to estimate the cost function.
# This data is available in the `Learning Dynamics Incoherently PennyLane Dataset <https://pennylane.ai/datasets/other/learning-dynamics-incoherently>`__ and downloadable through the :mod:`qml.data` module.
#
# To use this dataset, we first load it using :func:`~pennylane.data.load`:

[ds] = qml.data.load("other", name="learning-dynamics-incoherently")

# print the available data
print(ds.list_attributes())

# print more information about the hamiltonian
print(ds.attr_info["hamiltonian"]["doc"])

######################################################################
#
# The unknown target Hamiltonian, Haar-random intial states, and resulting classical shadow
# measurements are all already available in the dataset.
#
# .. note ::
#
#    We use few shadows and only one training state to keep the computational time low.

random_state = ds.training_states[0]

n_measurements = 1000
shadow_ds = qml.ClassicalShadow(
    ds.shadow_meas[0][:n_measurements], ds.shadow_bases[0][:n_measurements]
)

######################################################################
#
# We only need to create the model circuit, cost function, and train.
# For these we use the same model circuit as `above <#creating-a-model-circuit-that-will-learn-the-target-process>`_, updated to reflect the increased number
# of qubits:
#

dev = qml.device("default.qubit")


@qml.qnode(dev)
def model_circuit(params, random_state):
    # this is a parameterized quantum circuit with the same gate structure as the target Trotterized unitary
    qml.StatePrep(random_state, wires=range(16))
    for i in range(16):
        qml.RX(params[i], wires=i)

    for i in reversed(range(15)):
        qml.IsingZZ(params[16 + i], wires=[i, i + 1])
    return [qml.density_matrix(i) for i in range(16)]


initial_params = pnp.random.random(size=31)

qml.draw_mpl(model_circuit)(initial_params, random_state)
plt.show()

######################################################################
#
# For the cost function, we repeat the same code as `before <#training-a-model-circuit-using-classical-shadows-in-a-cost-function>`_, updated to use a single input state:
#


def cost_dataset(params):

    observable_mats = model_circuit(params, random_state)
    observable_pauli = [
        qml.pauli_decompose(observable_mat, wire_order=[qubit])
        for qubit, observable_mat in enumerate(observable_mats)
    ]
    cost = 1 - qml.math.sum(shadow_ds.expval(observable_pauli)) / 16
    return cost


######################################################################
#
# We can then minimize the cost to train the dataset to output the same states as the target circuit.
#

params = initial_params

optimizer = qml.GradientDescentOptimizer(stepsize=5)
steps = 50

cost_history = []
for i in range(steps):
    params, final_cost = optimizer.step_and_cost(cost_dataset, params)
    cost_history.append(final_cost)

print("Initial cost:", cost_dataset(initial_params))
print("Final cost:", final_cost)

######################################################################
#
# We can again take a look at the density matrices to confirm that the training was successful:
#

original_matrices = model_circuit(initial_params, random_state)
learned_matrices = model_circuit(params, random_state)
target_matrices_shadow = np.mean(shadow_ds.local_snapshots(), axis=0)

print("Untrained example output state\n", original_matrices[0])
print("Trained example output state\n", learned_matrices[0])
print("Target output state\n", target_matrices_shadow[0])

######################################################################
#
# After training, the model outputs are closer to the target outputs estimated
# using classical shadows, but not quite the same. This can be improved by using more training
# states and classical shadow measurements.
#


##############################################################################
#
# References
# ------------
#
# .. [#Jerbi]
#
#     Sofiene Jerbi, Joe Gibbs, Manuel S. Rudolph, Matthias C. Caro, Patrick J. Coles, Hsin-Yuan Huang, Zoë Holmes
#     "The power and limitations of learning quantum dynamics incoherently"
#     `arXiv:2303.12834 <https://arxiv.org/abs/2303.12834>`__, 2005.
#
# .. [#Huang]
#
#     Hsin-Yuan Huang, Michael Broughton, Jordan Cotler, Sitan Chen, Jerry Li, Masoud Mohseni, Hartmut Neven, Ryan Babbush, Richard Kueng, John Preskill, and Jarrod R. McClean
#     "Quantum advantage in learning from experiments"
#     `Science <http://dx.doi.org/10.1126/science.abn7293>`__, 2022
#

##############################################################################
# About the author
# ------------------
#
# .. include:: ../_static/authors/diego_guala.txt
