r"""How to learn quantum dynamics incoherently
==========================================

How can we recreate and simulate an unknown quantum process with a quantum circuit? One approach is to learn the
dynamics of this process incoherently, as done by Jerbi et al. [#Jerbi]_. Here, we'll reproduce the numerical
simulations of [#Jerbi]_ using the authors' data as provided in the
`Learning Dynamics Incoherently dataset <https://pennylane.ai/datasets/other/learning-dynamics-incoherently>`__. 

In simple terms, learning dynamics incoherently consists of two steps. First, we measure the output
of the unknown process for many different inputs. Then, we adjust a variational quantum circuit
until it produces the same input-output combinations as the unknown process. For step 1, we measure
classical shadows of the target process output.
For step 2, we simulate the model circuit to get its final state. To know how similar the simulated
state is to the target process output, we estimate the overlap of the states using the classical
shadows from step 1.

This is different to learning the quantum process *coherently* because it does not require the model
circuit to be connected to the target quantum process. That is, the model circuit does not receive
quantum information from the target process directly. Instead, we train the model circuit using
classical information obtained from the classical shadow measurements. One reason this is useful is
that it's not always possible to port the quantum output of a system directly to hardware without
first measuring it.

In this tutorial, we will use PennyLane to create an unknown target quantum process and the initial
states to go into it, simulate classical shadow measurements of the target process, and create and
train a model variational circuit that approximates the target process.

We can then replicate the investigation in [#Jerbi]_ by using the
`Learning Dynamics Incoherently dataset <https://pennylane.ai/datasets/other/learning-dynamics-incoherently>`__.
"""

######################################################################
# 1. Creating an unknown target quantum process
# ----------------------------------------------
#
# For our unknown quantum process, we will use a well-known quantum process,
# the `time evolution of a Hamiltonian <https://en.wikipedia.org/wiki/Hamiltonian_(quantum_mechanics)#Schr%C3%B6dinger_equation>`_:
#
# .. math:: U(H, t) = e^{-i H t / \hbar} .
#
# Specifically, we will use an approximation of :math:`U` via `Trotterization <https://en.wikipedia.org/wiki/Hamiltonian_simulation#Product_formulas>`_.
# For the Hamiltonian, :math:`H`, we choose a transverse-field Ising Hamiltonian, as in
# [#Jerbi]_:
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
n_qubits = 4

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
# For quantum systems, this means we want to sample `Haar random states <https://en.wikipedia.org/wiki/Haar_measure>`__.
# For more info, see our demo,
# `Understanding the Haar measure <https://pennylane.ai/qml/demos/tutorial_haar_measure/>`_:
#

from scipy.stats import unitary_group

n_random_states = 10

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
# Now we can evolve the initial states using a Trotterized version of the Hamiltonian above. This
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
# As done in Jerbi et al. [#Jerbi]_, we create a ``model_circuit`` with the same gate structure as the target
# structure. If the target quantum process were truly unknown, then we could choose a general
# variational quantum circuit like in the :doc:`Variational classifier demo </demos/tutorial_variational_classifier>`.
#
# .. note ::
#
#    We will be performing *local* measurements to keep the computational complexity lower and
#    because classical shadows are well-suited to estimating local observables [#Jerbi]_.
#    For this reason, the following circuit returns local density matrices for each qubit. In
#    hardware, the density matrix is obtained via Pauli measurements and state tomography.


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
# 5. Training a model circuit using the classical shadows in a cost function
# ----------------------------------------------------------------------------
#
# We now have to find the optimal parameters for ``model_circuit`` to mirror the ``target_circuit``.
# We can estimate the similarity between the circuits according to the cost function provided in
# appendix B of Jerbi et al. [#Jerbi]_.
#
# .. math:: C^l_N(\theta) = 1 - \frac{1}{nN}\sum^N_{j=1}\sum^n_{i=1}Tr[U|\psi^{(j)}\rangle\langle\psi^{(j)}|U^\dagger O^{(j)}_i(\theta)],
#
# Where :math:`n` is the number of qubits, :math:`N` is the number of initial states, :math:`\psi^{(j)}`
# are random states, :math:`U` is our target unitary operation, and :math:`O_i` is the local density
# matrix for qubit :math:`i` after applying the ``model_circuit``. That is, the local states
# :math:`\rho_{i}^{(j)}` are defined by:
# 
# .. math:: \rho_{i}^{(j)} =: O_{i}^{(j)}(\theta)
#
# We can calculate this cost for our system by using the shadow measurements made before to estimate
# the expectation value of :math:`O_i`. This gives an estimate of the overlap between the state of
# the target circuit and the model circuit:

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


print("Initial cost:", cost(initial_params))

params = initial_params

optimizer = qml.GradientDescentOptimizer(stepsize=5)
steps = 100
for i in range(steps):
    params, final_cost = optimizer.step_and_cost(cost, params)

print("Final cost:", final_cost)

######################################################################
#
# After training, we can take a look at the density matrix of a qubit for both the model and the
# target circuit estimate from the classical shadows. If the training was successful, these should
# be similar:
#

print("Untrained state for qubit 1:\n", model_circuit(initial_params, random_states[0])[1])

print("Trained state for qubit 1:\n", model_circuit(params, random_states[0])[1])

print("Target state for qubit 1:\n", pnp.mean(shadows[0].local_snapshots(), axis=0)[1])

######################################################################
# Using the learning dynamics incoherently dataset
# ----------------------------------------------------------------
#
# In Jerbi et al. [#Jerbi]_, the authors perform the procedure described above on a larger, 16-qubit transverse field Ising
# Hamiltonian and uses classical shadow samples from quantum hardware to estimate the cost function.
# This data is available in PennyLane through the :mod:`qml.data` module. More information about
# the dataset itself is available on the
# `Learning Dynamics Incoherently dataset page <https://pennylane.ai/datasets/other/learning-dynamics-incoherently>`_.
#
# We first load the dataset using :func:`~pennylane.data.load`:

[ds] = qml.data.load("other", name="learning-dynamics-incoherently")

# print the available data
print(ds.list_attributes())

# print more information about the hamiltonian
print(ds.attr_info["hamiltonian"]["doc"])

######################################################################
#
# The unknown target hamiltonian, Haar-random intial states, and resulting classical shadow
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
# For these we use the same model circuit as above, updated to reflect the increased number
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

######################################################################
#
# For the cost function, we repeat the same code as above, updated to use a single input state:
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
# After training, the model outputs are similar to the target outputs estimated
# using classical shadows.
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

##############################################################################
# About the author
# ------------------
#
# .. include:: ../_static/authors/diego_guala.txt
