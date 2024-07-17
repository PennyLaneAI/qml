r"""How to learn quantum dynamics incoherently
==========================================

If we have an unknown quantum process that takes a quantum state as input and outputs
another state, how can we recreate it or simulate it in a quantum circuit? How do we create a
model circuit that reproduces the target quantum process? One approach is to learn the
dynamics of this process incoherently [#Jerbi]_. In simple terms, this
consists of two steps:

1. Measure the output of the unknown process for many different inputs
2. Adjust a variational quantum circuit until it produces the same input-output combinations as the unknown process.


For step 1, we measure classical shadows of the target process output.
For step 2, we simulate the model circuit to get its final state. To know how similar the simulated
state is to the target process output, we use the classical shadows to estimate the overlap of the
states. [double check wording here, we aren't exactly estimating the overlap]

This is different to learning the quantum process *coherently* because it does not require the model
circuit to be connected to the target quantum process. That is, the model circuit does not receive
quantum information from the target process directly. Instead, we train the model circuit using
classical information obtained from the classical shadow measurements. One reason this is useful is
that it's not always possible to port the quantum output of a system directly to hardware without
first measuring it.

In this tutorial, we will use PennyLane to do the following:

1. Create an unknown target quantum process.
2. Create initial states to feed into the target process.
3. Measure the classical shadows of the target process.
4. Create a model variational circuit to learn the quantum process.
5. Train the variational circuit.


We can then replicate the investigation in [#Jerbi]_ by using the
`Learning Dynamics Incoherently dataset <https://pennylane.ai/datasets/other/learning-dynamics-incoherently>`__.
"""

######################################################################
# 1. Creating an unknown target quantum process
# -------------------------------------------
# 
# For our unknown quantum process, we will use a well-known quantum process,
# the `time evolution of a Hamiltonian <https://en.wikipedia.org/wiki/Hamiltonian_(quantum_mechanics)#Schr%C3%B6dinger_equation>`_:
# 
# .. math:: U(H, t) = e^{-i H t / \hbar} .
#
# Specifically, we will use an approximation of :math:`U` via Trotterization.
# For the Hamiltonian, :math:`H`, we choose a transverse-field Ising Hamiltonian, as in the
# referenced paper [#Jerbi]_:
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

# number of qubits for the Hamiltonian
n_qubits = 4

#set random seeds for reproducibility
pnp.random.seed(0)
np.random.seed(0)

alphas = pnp.random.normal(0, 0.5, size=n_qubits)
hamiltonian = qml.sum(
    *[qml.PauliZ(wires=i) @ qml.PauliZ(wires=i + 1) for i in range(n_qubits - 1)]
)+ qml.dot(alphas, [qml.PauliX(wires=i) for i in range(n_qubits)])

######################################################################
# 2. Creating random initial states
# ---------------------------------
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
# We will create Haar random states using a procedure from our demo,
# `Understanding the Haar measure <https://pennylane.ai/qml/demos/tutorial_haar_measure/>`_:
#
# .. note ::
#
#    On a personal computer, this method becomes slow (>1 second) around 10 qubits. 
#    

from numpy.linalg import qr

# reproduced from Understanding the Haar measure:
def qr_haar(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1: generate an N x N matrix Z with normally distributed complex numbers
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # Step 2: compute a QR decomposition of Z
    Q, R = qr(Z)

    # Step 3: create a diagonal matrix  
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4: a Haar-random matrix Q' = Q@Lambda
    return np.dot(Q, Lambda)

random_unitary = qr_haar(2**n_qubits)

random_states = [random_unitary[:,0]]


######################################################################
# 3. Time evolution and classical shadow measurements
# --------------------------------------------------
#
# Now we can evolve the initial states using a Trotterized version of the Hamiltonian above. This
# will approximate the time evolution of the corresponding transverse-field Ising system.
# Similarly to the paper [#Jerbi]_, we will use a Trotterization with time 0.1, order 1, and 1 step. We then
# measure classical shadows on this system using :func:`~pennylane.classical_shadow`.
#

dev = qml.device("default.qubit")
@qml.qnode(dev)
def target_circuit():
    # prepare training state
    qml.StatePrep(random_states[0], wires=range(n_qubits))

    # evolve according to desired hamiltonian
    qml.TrotterProduct(hamiltonian, 2, 1, 1)
    return qml.classical_shadow(wires=range(n_qubits))


print(qml.draw(target_circuit)())

n_shadows = 1000

bits, recipes = target_circuit(shots=n_shadows)
shadow = qml.ClassicalShadow(bits, recipes)


######################################################################
# 4. Creating a model circuit that will learn the target process
# -------------------------------------------------------------
#
# Now that we have the classical shadow measurements, we need to create a ``model_circuit`` that
# learns to produce the same output as the target circuit. We will then use the classical shadow
# measurements to estimate the similarity between the `model_circuit` and the `target_circuit`. 
#
# As done in [#Jerbi]_, we create a ``model_circuit`` with the same gate structure as the target
# structure. If the target quantum process were truly unknown, then we could choose a general
# variational quantum circuit like in the `Variational classifier demo <https://pennylane.ai/qml/demos/tutorial_variational_classifier/>`_. 
#
# .. note ::
#
#    We will be performing *local* measurements to keep the computational complexity lower and
#    because classical shadows are well-suited to estimating local observables [#Jerbi]_.
#    For this reason, the following circuit returns local density matrices for each qubit.

@qml.qnode(dev)
def model_circuit(params, random_state):
    qml.StatePrep(random_state, wires=range(n_qubits))
    # this is a parameterized quantum circuit with the same gate structure as the target Trotterized unitary
    qml.RX(params[0], wires=0)
    qml.RX(params[1], wires=1)
    qml.RX(params[2], wires=2)
    qml.RX(params[3], wires=3)
    qml.IsingZZ(params[4], wires=[2,3])
    qml.IsingZZ(params[5], wires=[1,2])
    qml.IsingZZ(params[6], wires=[0,1])
    return [qml.density_matrix(i) for i in range(n_qubits)]

initial_params = pnp.random.random(size=7, requires_grad=True)

print(qml.draw(model_circuit)(initial_params, random_states[0]))

######################################################################
# 5. Training a model circuit using the classical shadows in a cost function
# ---------------------------------------------------------------------------
#
# We now have to find the optimal parameters for `model_circuit` to mirror the `target_circuit`. 
# We can estimate the similarity between the circuits according to the cost function provided in 
# the paper. [insert equation]
#
# Our cost function is:


def cost(params):
    cost = 0.0
    # for random_state in random_states:
    observable_mats = model_circuit(params, random_states[0])
    observable_pauli = [qml.pauli_decompose(observable_mat, wire_order=[qubit]) for qubit, observable_mat in enumerate(observable_mats)]
    cost = cost - qml.math.sum(shadow.expval(observable_pauli))
    return cost

print("Initial cost:", cost(initial_params))

params = initial_params

optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 100
for i in range(steps):
    params, final_cost = optimizer.step_and_cost(cost, params)

print("Final cost:", final_cost)

print(model_circuit(params, random_states[0]))

print(pnp.mean(shadow.local_snapshots(),axis=0))

######################################################################
# Using the learning dynamics incoherently dataset
# ----------------------------------------------------------------
#
# [#Jerbi]_ Performs the procedure described above on a larger, 16-qubit transverse field Ising
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

n_shadows = 10
shadow_ds = qml.ClassicalShadow(ds.shadow_meas[0][:n_shadows], ds.shadow_bases[0][:n_shadows])

######################################################################
# 
# We only need to create the model circuit, cost function, and train:
# 

dev = qml.device('default.qubit')

@qml.qnode(dev)
def model_circuit(params, random_state):
    # this is a parameterized quantum circuit with the same gate structure as the target Trotterized unitary
    qml.StatePrep(random_state, wires=range(16))
    for i in range(16):
        print(qml.RX(params[i],wires=i))

    for i in reversed(range(15)):
        print(qml.IsingZZ(params[16+i], wires=[i,i+1]))
    return [qml.density_matrix(i) for i in range(16)]


def cost_dataset(params):

    cost = 0.0
    observable_mats = model_circuit(params, ds.training_states[0])
    observable_pauli = [qml.pauli_decompose(observable_mat, wire_order=[qubit]) for qubit, observable_mat in enumerate(observable_mats)]
    cost = cost - qml.math.sum(shadow_ds.expval(observable_pauli))
    return cost


params = initial_params = pnp.random.random(size=31, requires_grad=True)

cost_dataset(params)

optimizer = qml.GradientDescentOptimizer()
steps = 10

for i in range(steps):
    params, final_cost = optimizer.step_and_cost(cost_dataset, params)

print(final_cost)


##############################################################################
# 
# References
# ----------
#
# .. [#Jerbi]
#
#     Sofiene Jerbi, Joe Gibbs, Manuel S. Rudolph, Matthias C. Caro, Patrick J. Coles, Hsin-Yuan Huang, Zoë Holmes
#     "The power and limitations of learning quantum dynamics incoherently"
#     `arXiv:2303.12834 <https://arxiv.org/abs/quant-ph/0512209>`__, 2005.
#

##############################################################################
# About the author
# ----------------
#