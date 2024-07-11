r"""If we have an unknown quantum process ❓ that takes a quantum state as input and outputs
another state, how can we recreate it or simulate it in a quantum circuit? How do we create a
model circuit that reproduces the target quantum process? One approach is to learn the
[dynamics of this process incoherently](https://arxiv.org/abs/2303.12834). In simple terms, this
consists of two steps:
1. Measure the output of the unknown process for many different inputs
2. Adjust a variational quantum circuit until it produces the same input-output 
   combinations as the unknown process. 
  
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
1. Create an "unknown" target quantum process.
2. Create initial states to feed into the target process.
3. Measure the classical shadows of the target process.
4. Create a model variational circuit to learn the quantum process.
5. Train the variational circuit.
6. Repeat the procedure using the target quantum process and hardware measurements used in
   [The power and limitations of learning quantum dynamics incoherently]
   (https://arxiv.org/abs/2303.12834)
"""

######################################################################
# How to learn quantum dynamics incoherently
# -----------------------------------------------
#
######################################################################
# Creating the unknown target quantum process
# -------------------------------------------
# 
# We can perform a well known quantum process, imaginary time evolution of a Hamiltonian:
# $$U(H, t) = e^{-i H t / \hbar}$$
# Specifically, we will use the same Hamiltonian as in the paper,
# a transverse-field Ising Hamiltonian:
# :math:`H = \sum _{i=0}^{n-1} Z_iZ_{i+1} + \sum_{i=0}^{n}\alpha_iX_i`,
# where $n$ is the number of qubits and $\alpha$ are randomly generated weights.
# 
# We use a Trotterized version of this Hamiltonian through :class:`~pennylane.TrotterProduct`.
# 
# We first create the Hamiltonian. It will be trotterized later when we use it in a quantum circuit.

import pennylane as qml
from pennylane import numpy as np

# number of qubits for the Hamiltonian
n_qubits = 4

np.random.seed(0)
alphas = np.random.normal(0, 0.5, size=n_qubits)
hamiltonian = qml.sum(
    *[qml.PauliZ(wires=i) @ qml.PauliZ(wires=i + 1) for i in range(n_qubits - 1)]
) + qml.sum(*[alphas[i] * qml.PauliX(wires=i) for i in range(n_qubits)])

######################################################################
# Create random initial states
# -----------------------------
#
# The next step in our procedure is to prepare several initial states. We will then apply the
# "unknown" quantum process to each of these states to create input-output pairs. That is, for each
# input state, we will be able to measure the output state after appllying the "unknown" quantum
# process.
#
# Ideally, our input states should be uniformly distributed over the state space. If they are all
# clustered together, our model circuit will not learn to approximate the "unknown" quantum process
# behavior for states that are very different from our training set.
#
# For quantum systems, this means we want to sample [Haar random states](https://en.wikipedia.org/wiki/Haar_measure).
# We will create Haar random states using a procedure from our demo,
# [Understanding the Haar measure](https://pennylane.ai/qml/demos/tutorial_haar_measure/):
#
# .. note ::
#
#    Using this method to produce random Haar states starts to become computationally expensive 
#    around 10 qubits.

from numpy.linalg import qr

# reproduced from Understanding the Haar measure:
def qr_haar(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # Step 2
    Q, R = qr(Z)

    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4
    return np.dot(Q, Lambda)

random_unitary = qr_haar(2**n_qubits)
# requires_grad = False because we do not want differentiability
random_unitary.requires_grad = False
# Each column in the unitary is a Haar random state [this might be incorrect, so we are using only one column]
# random_states = [random_unitary[:,i] for i in range(2**n_qubits)]
random_states = [random_unitary[:,0]]


######################################################################
# Time evolution and classical shadow measurements
# ------------------------------------------------
#
# Now we can evolve the initial states using a Trotterized version of the Hamiltonian above. This
# will approximate the time evolution of the corresponding transverse-field Ising system.
# Similarly to the paper, we will use a Trotterization with time `0.1`, order 1, and 1 step. We then
# measure classical shadows on this system using :func:`~pennylane.classical_shadow`.
#

dev = qml.device("default.qubit")
@qml.qnode(dev)
def target_circuit():
    # prepare training state
    qml.StatePrep(random_states[0], wires=range(n_qubits))

    # evolve according to desired hamiltonian
    qml.TrotterProduct(hamiltonian, 0.1, 1, 1)
    return qml.classical_shadow(wires=range(n_qubits))


print(qml.draw(target_circuit)())

n_shadows = 1000
# shadows=[]
# for random_state in random_states:
#     bits, recipes = target_circuit(shots=n_shadows)
#     shadow = qml.ClassicalShadow(bits, recipes)
#     shadows.append(shadow)

#only using one shadow instead of commented code above
bits, recipes = target_circuit(shots=n_shadows)
shadow = qml.ClassicalShadow(bits, recipes)

######################################################################
# Create model circuit that will learn the target process
# -------------------------------------------------------
#
# Now that we have the classical shadow measurements, we need to create a `model_circuit` that
# learns to produce the same output as the target circuit. We will then use the classical shadow
# measurements to estimate the similarity between the `model_circuit` and the `target_circuit`. 
#
# Knowing that the `target_circuit` uses the :class:`~pennylane.TrotterProduct`, we choose a
# `model_circuit` that matches the :class:`~pennylane.TrotterProduct` structure. If the target
# target quantum process were truly unknown, then we could choose a general variational quantum
# circuit. 

@qml.qnode(dev)
def placeholder_trotter_circuit():
    # first order trotterization of hamiltonian with delta_t = 0.1
    qml.TrotterProduct(hamiltonian, 0.1, n=1, order=1)
    return qml.state()


placeholder_trotter_circuit()

ops = qml.compile(placeholder_trotter_circuit.tape)[0][0].operations

######################################################################
# .. note ::
#
#    We will be performing *local* measurements to keep the computational complexity lower and
#    because classical shadows are well-suited to estimating local observables [insert citation].
#    For this reason, the following circuit returns local density matrices for each qubit.

@qml.qnode(dev)
def model_circuit(params, random_state):
    # this is a parameterized quantum circuit with the same gate structure as the target trotterised unitary
    qml.StatePrep(random_state, wires=range(n_qubits))
    for op, param in zip(ops, params):
        if op.name == "RX":
            qml.RX(param, wires=op.wires)
        elif op.name == "IsingZZ":
            qml.IsingZZ(param, wires=op.wires)
    return [qml.density_matrix(i) for i in range(n_qubits)]

initial_params = np.random.random(size=len(ops), requires_grad=True)

print(qml.draw(model_circuit)(initial_params, random_states[0]))

######################################################################
# Train the model circuit using the classical shadows in a cost function
# ----------------------------------------------------------------------
#
# We now have to find the optimal parameters for `model_circuit` to mirror the `target_circuit`. 
# We can estimate the similarity between the circuits according to the cost function provided in 
# the paper. [insert equation]
#
# Our cost function is 


def cost(params):
    cost = 0.0
    # for random_state in random_states:
    observable_mats = model_circuit(params, random_states[0])
    observable_pauli = [qml.pauli_decompose(observable_mat, wire_order=[qubit]) for qubit, observable_mat in enumerate(observable_mats)]
    cost = cost - qml.math.sum(shadow.expval(observable_pauli))
    return cost

print("Initial cost:", cost(initial_params))

params = initial_params
observable_mat = model_circuit(params)
observable_pauli = qml.pauli_decompose(observable_mat)

optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 1000
for i in range(steps):
    params, final_cost = optimizer.step_and_cost(cost, params)

print("Final cost:", final_cost)

print(model_circuit(params))

print(target_circuit())

######################################################################
# Now we repeat using the learning dynamics incoherently dataset
# --------------------------------------------------------------
#

[ds] = qml.data.load("other", name="learning-dynamics-incoherently")

ds.list_attributes()

ds.attr_info["hamiltonian"]["doc"]

# use a few classical shadows
n_shadows = 10
shadow_ds = qml.ClassicalShadow(ds.shadow_meas[0][:n_shadows], ds.shadow_bases[0][:n_shadows])


@qml.qnode(dev)
def placeholder_trotter_circuit():
    # first order trotterization of hamiltonian with delta_t = 0.1
    qml.TrotterProduct(ds.hamiltonian, 0.1, n=1, order=1)
    return qml.state()


placeholder_trotter_circuit()

ops = qml.compile(placeholder_trotter_circuit.tape)[0][0].operations


@qml.qnode(dev)
def model_circuit(params, random_state):
    # this is a parameterized quantum circuit with the same gate structure as the target trotterised unitary
    qml.StatePrep(random_state, wires=range(16))
    for op, param in zip(ops, params):
        if op.name == "RX":
            qml.RX(param, wires=op.wires)
        elif op.name == "IsingZZ":
            qml.IsingZZ(param, wires=op.wires)
    return [qml.density_matrix(i) for i in range(16)]


def cost_dataset(params):

    cost = 0.0
    observable_mats = model_circuit(params, ds.training_states[0])
    observable_pauli = [qml.pauli_decompose(observable_mat, wire_order=[qubit]) for qubit, observable_mat in enumerate(observable_mats)]
    cost = cost - qml.math.sum(shadow.expval(observable_pauli))
    return cost


params = initial_params = np.random.random(size=len(ops), requires_grad=True)

cost_dataset(params)

optimizer = qml.GradientDescentOptimizer()
steps = 100

for i in range(steps):
    params, final_cost = optimizer.step_and_cost(cost_dataset, params)

print(final_cost)

print(np.mean(shadow_ds.local_snapshots(wires=[0]), axis=0))

print(model_circuit(params,ds.training_states[0]))
