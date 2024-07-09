r"""Intro: explain the problem and the approach and mention the paper. - Problem: have an unknown
quantum process and want to reproduce it - Solution: train a variational circuit on input-output
pairs of the quantum process - Solution implementation: measure classical shadows of the target
quantum process and use those to estimate the overlap between the final states. Minimize a cost
function that captures this. - How this is different from a coherent process and the potential
advantages (why we care). - Say what we will do in this demo: create an “unknown” process
(trotterized hamiltonian time evolution), create initial states (haar random for uniform
distribution), measure the output shadows for each state, create a model circuit to learn this
process, train the model circuit with the shadows, repeat for the dataset
"""

import pennylane as qml
from pennylane import numpy as np

n_qubits = 2

######################################################################
# Create an Ising Hamiltonian with random weights
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Points to cover here: - We use the same “unknown” process as in the paper - It’s a trotterized (time
# evolution) transverse field Ising Hamiltonian
# :math:`H = \sum _{i=0}^{14} Z_iZ_{i+1} + \sum_{i=0}^{15}\alpha_iX_i` with random weights -
#

dev = qml.device("default.qubit")
alphas = np.random.normal(0, 0.5, size=n_qubits)
hamiltonian = qml.sum(
    *[qml.PauliZ(wires=i) @ qml.PauliZ(wires=i + 1) for i in range(n_qubits - 1)]
) + qml.sum(*[alphas[i] * qml.PauliX(wires=i) for i in range(n_qubits)])

######################################################################
# Create a random initial state
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Points to cover here: - We want to train over many initial states to get good coverage of the
# unknown process behavior - Ideally we want to use haar random initial states so that they are
# uniformly distributed over the state space and not over-representing a certain area (link haar demo)
#

random_complex_vector = np.random.random(size=2**n_qubits) + 1j * np.random.random(size=2**n_qubits)
random_state = random_complex_vector / np.linalg.norm(random_complex_vector)

# from https://pennylane.ai/qml/demos/tutorial_haar_measure/
from numpy.linalg import qr


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


######################################################################
# Trotter product time evolution of the random state
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


@qml.qnode(dev)
def target_circuit():
    # prepare training state
    qml.StatePrep(random_state, wires=range(n_qubits))
    # evolve according to desired hamiltonian

    qml.TrotterProduct(hamiltonian, 0.1, 1, 1)
    return qml.state()


print(qml.draw(target_circuit)())

######################################################################
# Create model circuit that will learn this
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


@qml.qnode(dev)
def placeholder():
    # first order trotterization of hamiltonian with delta_t = 0.1
    qml.TrotterProduct(hamiltonian, 0.1, n=1, order=1)
    return qml.state()


placeholder()

ops = qml.compile(placeholder.tape)[0][0].operations


@qml.qnode(dev)
def model_circuit(params):
    # this is a parameterized quantum circuit with the same gate structure as the target trotterised unitary
    qml.StatePrep(random_state, wires=range(n_qubits))
    for op, param in zip(ops, params):
        if op.name == "RX":
            qml.RX(param, wires=op.wires)
        elif op.name == "IsingZZ":
            qml.IsingZZ(param, wires=op.wires)
    return qml.state()


initial_params = np.random.random(size=len(ops), requires_grad=True)
model_circuit(initial_params)

print(qml.draw(model_circuit)(initial_params))

######################################################################
# Now use classical shadows instead of full states
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


@qml.qnode(dev)
def target_circuit_shadow():
    # prepare training state
    qml.StatePrep(random_state, wires=range(n_qubits))
    # evolve according to desired hamiltonian

    qml.TrotterProduct(hamiltonian, 0.1, 1, 1)
    return qml.classical_shadow(wires=range(n_qubits))


bits, recipes = target_circuit_shadow(shots=1000)

shadow = qml.ClassicalShadow(bits, recipes)


@qml.qnode(qml.device("default.qubit"))
def model_observable(params):
    # this is a parameterized quantum circuit with the same gate structure as the target trotterised unitary
    qml.StatePrep(random_state, wires=range(n_qubits))
    for op, param in zip(ops, params):
        if op.name == "RX":
            qml.RX(param, wires=op.wires)
        elif op.name == "IsingZZ":
            qml.IsingZZ(param, wires=op.wires)
    return qml.density_matrix(wires=range(n_qubits))


model_observable(initial_params)

obs_mat = model_observable(initial_params)


def cost_shadow(params):
    observable_mat = model_observable(params)
    observable_pauli = qml.pauli_decompose(observable_mat)
    return -qml.math.sum(shadow.expval(observable_pauli))


params = initial_params = np.random.random(size=len(ops), requires_grad=True)
observable_mat = model_observable(params)
# observable_herm = qml.Hermitian(observable_mat, wires=range(n_qubits))
observable_pauli = qml.pauli_decompose(observable_mat)

optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 200
for i in range(steps):
    params, final_cost = optimizer.step_and_cost(cost_shadow, params)

final_cost

model_circuit(params)

target_circuit()

######################################################################
# Now use the dataset
# ~~~~~~~~~~~~~~~~~~~
#

[ds] = qml.data.load("other", name="learning-dynamics-incoherently")

ds.list_attributes()

ds.attr_info["hamiltonian"]["doc"]

# use a few classical shadows
n_shadows = 10
shadow_ds = qml.ClassicalShadow(ds.shadow_meas[0][:n_shadows], ds.shadow_bases[0][:n_shadows])


@qml.qnode(dev)
def placeholder():
    # first order trotterization of hamiltonian with delta_t = 0.1
    qml.TrotterProduct(ds.hamiltonian, 0.1, n=1, order=1)
    return qml.state()


placeholder()

ops = qml.compile(placeholder.tape)[0][0].operations


@qml.qnode(dev)
def model_observable(params, qubit):
    # this is a parameterized quantum circuit with the same gate structure as the target trotterised unitary
    qml.StatePrep(ds.training_states[0], wires=range(16))
    for op, param in zip(ops, params):
        if op.name == "RX":
            qml.RX(param, wires=op.wires)
        elif op.name == "IsingZZ":
            qml.IsingZZ(param, wires=op.wires)
    return qml.density_matrix(qubit)


@qml.qnode(dev)
def target_circuit(qubit):
    # this is a parameterized quantum circuit with the same gate structure as the target trotterised unitary
    qml.StatePrep(ds.training_states[0], wires=range(16))
    qml.TrotterProduct(ds.hamiltonian, 0.1, 1, 1)
    return qml.density_matrix(qubit)


######################################################################
# The density matrix simulated exactly and the density matrix obtained from the hardware classical
# shadows are similar, but not the same.
#

######################################################################
# Returning the density matrix for 16 qubits would be too large, instead return a local state.
#


def cost_shadow_dataset(params):
    cost = 0
    for qubit in range(1):
        obs_mat = model_observable(params, qubit)
        obs_combination = qml.pauli_decompose(obs_mat, wire_order=[qubit])
        cost = cost + qml.math.sum(shadow_ds.expval(obs_combination))

    return -cost


params = initial_params = np.random.random(size=len(ops), requires_grad=True)

cost_shadow_dataset(params)

optimizer = qml.GradientDescentOptimizer()
steps = 200
for i in range(steps):
    params, final_cost = optimizer.step_and_cost(cost_shadow_dataset, params)

print(final_cost)

np.mean(shadow_ds.local_snapshots(wires=[0]), axis=0)

model_observable(params, 0)

model_circuit(params)
