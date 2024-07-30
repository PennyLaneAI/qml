r"""How to implement VQD with PennyLane
===============================================================

Finding eigenvalues of an operator is a key task in quantum computing. Algorithms like VQE are used to find the smallest
one, but sometimes we are interested in other eigenvalues. This how-to shows you how to implement  Variational
Quantum Deflation (VQD) in PennyLane
and find the first excited state energy of the hydrogen molecule. To benefit the most from this tutorial, we recommend
a familiarization with the `Variational Quantum Eigensolver (VQE) <https://pennylane.ai/qml/demos/tutorial_vqe/>`_ tutorial.

.. figure:: ../_static/demonstration_assets/vqe_vqd/how_to_vqd_pennylane_opengraph.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

"""

######################################################################
#
# Variational quantum deflation
# ------------------------------
#
# VQD algorithm [#Vqd]_ is a method used to find excited states of a quantum system.
# The VQD algorithm is related on the VQE algorithm, which finds the ground state energy of a quantum system.
# The main idea in VQE is to define an ansatz that depends on adjustable parameters :math:`\theta` and minimizes the energy computed as:
#
# .. math:: C_0(\theta) = \left\langle\Psi (\theta)|\hat H |\Psi (\theta) \right\rangle,
#
# where :math:`\Psi(\theta)` is the ansatz. However, this is not enough if we are not looking for the ground state energy.
# We must find a function whose minimum is no longer the ground state energy but gives the next excited state.
# This is possible just by adding a penalty term to the above function that accounts for the orthogonality of the states:
#
# .. math:: C_1(\theta) = \left\langle\Psi(\theta)|\hat H |\Psi (\theta) \right\rangle + \beta | \left\langle \Psi (\theta)| \Psi_0 \right\rangle|^2,
#
# where :math:`\beta` is a hyperparameter that controls the penalty term and :math:`| \Psi_0 \rangle` is the ground state.
# The function can be minimized to give the energy but we are now restricting the new state to be orthogonal to the ground state.
#
# Note that the :math:`\beta` should be larger than the energy gap between the ground and excited states.
# Similarly, we could iteratively calculate the :math:`k`-th excited states by adding the corresponding penalty term to the previous :math:`k - 1` excitation states.
#
# As easy as that! Let's see how we can run this using PennyLane
#
#
# Finding the ground state
# -------------------------------------------
#
# To implement VQD, with first need to know the ground state of our system. The `datasets` package of PennyLane makes it a breeze to find the Hamiltonian and the ground state
# of several molecules including hydrogen.
# We use this dataset to obtain the ground state directly:
#

import pennylane as qml
from pennylane import numpy as np

# Load the dataset
h2 = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")[0]

# Extract the Hamiltonian
H, n_qubits = h2.hamiltonian, len(h2.hamiltonian.wires)


# Obtain the ground state from the operations given by the dataset
def generate_ground_state(wires):
    qml.BasisState(h2.hf_state, wires=wires)

    for op in h2.vqe_gates:  # use the gates data the datasets package provided
        op = qml.map_wires(op, {op.wires[i]: wires[i] for i in range(len(wires))})
        qml.apply(op)

######################################################################
# The ``generate_ground_state`` function prepares the ground state of the molecule using the data obtained from the dataset.
# Let's use it to check the energy of that state:
#

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    generate_ground_state(range(n_qubits))
    return qml.expval(H)

print(f"Ground state energy: {circuit()}")

######################################################################
# Let's use the ground state to find the first excited state.
#
# Finding the excited states
# ----------------------------
#
# To obtain the excited state we must define our ansatz that generates the state :math:`|\Psi(\theta)\rangle`.
#
# We use an ansatz constructed with Givens rotation described in `this tutorial <https://pennylane.ai/qml/demos/tutorial_givens_rotations/>`_. Let's define the circuit for finding the excited state.
#

from functools import partial

# This lines is added to better visualise the circuit
@partial(qml.devices.preprocess.decompose, stopping_condition = lambda obj:False, max_expansion=1)

def ansatz(theta, wires):
    singles, doubles = qml.qchem.excitations(2, n_qubits)
    singles = [[wires[i] for i in single] for single in singles]
    doubles = [[wires[i] for i in double] for double in doubles]
    qml.AllSinglesDoubles(theta, wires, np.array([1,1,0,0]), singles, doubles)

theta = np.random.rand(3) # 3 parameters for the ansatz
print(qml.draw(ansatz, decimals = 2)(theta, range(4)))

######################################################################
# The ``ansatz`` function is the one that generates the state :math:`|\Psi(\theta)\rangle`.
# The next step is to calculate the overlap between our generated state and the ground state, using a technique
# known as `SWAP test <https://en.wikipedia.org/wiki/Swap_test>`__.


@qml.qnode(dev)
def swap_test(params):
    generate_ground_state(range(1, n_qubits + 1))
    ansatz(params, range(n_qubits + 1, 2 * n_qubits + 1))

    qml.Barrier()  # added to better visualise the circuit
    qml.Hadamard(wires=0)
    for i in range(n_qubits):
        qml.CSWAP(wires=[0, 1 + i + n_qubits, 1 + i])
    qml.Hadamard(wires=0)
    return qml.expval(qml.Z(0))

print(qml.draw(swap_test)(theta))
print(f"\nOverlap between the ground state and the ansatz: {swap_test(theta)}")

######################################################################
# The ``swap_test`` function returns the overlap between the generated state and the ground state.
# In this demo we will not go deeper into this technique but we encourage the reader to explore it further.
#
# With this we have all the ingredients to define the loss function that we want to minimize:
#

@qml.qnode(dev)
def expected_value(theta):
    ansatz(theta, range(n_qubits))
    return qml.expval(H)

def loss_f(theta, beta):
    return expected_value(theta) + beta * swap_test(theta)

######################################################################
# The ``loss_f`` function returns the value of the cost function.
# The next step is to optimize the parameters of the ansatz to minimize the cost function.

import jax
import optax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
print()

theta = jax.numpy.array([0.1, 0.2, 0.3])
beta = 2

# Store the values of the cost function
energy = [loss_f(theta, beta)]

conv_tol = 1e-6
max_iterations = 100

opt = optax.sgd(learning_rate=0.4)

# Store the values of the circuit parameter
angle = [theta]

opt_state = opt.init(theta)

for n in range(max_iterations):
    gradient = jax.grad(loss_f)(theta, beta)
    updates, opt_state = opt.update(gradient, opt_state)
    theta = optax.apply_updates(theta, updates)
    angle.append(theta)
    energy.append(loss_f(theta, beta))

    conv = jax.numpy.abs(energy[-1] - energy[-2])

    if n % 5 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print(f"\nEstimated energy: {energy[-1].real:.8f}")

######################################################################
# Great! We have found a new energy value, but is this the right one?
# One way to check is to access the eigenvalues of the Hamiltonian directly:

print(np.sort(np.linalg.eigvals(H.matrix())))

######################################################################
# We have indeed found an eigenvalue of the Hamiltonian. It may seem that we have skipped the value :math:`-0.5389`,
# however the eigenvector corresponding to this eigenvalue belongs to a different particle number sector.
# The correct energy value for the first excited state of hydrogen is :math:`-0.53320939`, consistent with what we obtained with VQD!
# We have successfully found the first excited state!
#
# Conclusion
# ----------
#
# In this tutorial we delved into the capabilities of Variational Quantum Deflation (VQD) using PennyLane to compute not only the ground
# state but also the excited states of a hydrogen molecule.
# VQD is a variational method for calculating low-level excited state energies of quantum systems. Leveraging the
# orthogonality of the eigenstates, it adds a regularization penalty to the cost function to encourage the search for
# the next excited state from the ground state discovered by VQE.
# This illustrated how advanced quantum algorithms can extend beyond basic applications,
# offering deeper insights into quantum systems. We invite you to continue exploring these techniques and find more interesting use cases.
#
# References
# ----------
#
# .. [#Vqd]
#
#     Higgott, Oscar and Wang, Daochen and Brierley, Stephen
#     "Variational Quantum Computation of Excited States"
#     `Quantum 3, 156 (2019).: <https://dx.doi.org/10.22331/q-2019-07-01-156>`__.
#
# About the authors
# -----------------
#
