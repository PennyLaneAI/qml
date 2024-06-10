r"""Calculate the excited state energy with VQD
===============================================================

Understanding the ground state and excited state energies of quantum systems is paramount in various scientific fields. The **ground state energy** represents the lowest energy configuration of a system, crucial for predicting its stability, chemical reactivity, and electronic properties. **Excited state energies**, on the other hand, reveal the system's potential for transitions to higher energy levels. Both ground and excited state energies provide insights into fundamental properties of matter, guiding research in diverse areas such as drug discovery, semiconductor physics, and renewable energy technologies.

In this demo, we find the first excitation energy of Hydrogen using the ground state energy combined with Variational Quantum Deflation algorithm [#Vqd]_ . To benefit the most from this tutorial, we recommend a familiarization with the `VQE tutorial from Pennylane <https://pennylane.ai/qml/demos/tutorial_vqe/>`_.
"""

######################################################################
# Defining the Hydrogen molecule
# -------------------------------------------
# The `datasets` package from Pennylane makes it a breeze to find the Hamiltonian and the ground state
# of some molecules, which fortunately contain our molecule of interest.
# Let's see how we can build the ground state in a simple way:
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
# The ``generate_ground_state`` function prepares the ground state of the molecule.
# Let's use it to check the energy of that state:
#

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    generate_ground_state(range(n_qubits))
    return qml.expval(H)

print(f"Ground state energy: {circuit()}")

######################################################################
# This dataset do not currently offer a set of operators to find higher energy states so we will show how to do this
# through the technique known as VQD.
#
# Variational Quantum Deflation
# -------------------------------
# The Variational Quantum Deflation (VQD) algorithm [#Vqd]_ is a method to find excited states of a quantum system using the ground state energy.
# The algorithm is based on the Variational Quantum Eigensolver (VQE) algorithm, which finds the ground state energy of a quantum system.
# The idea of VQE is to define an ansatz that depends on some :math:`\theta` parameters and minimize the function:
#
# .. math:: C_0(\theta) = \left\langle\Psi(\theta)|\hat H |\Psi (\theta) \right\rangle.
#
# However, this is not enough if we are not looking for the ground state energy.
# We must find a function whose minimum is no longer the ground state and becomes the next excited state.
# This is possible just by adding a penalty term to the above function:
#
# .. math:: C_1(\theta) = \left\langle\Psi(\theta)|\hat H |\Psi (\theta) \right\rangle + \beta | \left\langle \Psi (\theta)| \Psi_0 \right\rangle|^2,
#
# where :math:`\beta` is a hyperparameter that controls the penalty term and :math:`| \Psi_0 \rangle` is the ground state.
# The function is still trying to minimize the energy but we are penalizing states that close to the ground state.
# This works thanks to the orthogonality that exists between the eigenvectors of an operator.
#
# From a physics perspective, :math:`\beta` should be larger than the energy gap between the excitement levels.
# In addition, we could iteratively calculate the excited :math:`k`-th states by adding the similarity penalty to the previous :math:`k - 1` excitation states.
#
# As easy as that! Let's see how we can run this on PennyLane
#
# VQD in Pennylane
# ----------------
#
# After nailing the theory down, first we must define our ansatz that generates state :math:`|\Psi(\theta)\rangle`.
#
# We are going to choose a particularly useful ansatz to simulate the promoting to higher orbitals of electrons,
# the Givens rotation ansatz, which you can find
# described on `this tutorial <https://pennylane.ai/qml/demos/tutorial_givens_rotations/>`_. Let's define the circuit for finding the excited state.
#

from functools import partial

# This lines is for drawing porpuses
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
# The next step is to calculate the overlap between our generated state and the ground state, using a technique known as SWAP test.


@qml.qnode(dev)
def swap_test(params):
    generate_ground_state(range(1, n_qubits + 1))
    ansatz(params, range(n_qubits + 1, 2 * n_qubits + 1))

    qml.Barrier()
    qml.Hadamard(wires=0)
    for i in range(n_qubits):
        qml.CSWAP(wires=[0, 1 + i + n_qubits, 1 + i])
    qml.Hadamard(wires=0)
    return qml.expval(qml.Z(0))

print(qml.draw(swap_test)(theta))
print(f"\nOverlap between the ground state and the ansatz: {swap_test(theta)}")

######################################################################
# The ``swap_test`` function return the overlap between the generated state and the ground state.
# In this demo we will not go deeper into this technique but we encourage the reader to perform the calculations
# since it is a very didactic exercise.
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

# store the values of the cost function
energy = [loss_f(theta, beta)]

conv_tol = 1e-6
max_iterations = 100

opt = optax.sgd(learning_rate=0.4)

# store the values of the circuit parameter
angle = [theta]

opt_state = opt.init(theta)

for n in range(max_iterations):
    gradient = jax.grad(loss_f)(theta, beta)
    updates, opt_state = opt.update(gradient, opt_state)
    theta = optax.apply_updates(theta, updates)
    angle.append(theta)
    energy.append(loss_f(theta, beta))

    conv = jax.numpy.abs(energy[-1] - energy[-2])

    if n % 10 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print(f"\nEstimated energy: {energy[-1].real:.8f}")

######################################################################
# Great! We have found a new energy value, but is this the right one?
# One way to check is to access the eigenvalues of the Hamiltonian directly:

first_excitation = np.sort(np.linalg.eigvals(H.matrix()))[1]
print(f"First excitation energy: {first_excitation.real:.8f}")

######################################################################
#
# Conclusion
# ----------
# We have used VQD to find the excited state of the :math:`H_2` molecule. One of the applications is
# in photovoltaic devices. For example, the design of solar cells relies on optimizing the energy levels of donor and acceptor
# materials to facilitate charge separation and collection, thereby enhancing solar energy conversion efficiency.
#
# To build up on this work, we recommend readers to run this script with more complex molecules and/or find the energy needed for
# higher excitation levels. Also do not forget check out other tutorials for Quantum chemistry here in Pennylane. Good luck on your Quantum chemistry journey!
#
# Acknowledgement
# ----------
# The authors is grateful Soran Jahangiri for his comments.
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
# .. [#vqe]
#
#     Peruzzo, Alberto and McClean, Jarrod and Shadbolt, Peter and Yung, Man-Hong and Zhou, Xiao-Qi and Love, Peter J. and Aspuru-Guzik, Alán and O’Brien, Jeremy L.
#     "A variational eigenvalue solver on a photonic quantum processor"
#     `Nature Communications 5, 1 (2014).: <http://dx.doi.org/10.1038/ncomms5213>`__.
#
# About the author
# ----------------
# .. include:: ../_static/authors/minh_chau.txt
#
