r"""Ground State and Excited State of H2 Molecule using VQE and VQD
===============================================================

Understanding the ground state and excited state energies of quantum systems is paramount in various scientific fields. The ground state energy represents the lowest energy configuration of a system, crucial for predicting its stability, chemical reactivity, and electronic properties. Excited state energies, on the other hand, reveal the system's potential for transitions to higher energy levels, essential in fields like spectroscopy, materials science, and quantum computing. Both ground and excited state energies provide insights into fundamental properties of matter, guiding research in diverse areas such as drug discovery, semiconductor physics, and renewable energy technologies.

In this demo, we solve this problem by employ two quantum algorithms, the Variational Quantum Eigensolver (VQE)
and the Variational Quantum Deflation (VQD). VQE offers a powerful tool for accurately determining ground state energies of quantum systems to find the ground state and VQD builds upon the result of VQE to find the energy of the excited state
:math:`H_2` molecule.
"""

import jax
import optax
import pennylane as qml
from pennylane import numpy as np

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

h2_dataset = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")
h2 = h2_dataset[0]
H, qubits = h2.hamiltonian, len(h2.hamiltonian.wires)
print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)

######################################################################
# The `hf_state` will contain the orbital config with the lowest energy. Let's see what it is
#
h2.hf_state


######################################################################
# Setting expectation for VQE and VQD
# -------------------------------------------
#
# Before any training takes place, let’s first look at some of the empirical measured value.
# Thankfully, :math:`H_2` is well studied
#
# - Ground state energy:
#     - :math:`H` atom: # :math:`E_1=-13.6eV`
#     - :math:`H_2` molecule: :math:`-1.136*27.21 Ha=-30.91 eV`
#
# - 1st level excitation
#     - The energy for :math:`H` atom: :math:`E_2=\frac{-13.6}{4}=-3.4eV`
#     - The energy to transition from # :math:`E_1` to :math:`E_2` for :math:`H` atom: :math:`10.2eV`
#


def hatree_energy_to_ev(hatree: float):
    return hatree * 27.2107


######################################################################
# Just like training a neural network, the VQE needs two ingredients to make it works. First we need to define
# an Ansatz (which plays the role of the neural network), then a loss function.
#

######################################################################
# Ansatz
# ------
#
# Before any run, we can assume that the Jordan Wigner representation ``[1 1 0 0]`` has the lowest
# energy. Let’s calculate that energy. Since we are studying the excitement, the ansatz is the Given rotation.
# circuit as below.
#

dev = qml.device("default.qubit", wires=qubits)


@qml.qnode(dev)
def circuit_expected():
    qml.BasisState(h2.hf_state, wires=range(qubits))
    for op in h2.vqe_gates:
        qml.apply(op)
    return qml.probs(), qml.state(), qml.expval(H)


print(f"HF state: {h2.hf_state}")
prob, state, expval = circuit_expected()
print(f"Ground state energy H_2: {expval}")

print(hatree_energy_to_ev(expval))

print(qml.draw(circuit_expected)())


######################################################################
# We would define the same circuit but without the :math:`\theta`. Given 2 :math:`H` and 4 qubits,
# after a double excitation, the HF is the superposition of the states
#
# .. math:: \alpha|1100\rangle+\beta|0011\rangle:=\cos(\theta)|1100\rangle-\sin(\theta)|0011\rangle
#


@qml.qnode(dev, diff_method="backprop")
def circuit(param):
    qml.BasisState(h2.hf_state, wires=range(qubits))
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    return qml.state(), qml.expval(H)


######################################################################
# Define the lost function
# ------------------------
#
# Remember that the lost function is the second ingredient. We use the first two equations in `this
# paper <https://www.nature.com/articles/s41524-023-00965-1>`
# .. math::`\begin{align}
# C_0\left( {{{\mathbf{\theta }}}} \right) &= \left\langle {{\Psi}\left( {{{\mathbf{\theta }}}} \right)\left| {\hat H} \right|{\Psi}\left( {{{\mathbf{\theta }}}} \right)} \right\rangle \label{eq:loss_1} \tag{1} \\
# C_1\left( {{{\mathbf{\theta }}}} \right) &= \left\langle {{\Psi}\left( {{{\mathbf{\theta }}}} \right)\left| {\hat H} \right|{\Psi}\left( {{{\mathbf{\theta }}}} \right)} \right\rangle + \beta \left| {\left\langle {{\Psi}\left( {{{\mathbf{\theta }}}} \right)\left| {{\Psi}_0} \right.} \right\rangle } \right|^2 \label{eq:loss_2} \tag{2}
# \end{align}`
#
# We can then define a lost function using the VQE and VQD methods
#
# At first sight, it might raise some eyebrows for someone from an ML background, because we
# define the loss function based on the predicted and the ground truth. However, note that we do not have any
# ground truth value here. In this context, a loss function is just a function that we want to
# minimize.
#
# Now we proceed to optimize the variational parameters. Note that :math:`\eqref{eq:loss_1}` has
# been implemented in ``circuit()``. For the term
# :math:`\beta \left| {\left\langle {{\Psi}\left( {{{\mathbf{\theta }}}} \right)\left| {{\Psi}_0} \right.} \right\rangle } \right|^2`
# in equation :math:`\eqref{eq:loss_2}`, there is no straight-forward method to compute it
# directly in a quantum machine. To make everything pure quantum, we rely on a swap test as below
#

dev_swap = qml.device("default.qubit", wires=qubits * 2 + 1)


@qml.qnode(dev_swap)
def circuit_loss_2(param, theta_0):
    """
    Constructs a quantum circuit for the variational quantum deflation (VQD) calculation to optimize for theta.

    Args:
    param (float): Rotation angle for the Double Excitation gate, to be optimized.
    theta_0 (float): The rotation angle corresponding to ground energy.

    Returns:
    tuple: A tuple containing two quantum measurements:
        - Expected value of the Hamiltonian (H) operator.
        - Probability distribution of measurement outcomes on the 8th wire.

    The circuit consists of operations to prepare the initial states for the excited and ground states of H_2,
    apply the Double Excitation gate with the provided parameters, perform a Hadamard gate operation on wire 8,
    and then execute controlled-swap (CSWAP) gates between wire 8 and wires 0 to (qubits-1) and (qubits) to (2*qubits-1).
    Finally, another Hadamard gate is applied on wire 8.

    Note:
    - The Hamiltonian reserves wires 0 to 3 for the excited state calculation and wires 4 to 7 for the ground state of H_2.
    - Wire 8 is reserved for the Hadamard gate operation.

    If psi and phi are orthogonal (|⟨psi|phi⟩|^2 = 1), the probability that 0 is measured is 1/2.
    If the states are equal (|⟨psi|phi⟩|^2 = 1), the probability that 0 is measured is 1.
    The measurement on the 0th wire, or 1st qubit, is 0.5 + 0.5(|⟨psi|phi⟩|^2).
    """
    qml.BasisState(h2.hf_state, wires=range(0, qubits))
    qml.BasisState(h2.hf_state, wires=range(qubits, qubits * 2))
    qml.DoubleExcitation(param, wires=range(0, qubits))
    qml.DoubleExcitation(theta_0, wires=range(qubits, qubits * 2))
    qml.Hadamard(8)
    for i in range(0, qubits):
        qml.CSWAP([8, i, i + qubits])
    qml.Hadamard(8)
    return qml.expval(H), qml.probs(8)


######################################################################
# Let’s preview the circuit...
#

print(qml.draw(circuit_loss_2)(param=0, theta_0=1))

######################################################################
# ... and define the loss functions
#


def loss_fn_1(theta):
    _, expval = circuit(theta)
    return expval


def loss_fn_2(theta, theta_0, beta):
    expval, measurement = circuit_loss_2(theta, theta_0)
    return expval + beta * (measurement[0] - 0.5) / 0.5


def optimize(loss_f, **kwargs):
    theta = np.array(0.0)

    # store the values of the cost function
    energy = [loss_fn_1(theta)]
    conv_tol = 1e-6
    max_iterations = 100
    opt = optax.sgd(learning_rate=0.4)

    # store the values of the circuit parameter
    angle = [theta]

    opt_state = opt.init(theta)

    for n in range(max_iterations):
        gradient = jax.grad(loss_f)(theta, **kwargs)
        updates, opt_state = opt.update(gradient, opt_state)
        theta = optax.apply_updates(theta, updates)

        angle.append(theta)
        energy.append(loss_fn_1(theta))

        conv = np.abs(energy[-1] - energy[-2])

        if n % 5 == 0:
            print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break
    return angle[-1], energy[-1]


######################################################################
# We now have all we need to run the ground state and 1st excited state optimization
#

ground_state_theta, ground_state_energy = optimize(loss_fn_1)

######################################################################
# For the excited state, we are going to choose the value for :math:`\beta`, such that :math:`\beta > E_1 - E_0`. In
# other word, :math:`\beta` needs to be larger than the gap between the ground state energy and the
# first excited state energy.
#

beta = 5

first_excite_theta, first_excite_energy = optimize(loss_fn_2, theta_0=ground_state_theta, beta=beta)

hatree_energy_to_ev(ground_state_energy), hatree_energy_to_ev(first_excite_energy)

######################################################################
# The result should produce something close to the first ionization energy of :math:`H_2` is
# :math:`1312.0 kJ/mol` according to Wikipedia. We now see how close the result is to reality.
#

kj_per_mol_per_hatree = 2625.5
ground_truth_in_kj_per_mol = 1312
prediction_in_kj_per_mol = first_excite_energy * kj_per_mol_per_hatree

error = np.abs(prediction_in_kj_per_mol - ground_truth_in_kj_per_mol)

print(
    f"The result is {error} kJ/mol different from reality, or {100 - (prediction_in_kj_per_mol / ground_truth_in_kj_per_mol * 100)} percent"
)

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/minh_chau.txt
